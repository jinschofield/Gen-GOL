import os
import subprocess
import random
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, Subset
from data.dataset import GolDataset
from models.unet import UNet
from models.diffusion import Diffusion
from models.discriminator import Discriminator
import argparse
import copy
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--live_weight', type=float, default=1.0, help='weight multiplier for live-cell loss')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear','cosine'], help='noise schedule')
    parser.add_argument('--ssim_weight', type=float, default=1.0, help='weight multiplier for SSIM loss')
    parser.add_argument('--bce_weight', type=float, default=0.0, help='weight multiplier for BCE loss on x0')
    parser.add_argument('--mae_weight', type=float, default=0.0, help='weight multiplier for MAE (L1) loss on noise prediction')
    parser.add_argument('--ramp_steps', type=int, default=0, help='steps over which to ramp SSIM/BCE weights from 0 to final')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max norm for gradient clipping; 0 disables')
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none','cosine'], help='learning rate scheduler')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='EMA decay for model weights')
    parser.add_argument('--noise_prob', type=float, default=0.0, help='probability of random cell flips as data augmentation')
    parser.add_argument('--adv_weight', type=float, default=0.0, help='weight for adversarial loss')
    parser.add_argument('--fm_weight', type=float, default=0.0, help='weight for feature-matching loss')
    parser.add_argument('--cf_prob', type=float, default=0.1, help='probability to drop conditioning (classifier-free)')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--val_split', type=float, default=0.1, help='fraction of data for validation (0=no val)')
    parser.add_argument('--random_baseline', action='store_true',
                        help='train on random boards instead of GoL data')
    parser.add_argument('--random_baseline_samples', type=int, default=20000,
                        help='number of random boards to use when --random_baseline')
    parser.add_argument('--grid_size', type=int, default=32,
                        help='grid size for random boards when using random baseline')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='guidance scale for diffusion model')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for UNet residual blocks')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # choose dataset: GoL or random baseline
    if args.random_baseline:
        from data.random_dataset import RandomPatternDataset
        full_dataset = RandomPatternDataset(
            num_samples=args.random_baseline_samples,
            grid_size=args.grid_size
        )
    else:
        # auto-generate dataset if missing, skip if any saved .npy present
        survive_dir = os.path.join(args.data_dir, 'survive')
        die_dir = os.path.join(args.data_dir, 'die')
        patterns = glob.glob(os.path.join(args.data_dir, '**', '*.npy'), recursive=True)
        if not (patterns or (os.path.isdir(survive_dir) and os.listdir(survive_dir)
                              and os.path.isdir(die_dir) and os.listdir(die_dir))):
            print("Dataset missing; generating via generate_dataset.py...")
            subprocess.run(["python", "data/generate_dataset.py", "--data_dir", args.data_dir], check=True)
        # load full dataset
        full_dataset = GolDataset(data_dir=args.data_dir, augment=True, noise_prob=args.noise_prob)
    # validation split via Subset
    if args.val_split > 0:
        n = len(full_dataset)
        idx = list(range(n)); random.shuffle(idx)
        split = int(n * (1 - args.val_split))
        train_idx, val_idx = idx[:split], idx[split:]
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(Subset(full_dataset, val_idx),   batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = None

    model = UNet(dropout=args.dropout).to(args.device)
    # resume from checkpoint or save initial (untrained) model
    if args.resume:
        if os.path.isfile(args.resume):
            state = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(state)
            print(f"Resumed training from checkpoint {args.resume}")
        else:
            raise FileNotFoundError(f"Resume checkpoint '{args.resume}' not found.")
    else:
        init_ckpt = os.path.join(args.save_dir, "model_init.pt")
        torch.save(model.state_dict(), init_ckpt)
        print(f"Saved initial untrained model to {init_ckpt}")

    # initialize EMA model
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    diffusion = Diffusion(timesteps=args.timesteps,
                         device=args.device,
                         schedule=args.schedule,
                         guidance_scale=args.guidance_scale,
                         live_weight=args.live_weight, 
                         ssim_weight=args.ssim_weight, 
                         bce_weight=args.bce_weight,
                         mae_weight=args.mae_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_decay = args.ema_decay

    # setup adversarial/feature-matching components
    disc = Discriminator(in_channels=1).to(args.device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr)
    adv_weight = args.adv_weight
    fm_weight = args.fm_weight

    # setup learning rate scheduler if requested
    scheduler = None
    if args.lr_scheduler == 'cosine':
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    step = 0
    for epoch in range(args.epochs):
        for x, c in train_loader:
            step += 1
            # ramp SSIM/BCE weights
            if args.ramp_steps > 0:
                ramp = min(1.0, step / args.ramp_steps)
            else:
                ramp = 1.0
            diffusion.ssim_weight = args.ssim_weight * ramp
            diffusion.bce_weight = args.bce_weight * ramp
            optimizer.zero_grad()
            x = x.to(args.device)
            c = c.to(args.device)
            # classifier-free guidance: randomly drop conditioning
            if random.random() < args.cf_prob:
                c_input = None
            else:
                c_input = c
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=args.device).long()
            # compute diffusion loss
            diff_loss = diffusion.p_losses(model, x, t, c_input)
            # adversarial and feature-matching
            if adv_weight > 0 or fm_weight > 0:
                # single-step reconstruction for GAN
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise)
                pred_noise = model(x_noisy, t, c_input)
                sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(-1,1,1,1)
                sqrt_om_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
                x0_pred = (x_noisy - sqrt_om_alpha * pred_noise) / sqrt_alpha
                # train discriminator
                real = x
                fake = x0_pred.detach()
                real_prob, real_feats = disc(real)
                # detach real_feats so parameter updates don't corrupt the graph
                real_feats = [f.detach() for f in real_feats]
                fake_prob, fake_feats = disc(fake)
                loss_d = - (torch.log(real_prob + 1e-8).mean() + torch.log(1 - fake_prob + 1e-8).mean())
                disc_opt.zero_grad(); loss_d.backward(retain_graph=True); disc_opt.step()
                # generator adversarial/feature-matching losses
                fake_prob_g, fake_feats_g = disc(x0_pred)
                adv_loss = - torch.log(fake_prob_g + 1e-8).mean() if adv_weight > 0 else 0
                if fm_weight > 0 and real_feats:
                    fm_loss = sum(F.l1_loss(fg, fr) for fg, fr in zip(fake_feats_g, real_feats)) / len(real_feats)
                else:
                    fm_loss = 0
                loss = diff_loss + adv_weight * adv_loss + fm_weight * fm_loss
            else:
                loss = diff_loss
            loss.backward()
            # gradient clipping
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            # scheduler step
            if scheduler is not None:
                scheduler.step()
            # update EMA weights
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data * (1 - ema_decay))

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            if step % 1000 == 0:
                ckpt = os.path.join(args.save_dir, f"model_{step}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved checkpoint {ckpt}")
                # save EMA checkpoint
                ema_ckpt = os.path.join(args.save_dir, f"model_{step}_ema.pt")
                torch.save(ema_model.state_dict(), ema_ckpt)
                print(f"EMA model saved to {ema_ckpt}")

        # end of epoch: validation loss
        if val_loader is not None:
            model.eval()
            val_losses = []
            for xv, cv in val_loader:
                xv = xv.to(args.device); cv = cv.to(args.device)
                tv = torch.randint(0, diffusion.timesteps, (xv.size(0),), device=args.device).long()
                lv = diffusion.p_losses(model, xv, tv, cv)
                val_losses.append(lv.item())
            avg_val = np.mean(val_losses)
            print(f"Epoch {epoch+1}/{args.epochs} Validation Loss: {avg_val:.4f}")
            model.train()

    # save final model
    final_ckpt = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Model saved to {final_ckpt}")
    # save EMA model
    ema_ckpt = os.path.join(args.save_dir, "model_final_ema.pt")
    torch.save(ema_model.state_dict(), ema_ckpt)
    print(f"EMA model saved to {ema_ckpt}")

if __name__ == '__main__':
    main()
