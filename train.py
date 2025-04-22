import os
import torch
from torch.utils.data import DataLoader
from data.dataset import GolDataset
from models.unet import UNet
from models.diffusion import Diffusion
import argparse
import copy
from torch.nn.utils import clip_grad_norm_

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
    parser.add_argument('--ramp_steps', type=int, default=0, help='steps over which to ramp SSIM/BCE weights from 0 to final')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max norm for gradient clipping; 0 disables')
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none','cosine'], help='learning rate scheduler')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='EMA decay for model weights')
    parser.add_argument('--noise_prob', type=float, default=0.0, help='probability of random cell flips as data augmentation')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = GolDataset(data_dir=args.data_dir, augment=True, noise_prob=args.noise_prob)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = UNet().to(args.device)
    # save initial (untrained) model for baseline comparison
    init_ckpt = os.path.join(args.save_dir, "model_init.pt")
    torch.save(model.state_dict(), init_ckpt)
    print(f"Saved initial untrained model to {init_ckpt}")

    # initialize EMA model
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device,
                          live_weight=args.live_weight, schedule=args.schedule,
                          ssim_weight=args.ssim_weight, bce_weight=args.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema_decay = args.ema_decay

    # setup learning rate scheduler if requested
    scheduler = None
    if args.lr_scheduler == 'cosine':
        total_steps = args.epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    step = 0
    for epoch in range(args.epochs):
        for x, c in dataloader:
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
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=args.device).long()
            loss = diffusion.p_losses(model, x, t, c)
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
