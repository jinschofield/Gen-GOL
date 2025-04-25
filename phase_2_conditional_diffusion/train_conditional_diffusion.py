#!/usr/bin/env python3
"""
Phase 2 - Step 2:
Train a conditional diffusion UNet on 32×32 patterns with class embeddings.
Produces checkpoints in `phase2_results/`.
"""
import os, sys, argparse, random
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
# ensure repo root on import path
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'phase_1_classification', 'utils'))

import torch
from torch.utils.data import DataLoader
from phase_2_conditional_diffusion.category_dataset import CategoryDataset
from models.unet import UNet
from models.diffusion import Diffusion

def main():
    parser = argparse.ArgumentParser(description='Train conditional diffusion model')
    parser.add_argument('--label_csv', type=str, required=True,
                        help='CSV of [filepath, category] labels')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--cf_prob', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--fm_weight', type=float, default=0.0)
    parser.add_argument('--adv_weight', type=float, default=0.0)
    parser.add_argument('--mae_weight', type=float, default=0.0)
    parser.add_argument('--noise_prob', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none','cosine'])
    parser.add_argument('--bce_weight', type=float, default=0.0)
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['linear','cosine'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = 'phase2_results'
    os.makedirs(out_dir, exist_ok=True)

    # dataset and loader
    dataset = CategoryDataset(args.label_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

    # model and diffusion (pass dynamic number of classes to UNet)
    num_classes = len(dataset.cat_to_idx)
    model = UNet(dropout=args.dropout, num_classes=num_classes).to(args.device)
    diffusion = Diffusion(
        timesteps=args.timesteps,
        schedule=args.schedule,
        ssim_weight=args.ssim_weight,
        bce_weight=args.bce_weight,
        mae_weight=args.mae_weight,
        guidance_scale=args.guidance_scale,
        device=args.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # scheduler
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

    # resume if requested
    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state)
        print(f"Resumed from checkpoint {args.resume}")

    # training loop
    for epoch in range(args.epochs):
        model.train()
        for x, c in loader:
            x = x.to(args.device)
            c = c.to(args.device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),),
                              device=args.device)
            loss = diffusion.p_losses(model, x, t, c)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
            optimizer.step()
        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
        # checkpoint every 50 epochs
        if (epoch+1) % 50 == 0:
            ckpt = os.path.join(out_dir, f"cond_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved {ckpt}")

    # final save
    final_ckpt = os.path.join(out_dir, "cond_model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Model saved to {final_ckpt}")

if __name__ == '__main__':
    main()
