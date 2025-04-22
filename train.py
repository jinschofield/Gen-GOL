import os
import torch
from torch.utils.data import DataLoader
from data.dataset import GolDataset
from models.unet import UNet
from models.diffusion import Diffusion
import argparse

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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = GolDataset(data_dir=args.data_dir, augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = UNet().to(args.device)
    diffusion = Diffusion(timesteps=args.timesteps, device=args.device,
                          live_weight=args.live_weight, schedule=args.schedule,
                          ssim_weight=args.ssim_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            step += 1
            optimizer.zero_grad()
            x = batch.to(args.device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=args.device).long()
            loss = diffusion.p_losses(model, x, t)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            if step % 1000 == 0:
                ckpt = os.path.join(args.save_dir, f"model_{step}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"Saved checkpoint {ckpt}")

    final_ckpt = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Model saved to {final_ckpt}")

if __name__ == '__main__':
    main()
