import torch
import torch.nn.functional as F
from typing import Optional
from torchmetrics.functional import structural_similarity_index_measure
import warnings

class Diffusion:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02,
                 device='cpu', live_weight=1.0, schedule: Optional[str]='linear',
                 ssim_weight: float=0.0, bce_weight: float=0.0, mae_weight: float=0.0):
        """
        Args:
            live_weight: weight multiplier for loss on live cells (>1 to emphasize alive transitions)
            ssim_weight: weight multiplier for structural SSIM loss
            bce_weight: weight multiplier for binary cross-entropy loss on x0
            mae_weight: weight multiplier for L1 (MAE) loss on noise prediction
        """
        self.timesteps = timesteps
        self.device = device
        # choose noise schedule: 'linear' or 'cosine'
        if schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps, device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # loss weight for live cells
        self.live_weight = live_weight
        # structural SSIM loss weight
        self.ssim_weight = ssim_weight
        # binary cross-entropy loss weight on x0 reconstruction
        self.bce_weight = bce_weight
        # weight for L1 (MAE) loss on noise prediction
        self.mae_weight = mae_weight
        self.schedule = schedule

    @staticmethod
    def cosine_beta_schedule(timesteps: int, device: str='cpu', s: float=0.008):
        # cosine schedule as in Nichol & Dhariwal
        steps = timesteps
        t = torch.linspace(0, steps, steps+1, device=device) / steps
        alphas_cumprod = torch.cos(((t + s)/(1 + s)) * torch.pi/2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = []
        for i in range(1, len(alphas_cumprod)):
            prev = alphas_cumprod[i-1]
            beta = min(1 - alphas_cumprod[i]/prev, 0.999)
            betas.append(beta)
        return torch.tensor(betas, dtype=torch.float32, device=device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_acp * x_start + sqrt_om_acp * noise

    def p_losses(self, model, x_start, t, c=None):
        # sample noise and noised input
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t, c)
        # per-pixel MSE
        loss = (pred_noise - noise).pow(2)
        # emphasize live cells if weight>1
        if self.live_weight != 1.0:
            # x_start is binary 0/1
            mask = (x_start > 0.5).float()
            w = 1 + (self.live_weight - 1) * mask
            loss = loss * w
        loss = loss.mean()
        # optional MAE loss on noise prediction
        if self.mae_weight > 0.0:
            mae_loss = (pred_noise - noise).abs().mean()
            loss = loss + self.mae_weight * mae_loss
        # reconstruct x0_pred for SSIM/BCE if needed
        if self.ssim_weight > 0 or self.bce_weight > 0:
            sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
            sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
            x0_pred = (x_noisy - sqrt_om_acp * pred_noise) / sqrt_acp
        # add structural SSIM loss on reconstructed x0
        if self.ssim_weight > 0:
            ssim_val = structural_similarity_index_measure(x0_pred, x_start, data_range=1.0)
            ssim_loss = torch.mean(1.0 - ssim_val)
            loss = loss + self.ssim_weight * ssim_loss
        # add BCE loss on x0 if requested
        if self.bce_weight > 0:
            bce = F.binary_cross_entropy(x0_pred.clamp(0.0,1.0), x_start)
            loss = loss + self.bce_weight * bce
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, c=None):
        betas_t = self.betas[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1,1,1,1)
        eps = model(x, t, c)
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps)
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_var = self.posterior_variance[t].view(-1,1,1,1)
            return model_mean + torch.sqrt(posterior_var) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, shape, c=None):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, c)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, eta: float = 0.0, c=None):
        """Deterministic DDIM sampling (eta=0 for deterministic)."""
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = model(x, t, c)
            alpha_t = self.alphas_cumprod[t].view(-1,1,1,1)
            alpha_prev = self.alphas_cumprod_prev[t].view(-1,1,1,1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            # predict x0
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
            if i > 0:
                x = sqrt_alpha_prev * x0_pred + torch.sqrt(1.0 - alpha_prev) * eps
            else:
                x = x0_pred
        return x
