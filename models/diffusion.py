import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_acp * x_start + sqrt_om_acp * noise

    def p_losses(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1,1,1,1)
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t))
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_var = self.posterior_variance[t].view(-1,1,1,1)
            return model_mean + torch.sqrt(posterior_var) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, model, shape):
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x
