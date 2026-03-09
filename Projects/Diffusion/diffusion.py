import torch
import torch.nn as nn
import unittest

class LinearNoiseSchedule:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """Extract coefficients for specific timesteps t and reshape them for broadcasting."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def q_sample(self, x_start, t, noise=None):
        """Forward pass: Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.schedule.extract(
            self.schedule.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.schedule.extract(
            self.schedule.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, model, x_start, t, noise=None):
        """Compute the diffusion model loss (MSE between true noise and predicted noise)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Reverse step: Sample x_{t-1} given x_t"""
        betas_t = self.schedule.extract(self.schedule.betas.to(x.device), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.extract(
            self.schedule.sqrt_one_minus_alphas_cumprod.to(x.device), t, x.shape
        )
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.schedule.extract(self.schedule.alphas.to(x.device), t, x.shape))
        
        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.schedule.extract(self.schedule.posterior_variance.to(x.device), t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample(self, model, shape, device='cpu'):
        """Full reverse loop: Generate samples from pure noise"""
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        # Keep track of intermediate images for visualization
        intermediates = []

        for i in reversed(range(0, self.schedule.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
            if i % (self.schedule.num_timesteps // 10) == 0 or i == self.schedule.num_timesteps - 1:
                intermediates.append(x.cpu().clone())

        return x, intermediates


class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.schedule = LinearNoiseSchedule(num_timesteps=100)
        self.ddpm = DDPM(self.schedule)

    def test_schedule_values(self):
        self.assertEqual(len(self.schedule.betas), 100)
        self.assertEqual(len(self.schedule.alphas_cumprod), 100)
        self.assertTrue(torch.all(self.schedule.alphas_cumprod <= 1.0))
        self.assertTrue(torch.all(self.schedule.alphas_cumprod >= 0.0))

    def test_q_sample_shapes(self):
        x = torch.zeros(4, 1, 28, 28)
        t = torch.randint(0, 100, (4,))
        x_noisy = self.ddpm.q_sample(x, t)
        self.assertEqual(x_noisy.shape, x.shape)

    def test_q_sample_extreme_t(self):
        x = torch.zeros(1, 1, 32, 32)
        # At t=0, noise should be minimal
        t_0 = torch.tensor([0])
        x_noisy_0 = self.ddpm.q_sample(x, t_0)
        # At t=T-1, should be almost pure noise
        t_max = torch.tensor([99])
        x_noisy_max = self.ddpm.q_sample(x, t_max)
        
        self.assertTrue(torch.std(x_noisy_0) < torch.std(x_noisy_max))

if __name__ == '__main__':
    unittest.main()
