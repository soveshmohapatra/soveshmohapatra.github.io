import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import math

from diffusion import LinearNoiseSchedule, DDPM

# ---------------------------------------------------------------------------
# Sinusoidal Time Embedding
# ---------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ---------------------------------------------------------------------------
# Simple UNet for 28x28 MNIST
# ---------------------------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2) # 28 -> 14
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2) # 14 -> 7
        
        # Bottleneck
        self.bot_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.time_inject = nn.Linear(time_dim, 128)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 7 -> 14
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 64 + 64 (skip) = 128
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # 14 -> 28
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 + 32 (skip) = 64
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, time):
        # Time embedding
        t = self.time_mlp(time)
        
        # Encoder
        x1 = nn.functional.gelu(self.conv1(x))
        x2 = self.pool1(x1)
        x2 = nn.functional.gelu(self.conv2(x2))
        x3 = self.pool2(x2)
        
        # Bottleneck
        x3 = nn.functional.gelu(self.bot_conv(x3))
        # Add time embedding
        time_emb = self.time_inject(t)[:, :, None, None]
        x3 = x3 + time_emb
        
        # Decoder
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1) # skip connection
        x = nn.functional.gelu(self.conv3(x))
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1) # skip connection
        x = nn.functional.gelu(self.conv4(x))
        
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Training Loop & Sampling
# ---------------------------------------------------------------------------
def train_ddpm(epochs=20, batch_size=128, lr=2e-4, num_timesteps=1000):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # scale to [-1, 1]
    ])
    
    # Download MNIST to ./data
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = SimpleUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet Parameters: {total_params:,}")
    
    schedule = LinearNoiseSchedule(num_timesteps=num_timesteps)
    ddpm = DDPM(schedule)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            loss = ddpm.compute_loss(model, images, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")
        
    return model, ddpm, loss_history

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    NUM_TIMESTEPS = 1000
    model, ddpm, losses = train_ddpm(epochs=15, num_timesteps=NUM_TIMESTEPS)
    
    device = next(model.parameters()).device
    model.eval()
    
    print("\nGenerating final samples grid...")
    # Generate 32 samples (4x8 grid)
    final_samples, _ = ddpm.sample(model, (32, 1, 28, 28), device=device)
    # Scale back to [0, 1]
    final_samples = (final_samples + 1.0) / 2.0
    final_samples = torch.clamp(final_samples, 0.0, 1.0)
    
    grid = make_grid(final_samples, nrow=8)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('generate_grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("Generating denoising trajectory...")
    # Generate 1 sample tracking intermediates
    _, intermediates = ddpm.sample(model, (1, 1, 28, 28), device=device)
    
    plt.figure(figsize=(15, 3))
    # Pick ~8 intermediate steps
    num_steps = len(intermediates)
    indices = torch.linspace(0, num_steps - 1, steps=8).long().tolist()
    
    for i, idx in enumerate(indices):
        plt.subplot(1, 8, i + 1)
        img = intermediates[idx][0, 0].numpy()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("T=1000\n(Noise)")
        elif i == 7:
            plt.title("T=0\n(Data)")
        else:
            t_val = NUM_TIMESTEPS - int((idx / (num_steps-1)) * (NUM_TIMESTEPS))
            plt.title(f"T={t_val}")
            
    plt.tight_layout()
    plt.savefig('denoising_trajectory.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print("Plotting training loss...")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, color='#1565C0', lw=2)
    plt.title('DDPM Training Loss (MSE)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Epsilon Prediction)')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png')
    plt.close()
    
    print("Done! Generated generate_grid.png, denoising_trajectory.png, and training_loss.png")
