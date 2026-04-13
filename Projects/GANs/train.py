"""
Training script for Vanilla GAN and DCGAN on MNIST.

Trains both architectures on a 5000-image MNIST subset for 50 epochs,
logs losses, and generates sample grids + training curves.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

from gan import Generator, Discriminator
from dcgan import DCGenerator, DCDiscriminator

# ============================================================
# Hyperparameters
# ============================================================
Z_DIM = 100
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
BETAS = (0.5, 0.999)
SUBSET_SIZE = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Logging setup
# ============================================================
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.log")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class Logger:
    """Dual logger: writes to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)

# ============================================================
# Data loading (MNIST, 5000-image subset)
# ============================================================
print("=" * 60)
print("GANs from Scratch — Training Script")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Subset size: {SUBSET_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")
print(f"Z dimension: {Z_DIM}")
print()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1] for Tanh
])

full_dataset = datasets.MNIST(
    root=os.path.join(SAVE_DIR, "data"),
    train=True,
    download=True,
    transform=transform,
)

# Take a 5000-image subset
subset_indices = torch.randperm(len(full_dataset))[:SUBSET_SIZE]
subset = torch.utils.data.Subset(full_dataset, subset_indices)

dataloader = torch.utils.data.DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

print(f"Dataset loaded: {SUBSET_SIZE} images, {len(dataloader)} batches per epoch")
print()


# ============================================================
# Utility functions
# ============================================================
def save_sample_grid(generator, epoch, tag, nrow=4, n_samples=16):
    """Generate and save a grid of fake images."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, Z_DIM, device=DEVICE)
        fake_imgs = generator(z)
        fake_imgs = (fake_imgs + 1) / 2  # Rescale from [-1,1] to [0,1]
    grid = make_grid(fake_imgs, nrow=nrow, padding=2, normalize=False)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"{tag} — Epoch {epoch}")
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{tag}_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    generator.train()


def save_final_grid(generator, tag, nrow=8, n_samples=64):
    """Generate and save a large final grid of fake images."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, Z_DIM, device=DEVICE)
        fake_imgs = generator(z)
        fake_imgs = (fake_imgs + 1) / 2
    grid = make_grid(fake_imgs, nrow=nrow, padding=2, normalize=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"{tag} — Final Generated Samples (8x8)")
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{tag}_final_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    generator.train()


def save_training_curves(g_losses, d_losses, tag):
    """Plot and save G and D loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss", color="blue")
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{tag} — Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{tag}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Training function
# ============================================================
def train_gan(generator, discriminator, tag):
    """
    Standard GAN training loop.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        tag: String identifier for saving files (e.g., "VanillaGAN", "DCGAN")
    Returns:
        g_losses: List of average generator losses per epoch
        d_losses: List of average discriminator losses per epoch
    """
    criterion = nn.BCELoss()

    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    g_losses = []
    d_losses = []

    print(f"{'=' * 60}")
    print(f"Training {tag}")
    print(f"{'=' * 60}")
    print(f"Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    print()

    for epoch in range(1, EPOCHS + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

            # ---- Train Discriminator ----
            opt_d.zero_grad()

            # Real images
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # ---- Train Generator ----
            opt_g.zero_grad()

            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)  # G wants D to say "real"

            g_loss.backward()
            opt_g.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1

        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"[{tag}] Epoch {epoch:3d}/{EPOCHS} | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")

        # Save sample grid every 10 epochs
        if epoch % 10 == 0:
            save_sample_grid(generator, epoch, tag)

    return g_losses, d_losses


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":

    # --- Vanilla GAN ---
    gen_vanilla = Generator(z_dim=Z_DIM).to(DEVICE)
    disc_vanilla = Discriminator().to(DEVICE)

    g_losses_vanilla, d_losses_vanilla = train_gan(gen_vanilla, disc_vanilla, "VanillaGAN")

    save_final_grid(gen_vanilla, "VanillaGAN")
    save_training_curves(g_losses_vanilla, d_losses_vanilla, "VanillaGAN")

    print()

    # --- DCGAN ---
    gen_dc = DCGenerator(z_dim=Z_DIM).to(DEVICE)
    disc_dc = DCDiscriminator().to(DEVICE)

    g_losses_dc, d_losses_dc = train_gan(gen_dc, disc_dc, "DCGAN")

    save_final_grid(gen_dc, "DCGAN")
    save_training_curves(g_losses_dc, d_losses_dc, "DCGAN")

    # ============================================================
    # Final Summary
    # ============================================================
    print()
    print("=" * 60)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 60)
    print()
    print("--- Vanilla GAN ---")
    print(f"  Final G_loss: {g_losses_vanilla[-1]:.4f}")
    print(f"  Final D_loss: {d_losses_vanilla[-1]:.4f}")
    print(f"  Min G_loss:   {min(g_losses_vanilla):.4f} (Epoch {g_losses_vanilla.index(min(g_losses_vanilla)) + 1})")
    print(f"  Min D_loss:   {min(d_losses_vanilla):.4f} (Epoch {d_losses_vanilla.index(min(d_losses_vanilla)) + 1})")
    print()
    print("--- DCGAN ---")
    print(f"  Final G_loss: {g_losses_dc[-1]:.4f}")
    print(f"  Final D_loss: {d_losses_dc[-1]:.4f}")
    print(f"  Min G_loss:   {min(g_losses_dc):.4f} (Epoch {g_losses_dc.index(min(g_losses_dc)) + 1})")
    print(f"  Min D_loss:   {min(d_losses_dc):.4f} (Epoch {d_losses_dc.index(min(d_losses_dc)) + 1})")
    print()
    print("Saved artifacts:")
    print(f"  - VanillaGAN sample grids (every 10 epochs)")
    print(f"  - VanillaGAN final 8x8 grid")
    print(f"  - VanillaGAN training curves")
    print(f"  - DCGAN sample grids (every 10 epochs)")
    print(f"  - DCGAN final 8x8 grid")
    print(f"  - DCGAN training curves")
    print(f"  - Training log: train.log")
    print()
    print("=" * 60)
