"""
Vanilla GAN — Generator and Discriminator from scratch in pure PyTorch.

Architecture follows Goodfellow et al. (2014) adapted for MNIST (28x28).
"""

import torch
import torch.nn as nn


def weights_init(m):
    """Custom weight initialization: Normal(mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator network for vanilla GAN.

    Takes a noise vector z of dimension `z_dim` and produces
    a 28x28 grayscale image via fully-connected layers.

    Architecture:
        Linear(z_dim -> 256) -> BatchNorm1d -> ReLU
        Linear(256 -> 512)   -> BatchNorm1d -> ReLU
        Linear(512 -> 1024)  -> BatchNorm1d -> ReLU
        Linear(1024 -> 784)  -> Tanh
        Reshape to (1, 28, 28)
    """

    def __init__(self, z_dim=100, img_channels=1, img_size=28):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, z):
        """
        Args:
            z: Noise tensor of shape (batch_size, z_dim)
        Returns:
            img: Generated image of shape (batch_size, 1, 28, 28)
        """
        out = self.net(z)
        img = out.view(out.size(0), self.img_channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    """
    Discriminator network for vanilla GAN.

    Takes a 28x28 grayscale image and outputs a probability
    of the image being real.

    Architecture:
        Flatten
        Linear(784 -> 512)  -> LeakyReLU(0.2)
        Linear(512 -> 256)  -> LeakyReLU(0.2)
        Linear(256 -> 1)    -> Sigmoid
    """

    def __init__(self, img_channels=1, img_size=28):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(img_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, img):
        """
        Args:
            img: Image tensor of shape (batch_size, 1, 28, 28)
        Returns:
            prob: Probability of being real, shape (batch_size, 1)
        """
        return self.net(img)
