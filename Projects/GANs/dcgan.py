"""
DCGAN — Deep Convolutional GAN from scratch in pure PyTorch.

Architecture follows Radford et al. (2015) adapted for MNIST (28x28).
Key principles:
    - Replace pooling with strided convolutions (D) and transposed convolutions (G)
    - Use BatchNorm in both G and D (except D input layer and G output layer)
    - Use ReLU in G, LeakyReLU in D
    - Tanh output in G, Sigmoid output in D
"""

import torch
import torch.nn as nn


def weights_init(m):
    """Custom weight initialization: Normal(mean=0, std=0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class DCGenerator(nn.Module):
    """
    DCGAN Generator using transposed convolutions.

    Projects noise z into a spatial feature map, then upsamples
    via ConvTranspose2d layers to produce a 28x28 image.

    Architecture:
        Linear(z_dim -> 256*7*7) -> Reshape to (256, 7, 7)
        ConvTranspose2d(256, 128, 4, 2, 1) -> BatchNorm2d -> ReLU   => (128, 14, 14)
        ConvTranspose2d(128, 64, 4, 2, 1)  -> BatchNorm2d -> ReLU   => (64, 28, 28)
        ConvTranspose2d(64, 1, 3, 1, 1)    -> Tanh                  => (1, 28, 28)
    """

    def __init__(self, z_dim=100, img_channels=1):
        super(DCGenerator, self).__init__()
        self.z_dim = z_dim

        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )

        # Transposed convolution layers
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (64, 28, 28)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1),  # -> (1, 28, 28)
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
        out = self.project(z)
        out = out.view(out.size(0), 256, 7, 7)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    """
    DCGAN Discriminator using strided convolutions.

    Takes a 28x28 image and classifies it as real or fake.

    Architecture:
        Conv2d(1, 64, 4, 2, 1)   -> LeakyReLU(0.2)               => (64, 14, 14)
        Conv2d(64, 128, 4, 2, 1) -> BatchNorm2d -> LeakyReLU(0.2) => (128, 7, 7)
        Flatten -> Linear(128*7*7, 1) -> Sigmoid
    """

    def __init__(self, img_channels=1):
        super(DCDiscriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            # No BatchNorm on first layer (per DCGAN paper)
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
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
        features = self.conv_blocks(img)
        return self.classifier(features)
