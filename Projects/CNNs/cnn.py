import torch
import torch.nn as nn
import torch.nn.functional as F
from convolution import Conv2D, MaxPool2D, Flatten, ActivationFunctions


class LeNet5(nn.Module):
    """
    LeNet-5 Architecture (1998)
    One of the earliest successful CNNs, designed for handwritten digit recognition.
    
    Architecture:
    - Convolutional layers with average pooling
    - Fully connected layers
    - Output classification
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        
        # Feature extractor
        self.conv1 = Conv2D(in_channels, 6, kernel_size=5, padding=2)  # 32x32 -> 32x32
        self.pool1 = AveragePool2D(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        self.conv2 = Conv2D(6, 16, kernel_size=5)  # 16x16 -> 12x12
        self.pool2 = AveragePool2D(kernel_size=2, stride=2)  # 12x12 -> 6x6
        
        # Classifier
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # Convolutional block 1
        x = ActivationFunctions.tanh(self.conv1(x))
        x = self.pool1(x)
        
        # Convolutional block 2
        x = ActivationFunctions.tanh(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = Flatten()(x)
        
        # Fully connected layers
        x = ActivationFunctions.tanh(self.fc1(x))
        x = ActivationFunctions.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification.
    Demonstrates the classic Conv -> ReLU -> Pool pattern.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = Conv2D(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assumes 28x28 input
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1: Conv -> ReLU -> Pool
        x = ActivationFunctions.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2: Conv -> ReLU -> Pool
        x = ActivationFunctions.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = Flatten()(x)
        
        # Classifier
        x = self.dropout(x)
        x = ActivationFunctions.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNBlock(nn.Module):
    """
    A reusable CNN block with Conv -> BatchNorm -> ReLU -> Pool pattern.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 pool: bool = True):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = MaxPool2D(kernel_size=2, stride=2) if pool else nn.Identity()
    
    def forward(self, x):
        x = ActivationFunctions.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x


class DeepCNN(nn.Module):
    """
    A deeper CNN architecture with multiple convolutional blocks.
    Inspired by VGG-style architecture design.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1, 
                 num_blocks: int = 4, initial_channels: int = 32):
        super().__init__()
        
        # Build convolutional blocks
        blocks = []
        channels = initial_channels
        
        for i in range(num_blocks):
            blocks.append(CNNBlock(in_channels if i == 0 else channels, channels, pool=(i < num_blocks - 1)))
            if i < num_blocks - 1:
                channels *= 2  # Double channels after each block
        
        self.features = nn.Sequential(*blocks)
        
        # Classifier head
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(channels * 3 * 3, 256),  # Assumes 28x28 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
