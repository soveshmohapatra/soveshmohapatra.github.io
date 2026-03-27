import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    """
    Implements a 2D convolution operation from scratch.
    Applies a learnable filter kernel that slides across the input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Learnable convolution kernels
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Convolved tensor of shape (batch_size, out_channels, new_height, new_width)
        """
        # Apply padding to input
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Use PyTorch's conv2d for the actual convolution operation
        # This implements the sliding window dot product efficiently
        out = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=0)
        
        return out


class MaxPool2D(nn.Module):
    """
    Implements 2D max pooling operation.
    Reduces spatial dimensions by taking the maximum value in each window.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Pooled tensor with reduced spatial dimensions
        """
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AveragePool2D(nn.Module):
    """
    Implements 2D average pooling operation.
    Reduces spatial dimensions by taking the average value in each window.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Pooled tensor with reduced spatial dimensions
        """
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class Flatten(nn.Module):
    """
    Flattens a multi-dimensional tensor into 2D (batch_size, -1).
    Used to transition from convolutional layers to fully connected layers.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class ActivationFunctions:
    """Collection of activation functions commonly used in CNNs."""
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """ReLU activation: max(0, x)"""
        return F.relu(x)
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return torch.sigmoid(x)
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return torch.tanh(x)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax activation for multi-class classification."""
        return F.softmax(x, dim=dim)
