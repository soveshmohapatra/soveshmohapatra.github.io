import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import logging
from cnn import SimpleCNN, LeNet5, DeepCNN

# Setup logging
logging.basicConfig(
    filename='train.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_mnist(batch_size: int = 64):
    """Load MNIST dataset for digit classification."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def visualize_filters(model, layer_name: str = 'conv1'):
    """Visualize learned convolutional filters."""
    model.eval()
    
    # Get the convolutional layer weights
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        if hasattr(layer, 'weight'):
            filters = layer.weight.data.cpu()
            
            # Normalize filters for visualization
            filters = (filters - filters.min()) / (filters.max() - filters.min() + 1e-8)
            
            # Create grid visualization
            num_filters = filters.size(0)
            grid_size = int(np.ceil(np.sqrt(num_filters)))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            if grid_size == 1:
                axes = np.array([[axes]])
            elif axes.ndim == 1:
                axes = axes.reshape(1, -1)
            
            for i, ax_row in enumerate(axes):
                for j, ax in enumerate(ax_row):
                    idx = i * grid_size + j
                    if idx < num_filters:
                        # Handle multi-channel filters (take mean or first channel)
                        if filters[idx].dim() == 3:
                            filter_img = filters[idx].mean(dim=0)
                        else:
                            filter_img = filters[idx]
                        ax.imshow(filter_img, cmap='viridis')
                    ax.axis('off')
            
            plt.tight_layout()
            plt.savefig('filters.png', dpi=300)
            plt.close()
            logger.info("Saved filter visualization to filters.png")


def visualize_feature_maps(model, test_loader, device, num_maps: int = 8):
    """Visualize feature maps from convolutional layers."""
    model.eval()
    
    # Get a sample image
    data, target = next(iter(test_loader))
    data = data[:1].to(device)  # Take first image
    
    with torch.no_grad():
        # Forward pass through first conv layer
        x = F.relu(model.conv1(data))
        
        # Get feature maps
        feature_maps = x[0].cpu().numpy()
        
        # Select subset of feature maps to visualize
        num_maps = min(num_maps, feature_maps.shape[0])
        
        fig, axes = plt.subplots(2, num_maps // 2, figsize=(12, 3))
        
        for i, ax in enumerate(axes.flat):
            if i < num_maps:
                ax.imshow(feature_maps[i], cmap='viridis')
                ax.set_title(f'Map {i+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('feature_maps.png', dpi=300)
        plt.close()
        logger.info("Saved feature map visualization to feature_maps.png")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(BATCH_SIZE)
    
    # Initialize model
    model = SimpleCNN(num_classes=10, in_channels=1).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch}/{EPOCHS} | "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    logger.info("Saved training curves to training_curves.png")
    
    # Visualize learned features
    visualize_filters(model, 'conv1')
    visualize_feature_maps(model, test_loader, device)
    
    logger.info("Training complete. Artifacts saved: train.log, training_curves.png, filters.png, feature_maps.png")
    print("Training complete. Artifacts saved: train.log, training_curves.png, filters.png, feature_maps.png")


if __name__ == '__main__':
    train()
