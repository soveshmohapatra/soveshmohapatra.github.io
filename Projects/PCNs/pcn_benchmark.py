import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import numpy as np

# Import our custom PCN components
from pcn_layer import PCNetwork

# =============================================================================
# Helper: Standard MLP for Comparison
# =============================================================================
class StandardMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# =============================================================================
# Benchmark 1: Regression (Sine Wave Prediction)
# =============================================================================
def run_regression_benchmark():
    print("\n" + "="*50)
    print("BENCHMARK 1: Regression (Sine Wave)")
    print("="*50)
    
    # 1. Generate Data
    X = torch.linspace(0, 4*math.pi, 1000).unsqueeze(1)
    y = torch.sin(X) + torch.randn_like(X) * 0.1  # Add some noise
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    layer_sizes = [1, 32, 32, 1]
    
    # 2. Setup Models
    pcn_model = PCNetwork(layer_sizes, activation=torch.tanh) # Tanh often better for regression state
    mlp_model = StandardMLP(layer_sizes)
    
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 50
    pcn_losses = []
    mlp_losses = []
    
    # --- Train PCN ---
    print("\nTraining PCN (Regression)...")
    start_time = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for bx, by in loader:
            # PCN train_batch handles inference + weight update internally
            energy = pcn_model.train_batch(bx, y=by, 
                                           inference_steps=20, 
                                           inference_lr=0.05, 
                                           weight_lr=0.005, 
                                           is_supervised=True)
            # Evaluate true MSE error after step
            # Note: We don't use torch.no_grad() because predict() requires autograd 
            # to compute state updates.
            pred = pcn_model.predict(bx, inference_steps=30)
            ep_loss += criterion(pred, by).item()
        pcn_losses.append(ep_loss / len(loader))
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1:2d} | PCN MSE: {pcn_losses[-1]:.4f}")
    pcn_time = time.time() - start_time

    # --- Train MLP ---
    print("\nTraining Standard MLP (Regression)...")
    start_time = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for bx, by in loader:
            mlp_optimizer.zero_grad()
            pred = mlp_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            mlp_optimizer.step()
            ep_loss += loss.item()
        mlp_losses.append(ep_loss / len(loader))
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1:2d} | MLP MSE: {mlp_losses[-1]:.4f}")
    mlp_time = time.time() - start_time
    
    # --- Visualization ---
    # PCN prediction needs autograd enabled map
    pcn_preds = pcn_model.predict(X, inference_steps=30).detach().numpy()
    with torch.no_grad():
        mlp_preds = mlp_model(X).numpy()
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), color='gray', alpha=0.3, label='Data', s=10)
    plt.plot(X.numpy(), mlp_preds, color='blue', label='MLP Pred', linewidth=2)
    plt.plot(X.numpy(), pcn_preds, color='red', label='PCN Pred', linestyle='--', linewidth=2)
    plt.title("Regression Fits")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_losses, label=f'MLP (Time: {mlp_time:.2f}s)')
    plt.plot(pcn_losses, label=f'PCN (Time: {pcn_time:.2f}s)')
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pcn_vs_mlp_regression.png')
    print("Saved 'pcn_vs_mlp_regression.png'")


# =============================================================================
# Benchmark 2: Classification (MNIST subset for speed)
# =============================================================================
def run_mnist_benchmark():
    print("\n" + "="*50)
    print("BENCHMARK 2: Classification (MNIST)")
    print("="*50)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Use a subset of 5000 images for a quick benchmark
    subset_indices = torch.randperm(len(dataset))[:5000]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=True)
    
    # Architecture: 784 (Flattened 28x28) -> 128 -> 10 classes
    layer_sizes = [784, 128, 10]
    
    # Note: Using Tanh instead of ReLU because ReLU causes dead gradients during inference for negative latents
    pcn_model = PCNetwork(layer_sizes, activation=torch.tanh)
    mlp_model = StandardMLP(layer_sizes)
    
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    pcn_accs = []
    mlp_accs = []
    
    # --- Train PCN (Classification) ---
    print("\nTraining PCN (MNIST)...")
    start_time = time.time()
    for ep in range(epochs):
        correct = 0
        total = 0
        for bx, by in loader:
            batch_sz = bx.size(0)
            bx_flat = bx.view(batch_sz, -1)
            
            # One hot encode labels for PCN clamping
            by_onehot = F.one_hot(by, num_classes=10).float()
            
            pcn_model.train_batch(bx_flat, y=by_onehot, 
                                  inference_steps=50, 
                                  inference_lr=0.1, 
                                  weight_lr=0.005, 
                                  is_supervised=True)
                                  
            # Accuracy
            # Note: PCNs need autograd to run predict
            preds = pcn_model.predict(bx_flat, inference_steps=50)
            # Use softmax for numerical stability of the 'belief' states
            preds_probs = F.softmax(preds, dim=1)
            pred_labels = preds_probs.argmax(dim=1)
            correct += (pred_labels == by).sum().item()
            total += batch_sz
        
        acc = correct / total
        pcn_accs.append(acc)
        print(f"Epoch {ep+1:2d} | PCN Accuracy: {acc*100:.2f}%")
    pcn_time = time.time() - start_time

    # --- Train MLP (Classification) ---
    print("\nTraining Standard MLP (MNIST)...")
    start_time = time.time()
    for ep in range(epochs):
        correct = 0
        total = 0
        for bx, by in loader:
            batch_sz = bx.size(0)
            bx_flat = bx.view(batch_sz, -1)
            
            mlp_optimizer.zero_grad()
            out = mlp_model(bx_flat)
            loss = criterion(out, by)
            loss.backward()
            mlp_optimizer.step()
            
            pred_labels = out.argmax(dim=1)
            correct += (pred_labels == by).sum().item()
            total += batch_sz
            
        acc = correct / total
        mlp_accs.append(acc)
        print(f"Epoch {ep+1:2d} | MLP Accuracy: {acc*100:.2f}%")
    mlp_time = time.time() - start_time

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_accs, marker='o', label=f'MLP (Time: {mlp_time:.2f}s)')
    plt.plot(pcn_accs, marker='s', label=f'PCN (Time: {pcn_time:.2f}s)')
    plt.title("MNIST Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pcn_vs_mlp_mnist.png')
    print("Saved 'pcn_vs_mlp_mnist.png'")

if __name__ == "__main__":
    import math # needed for regression
    run_regression_benchmark()
    run_mnist_benchmark()
