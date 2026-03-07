import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from snn_neuron import SNNClassifier, count_parameters


# ---------------------------------------------------------------------------
# ANN Baseline (standard 2-layer MLP)
# ---------------------------------------------------------------------------

class ANNClassifier(nn.Module):
    """
    Standard 2-hidden-layer MLP (ReLU) for MNIST, comparable parameter count.
    """
    def __init__(self, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int = 256) -> Tuple:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),   # flatten to (784,)
    ])
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch_snn(model, loader, optimizer, criterion, num_steps, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, num_steps=num_steps)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def train_epoch_ann(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate_snn(model, loader, criterion, num_steps, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, num_steps=num_steps)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate_ann(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Energy Estimation (Synaptic Operations)
# ---------------------------------------------------------------------------

def estimate_snn_sops(model, num_steps: int, avg_firing_rate: float) -> int:
    """
    Estimate Synaptic Operations (SOPs) for one SNN inference.

    SOPs = sum over layers of (fan_in × spikes_per_step × T)
    For binary spikes, each MAC becomes an accumulate (AC), which is ~5x cheaper
    than a multiply-accumulate (MAC) on neuromorphic hardware.

    avg_firing_rate: estimated fraction of neurons firing per step (0–1).
    """
    total_sops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            fan_out = module.out_features
            # Each output neuron receives fan_in spikes * avg_firing_rate * T AC ops
            total_sops += fan_in * fan_out * avg_firing_rate * num_steps
    return int(total_sops)


def estimate_ann_macs(model) -> int:
    """Estimate total Multiply-Accumulate (MAC) operations for one ANN forward pass."""
    total_macs = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features
    return total_macs


def measure_avg_firing_rate(model, loader, num_steps: int, device, n_batches: int = 5) -> float:
    """Measure average fraction of neurons that fire per step across sample batches."""
    model.eval()
    rates = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            # Replay forward pass, collecting spike stats
            from snn_neuron import LIFNeuron
            mem1 = model.lif1.init_membrane(batch_size, device)
            mem2 = model.lif2.init_membrane(batch_size, device)
            total_spikes1, total_spikes2 = 0, 0
            for _ in range(num_steps):
                cur1 = model.fc1(imgs)
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)
                total_spikes1 += spk1.sum().item()
                total_spikes2 += spk2.sum().item()
            total_neurons = (model.hidden_size + model.hidden_size) * batch_size * num_steps
            rates.append((total_spikes1 + total_spikes2) / total_neurons)
    return float(np.mean(rates))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(snn_train_accs, ann_train_accs, snn_test_accs, ann_test_accs,
                 snn_params, ann_params, snn_acc, ann_acc):
    """Generate a clean, light-background benchmark figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    epochs = range(1, len(snn_train_accs) + 1)

    # --- Panel 1: Training accuracy curves ---
    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(epochs, ann_train_accs, label=f"ANN Train ({ann_params:,} params)",
            linewidth=2.5, color="#2196F3")
    ax.plot(epochs, snn_train_accs, label=f"SNN Train ({snn_params:,} params)",
            linewidth=2.5, color="#FF5722")
    ax.plot(epochs, ann_test_accs, label="ANN Test",
            linewidth=2, color="#2196F3", linestyle="--", alpha=0.7)
    ax.plot(epochs, snn_test_accs, label="SNN Test",
            linewidth=2, color="#FF5722", linestyle="--", alpha=0.7)
    ax.set_title("Training & Test Accuracy over Epochs", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel 2: Final test accuracy bar chart ---
    ax2 = axes[1]
    ax2.set_facecolor("white")
    models = ["ANN (MLP)", "SNN (LIF)"]
    accs = [ann_acc, snn_acc]
    colors = ["#2196F3", "#FF5722"]
    bars = ax2.bar(models, accs, color=colors, width=0.45, edgecolor="white",
                   linewidth=1.5, alpha=0.9)
    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Final Test Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3, color="#cccccc")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("SNN vs ANN on MNIST — Accuracy & Parameter Efficiency", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("snn_vs_ann_benchmark.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved benchmark plot to 'snn_vs_ann_benchmark.png'")
    plt.close()


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Hyperparameters
    EPOCHS     = 10
    BATCH_SIZE = 256
    LR         = 1e-3
    NUM_STEPS  = 25          # SNN simulation timesteps per sample
    HIDDEN     = 256

    # Data
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}\n")

    # Models
    snn_model = SNNClassifier(784, HIDDEN, 10).to(device)
    ann_model = ANNClassifier(784, HIDDEN, 10).to(device)
    snn_params = count_parameters(snn_model)
    ann_params = count_parameters(ann_model)
    print(f"SNN Parameters: {snn_params:,}")
    print(f"ANN Parameters: {ann_params:,}\n")

    # Optimizers & loss
    criterion = nn.CrossEntropyLoss()
    snn_opt = torch.optim.Adam(snn_model.parameters(), lr=LR)
    ann_opt = torch.optim.Adam(ann_model.parameters(), lr=LR)

    snn_train_accs, ann_train_accs = [], []
    snn_test_accs,  ann_test_accs  = [], []

    print(f"Training for {EPOCHS} epochs (SNN uses T={NUM_STEPS} timesteps)...\n")
    for epoch in range(1, EPOCHS + 1):
        _, snn_tr_acc = train_epoch_snn(snn_model, train_loader, snn_opt, criterion, NUM_STEPS, device)
        _, ann_tr_acc = train_epoch_ann(ann_model, train_loader, ann_opt, criterion, device)

        _, snn_te_acc = evaluate_snn(snn_model, test_loader, criterion, NUM_STEPS, device)
        _, ann_te_acc = evaluate_ann(ann_model, test_loader, criterion, device)

        snn_train_accs.append(snn_tr_acc)
        ann_train_accs.append(ann_tr_acc)
        snn_test_accs.append(snn_te_acc)
        ann_test_accs.append(ann_te_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"SNN Train: {snn_tr_acc:.2f}%  Test: {snn_te_acc:.2f}% | "
              f"ANN Train: {ann_tr_acc:.2f}%  Test: {ann_te_acc:.2f}%")

    final_snn_acc = snn_test_accs[-1]
    final_ann_acc = ann_test_accs[-1]

    print(f"\nFinal SNN Test Accuracy: {final_snn_acc:.2f}%")
    print(f"Final ANN Test Accuracy: {final_ann_acc:.2f}%")

    # Energy estimation
    avg_rate = measure_avg_firing_rate(snn_model, test_loader, NUM_STEPS, device)
    snn_sops = estimate_snn_sops(snn_model, NUM_STEPS, avg_rate)
    ann_macs = estimate_ann_macs(ann_model)
    print(f"\nAverage SNN firing rate: {avg_rate:.3f} (fraction of neurons active per step)")
    print(f"SNN Synaptic Operations (SOPs) per inference: {snn_sops:,}")
    print(f"ANN Multiply-Accumulate Ops (MACs) per inference: {ann_macs:,}")
    energy_ratio = ann_macs / snn_sops if snn_sops > 0 else float("inf")
    print(f"MAC/SOP ratio: {energy_ratio:.2f}x  (SNNs replace MACs with cheaper ACs on neuromorphic HW)\n")

    # Plot
    plot_results(snn_train_accs, ann_train_accs, snn_test_accs, ann_test_accs,
                 snn_params, ann_params, final_snn_acc, final_ann_acc)

    # Save results
    with open("benchmark_results.txt", "w") as f:
        f.write(f"SNN Parameters: {snn_params}\n")
        f.write(f"ANN Parameters: {ann_params}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"SNN Timesteps (T): {NUM_STEPS}\n")
        f.write(f"Final SNN Train Accuracy: {snn_train_accs[-1]:.2f}\n")
        f.write(f"Final SNN Test Accuracy:  {final_snn_acc:.2f}\n")
        f.write(f"Final ANN Train Accuracy: {ann_train_accs[-1]:.2f}\n")
        f.write(f"Final ANN Test Accuracy:  {final_ann_acc:.2f}\n")
        f.write(f"Avg SNN Firing Rate: {avg_rate:.4f}\n")
        f.write(f"SNN SOPs per Inference: {snn_sops}\n")
        f.write(f"ANN MACs per Inference: {ann_macs}\n")
        f.write(f"MAC/SOP Ratio: {energy_ratio:.2f}\n")
    print("Saved benchmark results to 'benchmark_results.txt'")


if __name__ == "__main__":
    run_benchmark()
