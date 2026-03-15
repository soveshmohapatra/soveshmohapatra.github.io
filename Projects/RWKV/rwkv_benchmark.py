import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from rwkv_layer import RWKV, count_parameters


def generate_next_token_prediction(batch_size, seq_len, vocab_size=64):
    """Standard next token prediction task."""
    x = torch.randint(0, vocab_size // 2, (batch_size, seq_len))
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:].clone()
    y[:, -1] = x[:, 0]
    return x, y


class TransformerBaseline(nn.Module):
    """Standard Transformer encoder baseline."""
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.head(self.ln_out(x))


class LSTMBaseline(nn.Module):
    """Standard LSTM baseline."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.head(out)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        # Handle tuple output (RWKV returns (logits, states))
        logits = output[0] if isinstance(output, tuple) else output
        loss = criterion(logits.view(-1, logits.shape[-1]), batch_y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_y.numel()
        total_correct += (logits.argmax(dim=-1) == batch_y).sum().item()
        total_tokens += batch_y.numel()
    return total_loss / total_tokens, 100.0 * total_correct / total_tokens


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits.view(-1, logits.shape[-1]), batch_y.view(-1))
            total_loss += loss.item() * batch_y.numel()
            total_correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total_tokens += batch_y.numel()
    return total_loss / total_tokens, 100.0 * total_correct / total_tokens


def measure_inference_latency(model, seq_len, device, num_runs=50):
    """Measure per-token inference latency."""
    model.eval()
    x = torch.randint(0, 100, (1, seq_len)).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model(x)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)
    total_time = time.time() - start_time
    return total_time / num_runs / seq_len * 1000  # ms per token


def run_benchmark():
    VOCAB_SIZE, EMBED_DIM, NUM_LAYERS = 64, 128, 4
    SEQ_LENS = [32, 64, 128, 256]
    EPOCHS, BATCH_SIZE, LR = 50, 32, 1e-3
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Generating synthetic sequence dataset...")
    X_train, Y_train = generate_next_token_prediction(5000, 256, VOCAB_SIZE)
    X_test, Y_test = generate_next_token_prediction(1000, 256, VOCAB_SIZE)
    
    print("Creating models...")
    rwkv_model = RWKV(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS).to(device)
    transformer_model = TransformerBaseline(VOCAB_SIZE, EMBED_DIM, NUM_LAYERS).to(device)
    lstm_model = LSTMBaseline(VOCAB_SIZE, EMBED_DIM, EMBED_DIM, NUM_LAYERS).to(device)
    
    print(f"RWKV Parameters:      {count_parameters(rwkv_model):,}")
    print(f"Transformer Parameters: {count_parameters(transformer_model):,}")
    print(f"LSTM Parameters:       {count_parameters(lstm_model):,}\n")
    
    criterion = nn.CrossEntropyLoss()
    rwkv_opt = optim.AdamW(rwkv_model.parameters(), lr=LR, betas=(0.9, 0.95))
    tf_opt = optim.AdamW(transformer_model.parameters(), lr=LR, betas=(0.9, 0.95))
    lstm_opt = optim.AdamW(lstm_model.parameters(), lr=LR, betas=(0.9, 0.95))
    
    results = {
        'RWKV': {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []},
        'Transformer': {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []},
        'LSTM': {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    }
    
    print(f"Training for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        for name, model, optimizer in [
            ('RWKV', rwkv_model, rwkv_opt),
            ('Transformer', transformer_model, tf_opt),
            ('LSTM', lstm_model, lstm_opt)
        ]:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            results[name]['train_loss'].append(train_loss)
            results[name]['test_loss'].append(test_loss)
            results[name]['train_acc'].append(train_acc)
            results[name]['test_acc'].append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}")
            for name in ['RWKV', 'Transformer', 'LSTM']:
                print(f"  {name}: Train Loss: {results[name]['train_loss'][-1]:.4f}, Test Loss: {results[name]['test_loss'][-1]:.4f}")
            print()
    
    print("\n--- Inference Benchmarks ---")
    inference_results = {}
    for seq_len in SEQ_LENS:
        print(f"\nSequence Length: {seq_len}")
        rwkv_lat = measure_inference_latency(rwkv_model, seq_len, device)
        tf_lat = measure_inference_latency(transformer_model, seq_len, device)
        lstm_lat = measure_inference_latency(lstm_model, seq_len, device)
        inference_results[seq_len] = {
            'RWKV': rwkv_lat, 'Transformer': tf_lat, 'LSTM': lstm_lat
        }
        print(f"  RWKV:       {rwkv_lat:.2f} ms/token")
        print(f"  Transformer: {tf_lat:.2f} ms/token")
        print(f"  LSTM:       {lstm_lat:.2f} ms/token")
    
    # Save results
    with open("benchmark_results.txt", "w") as f:
        f.write(f"RWKV Parameters: {count_parameters(rwkv_model)}\n")
        f.write(f"Transformer Parameters: {count_parameters(transformer_model)}\n")
        f.write(f"LSTM Parameters: {count_parameters(lstm_model)}\n\n")
        for name in ['RWKV', 'Transformer', 'LSTM']:
            f.write(f"{name}:\n")
            f.write(f"  Final Train Loss: {results[name]['train_loss'][-1]:.4f}\n")
            f.write(f"  Final Test Loss: {results[name]['test_loss'][-1]:.4f}\n\n")
        f.write("\nInference Latency (ms/token):\n")
        for seq_len, data in inference_results.items():
            f.write(f"  L={seq_len}: RWKV={data['RWKV']:.2f}, Transformer={data['Transformer']:.2f}, LSTM={data['LSTM']:.2f}\n")
    print("\nSaved benchmark_results.txt")
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    epochs = range(1, EPOCHS + 1)
    colors = {'RWKV': '#1565C0', 'Transformer': '#EE4C2C', 'LSTM': '#2E7D32'}
    
    # Plot 1: Training Loss
    axes[0, 0].set_facecolor("white")
    for name in ['RWKV', 'Transformer', 'LSTM']:
        axes[0, 0].plot(epochs, results[name]['train_loss'], label=name, color=colors[name], linewidth=2)
    axes[0, 0].set_title('Training Loss', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test Loss
    axes[0, 1].set_facecolor("white")
    for name in ['RWKV', 'Transformer', 'LSTM']:
        axes[0, 1].plot(epochs, results[name]['test_loss'], label=name, color=colors[name], linewidth=2)
    axes[0, 1].set_title('Test Loss (Generalization)', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Inference Latency vs Sequence Length
    axes[1, 0].set_facecolor("white")
    for name in ['RWKV', 'Transformer', 'LSTM']:
        lats = [inference_results[s][name] for s in SEQ_LENS]
        axes[1, 0].plot(SEQ_LENS, lats, 'o-', label=name, color=colors[name], linewidth=2, markersize=8)
    axes[1, 0].set_title('Inference Latency vs Sequence Length', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Sequence Length', fontsize=12)
    axes[1, 0].set_ylabel('Latency (ms/token)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final Test Accuracy Comparison
    axes[1, 1].set_facecolor("white")
    models = ['RWKV', 'Transformer', 'LSTM']
    final_accs = [results[m]['test_acc'][-1] for m in models]
    bars = axes[1, 1].bar(models, final_accs, color=[colors[m] for m in models], alpha=0.8, edgecolor='white', linewidth=2)
    axes[1, 1].set_title('Final Test Accuracy Comparison', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, acc in zip(bars, final_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('RWKV vs Transformer vs LSTM Benchmark', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('rwkv_benchmark.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated rwkv_benchmark.png\nBenchmark complete!")


if __name__ == '__main__':
    run_benchmark()
