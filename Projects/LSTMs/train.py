import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
from lstm_architectures import LSTMClassifier, LSTMTagger, CharLSTM

# Setup logging
logging.basicConfig(
    filename='train.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def generate_sequence_data(num_samples: int, seq_len: int = 30, 
                           input_size: int = 10, num_classes: int = 2):
    """
    Generate synthetic sequence classification data.
    Task: Classify based on long-range dependency (first and last elements).
    """
    X = torch.randn(num_samples, seq_len, input_size)
    
    # Long-range dependency task: classify based on sign of first + last element
    first_elem = X[:, 0, 0]
    last_elem = X[:, -1, 0]
    y = ((first_elem + last_elem) > 0).long()
    
    return X, y


def train_classifier():
    """Train LSTM on sequence classification with long-range dependencies."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    INPUT_SIZE = 10
    HIDDEN_SIZE = 128
    NUM_CLASSES = 2
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.001
    SEQ_LEN = 30
    
    # Generate data
    logger.info("Generating synthetic sequence data with long-range dependencies...")
    X_train, y_train = generate_sequence_data(5000, SEQ_LEN, INPUT_SIZE, NUM_CLASSES)
    X_test, y_test = generate_sequence_data(1000, SEQ_LEN, INPUT_SIZE, NUM_CLASSES)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Initialize model
    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_LAYERS).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training history
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Mini-batch training
        indices = torch.randperm(len(X_train))
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test).item()
            test_pred = test_logits.argmax(dim=1)
            test_acc = 100 * test_pred.eq(y_test).sum().item() / y_test.size(0)
        
        avg_loss = total_loss / (len(indices) // BATCH_SIZE)
        train_acc = 100 * correct / total
        
        train_losses.append(avg_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch}/{EPOCHS} | "
                   f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                   f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | "
                  f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LSTM Classification Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    logger.info("Saved training curves to training_curves.png")
    
    visualize_gates(model, SEQ_LEN, INPUT_SIZE, device)


def visualize_gates(model, seq_len: int, input_size: int, device):
    """Visualize LSTM gate activations over time."""
    model.eval()
    
    # Create a sample sequence
    x = torch.randn(1, seq_len, input_size).to(device)
    
    # Manually run through first LSTM cell to get gate activations
    lstm_cell = model.lstm.cells[0]
    
    h = torch.zeros(1, model.lstm.hidden_size, device=device)
    c = torch.zeros(1, model.lstm.hidden_size, device=device)
    
    gate_activations = {
        'input_gate': [],
        'forget_gate': [],
        'output_gate': [],
        'cell_candidate': []
    }
    
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            i = torch.sigmoid(lstm_cell.Wxi(x_t) + lstm_cell.Whi(h))
            f = torch.sigmoid(lstm_cell.Wxf(x_t) + lstm_cell.Whf(h))
            g = torch.tanh(lstm_cell.Wxc(x_t) + lstm_cell.Whc(h))
            o = torch.sigmoid(lstm_cell.Wxo(x_t) + lstm_cell.Who(h))
            
            gate_activations['input_gate'].append(i[0].cpu().numpy())
            gate_activations['forget_gate'].append(f[0].cpu().numpy())
            gate_activations['output_gate'].append(o[0].cpu().numpy())
            gate_activations['cell_candidate'].append(g[0].cpu().numpy())
            
            c = f * c + i * g
            h = o * torch.tanh(c)
    
    # Plot gate activations
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    titles = ['Input Gate', 'Forget Gate', 'Output Gate', 'Cell Candidate']
    keys = ['input_gate', 'forget_gate', 'output_gate', 'cell_candidate']
    
    for idx, (ax, key, title) in enumerate(zip(axes.flat, keys, titles)):
        activations = np.array(gate_activations[key])
        
        # Plot first 8 units
        for i in range(min(8, activations.shape[1])):
            ax.plot(activations[:, i], label=f'Unit {i+1}', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activation')
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gate_activations.png', dpi=300)
    plt.close()
    logger.info("Saved gate activation visualization to gate_activations.png")


def train():
    """Main training function."""
    train_classifier()
    logger.info("Training complete. Artifacts saved: train.log, training_curves.png, gate_activations.png")
    print("Training complete. Artifacts saved: train.log, training_curves.png, gate_activations.png")


if __name__ == '__main__':
    train()
