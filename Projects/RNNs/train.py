import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
import random
from rnn_architectures import SequenceClassifier, SequenceTagger, CharRNN

# Setup logging
logging.basicConfig(
    filename='train.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def generate_sequence_data(num_samples: int, seq_len: int = 20, 
                           input_size: int = 10, num_classes: int = 5):
    """
    Generate synthetic sequence classification data.
    Task: Classify sequence based on sum of values (threshold-based).
    """
    X = torch.randn(num_samples, seq_len, input_size)
    
    # Simple rule: if mean of sequence > 0, class 1, else class 0
    y = (X.mean(dim=[1, 2]) > 0).long()
    
    return X, y


def generate_tagging_data(num_samples: int, seq_len: int = 20,
                          input_size: int = 10, num_tags: int = 5):
    """
    Generate synthetic sequence tagging data.
    Task: Tag each position based on value threshold.
    """
    X = torch.randn(num_samples, seq_len, input_size)
    
    # Tag based on sign of values
    y = (X[:, :, 0] > 0).long()
    
    return X, y


def train_classifier():
    """Train RNN on sequence classification task."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    INPUT_SIZE = 10
    HIDDEN_SIZE = 64
    NUM_CLASSES = 2
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    SEQ_LEN = 20
    
    # Generate data
    logger.info("Generating synthetic sequence data...")
    X_train, y_train = generate_sequence_data(5000, SEQ_LEN, INPUT_SIZE, NUM_CLASSES)
    X_test, y_test = generate_sequence_data(1000, SEQ_LEN, INPUT_SIZE, NUM_CLASSES)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    # Initialize model
    model = SequenceClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_LAYERS).to(device)
    
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
        indices = torch.randperm(len(X_train))[:BATCH_SIZE * 10]
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
    plt.title('RNN Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('RNN Classification Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    logger.info("Saved training curves to training_curves.png")
    
    visualize_hidden_states(model, SEQ_LEN, INPUT_SIZE, device)


def visualize_hidden_states(model, seq_len: int, input_size: int, device):
    """Visualize hidden state dynamics over time."""
    model.eval()
    
    # Create a sample sequence
    x = torch.randn(1, seq_len, input_size).to(device)
    
    with torch.no_grad():
        outputs, h_final = model.rnn(x)
    
    # outputs shape: (1, seq_len, hidden_size)
    hidden_states = outputs[0].cpu().numpy()
    
    # Plot hidden state dynamics
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot first 16 hidden units over time
    for i in range(min(16, hidden_states.shape[1])):
        axes[i].plot(hidden_states[:, i], linewidth=2)
        axes[i].set_title(f'Hidden Unit {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Activation')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(16, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('hidden_states.png', dpi=300)
    plt.close()
    logger.info("Saved hidden state visualization to hidden_states.png")


def train():
    """Main training function."""
    train_classifier()
    logger.info("Training complete. Artifacts saved: train.log, training_curves.png, hidden_states.png")
    print("Training complete. Artifacts saved: train.log, training_curves.png, hidden_states.png")


if __name__ == '__main__':
    train()
