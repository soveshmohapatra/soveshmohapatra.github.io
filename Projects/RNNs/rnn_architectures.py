import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import RNN, RNNCell, ActivationFunctions


class SequenceClassifier(nn.Module):
    """
    RNN-based sequence classifier.
    Takes a sequence and outputs a single class prediction.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.rnn = RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # RNN forward pass
        outputs, h_final = self.rnn(x)
        
        # Use final hidden state for classification
        # h_final shape: (num_layers, batch_size, hidden_size)
        h_final = h_final[-1]  # Take last layer
        
        h_final = self.dropout(h_final)
        logits = self.fc(h_final)
        
        return logits


class SequenceTagger(nn.Module):
    """
    RNN-based sequence tagger (many-to-many).
    Outputs a prediction for each time step.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.rnn = RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            logits: Output logits for each time step (batch_size, seq_len, output_size)
        """
        # RNN forward pass
        outputs, h_final = self.rnn(x)
        
        # Apply dropout and projection to each time step
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        
        return logits


class CharRNN(nn.Module):
    """
    Character-level RNN for text generation.
    Takes character indices and predicts the next character.
    """
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # RNN layers
        self.rnn = RNN(embedding_size, hidden_size, num_layers, dropout=dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h_init=None):
        """
        Args:
            x: Input character indices (batch_size, seq_len)
            h_init: Initial hidden state (optional)
        Returns:
            logits: Character logits (batch_size, seq_len, vocab_size)
            h_final: Final hidden state
        """
        # Embed characters
        embedded = self.dropout(self.embedding(x))
        
        # RNN forward pass
        outputs, h_final = self.rnn(embedded, h_init)
        
        # Output projection
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        
        return logits, h_final
    
    def generate(self, start_token: int, seq_len: int, temperature: float = 1.0):
        """
        Generate text character by character.
        
        Args:
            start_token: Starting character index
            seq_len: Number of characters to generate
            temperature: Sampling temperature (higher = more random)
        Returns:
            generated: List of generated character indices
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Start with single token
        current = torch.tensor([[start_token]], device=device)
        hidden = None
        generated = [start_token]
        
        with torch.no_grad():
            for _ in range(seq_len - 1):
                # Forward pass
                embedded = self.embedding(current)
                logits, hidden = self.forward(current, hidden)
                
                # Get logits for last time step
                next_logits = logits[:, -1, :] / temperature
                
                # Sample from distribution
                probs = ActivationFunctions.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated.append(next_token.item())
                current = next_token
        
        return generated


class BidirectionalRNN(nn.Module):
    """
    Bidirectional RNN that processes sequences in both directions.
    Concatenates forward and backward hidden states.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.forward_rnn = RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.backward_rnn = RNN(input_size, hidden_size, num_layers, dropout=dropout)
        
        self.output_size = hidden_size * 2  # Concatenate both directions
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            outputs: Concatenated forward and backward outputs
        """
        # Forward pass
        fwd_outputs, fwd_h = self.forward_rnn(x)
        
        # Backward pass (reverse sequence)
        x_reversed = x.flip(dims=[1])
        bwd_outputs, bwd_h = self.backward_rnn(x_reversed)
        
        # Reverse backward outputs back to original order
        bwd_outputs = bwd_outputs.flip(dims=[1])
        
        # Concatenate forward and backward
        outputs = torch.cat([fwd_outputs, bwd_outputs], dim=-1)
        
        return outputs
