import torch
import torch.nn as nn
import torch.nn.functional as F
from lstm import LSTM, LSTMCell, ActivationFunctions


class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier.
    Uses the final hidden state for classification.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        
        if bidirectional:
            self.lstm_backward = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        
        fc_input_size = hidden_size * self.num_directions * num_layers
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Forward LSTM
        outputs, (h_final, c_final) = self.lstm(x)
        
        if self.bidirectional:
            # Backward LSTM
            x_reversed = x.flip(dims=[1])
            _, (h_final_bwd, _) = self.lstm_backward(x_reversed)
            
            # Concatenate forward and backward final states
            all_hidden = []
            for layer in range(self.lstm.num_layers):
                all_hidden.append(h_final[layer])
                all_hidden.append(h_final_bwd[layer])
            h_final = torch.cat(all_hidden, dim=-1)
        else:
            # Flatten all layer hidden states
            h_final = torch.cat([h_final[layer] for layer in range(self.lstm.num_layers)], dim=-1)
        
        h_final = self.dropout(h_final)
        logits = self.fc(h_final)
        
        return logits


class LSTMTagger(nn.Module):
    """
    LSTM-based sequence tagger (many-to-many).
    Outputs a prediction for each time step.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.lstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        Returns:
            logits: Output logits for each time step (batch_size, seq_len, output_size)
        """
        # LSTM forward pass
        outputs, _ = self.lstm(x)
        
        # Apply dropout and projection to each time step
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        
        return logits


class CharLSTM(nn.Module):
    """
    Character-level LSTM for text generation.
    Takes character indices and predicts the next character.
    """
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # LSTM layers
        self.lstm = LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, h_init=None):
        """
        Args:
            x: Input character indices (batch_size, seq_len)
            h_init: Tuple of (hidden_state, cell_state) initial states (optional)
        Returns:
            logits: Character logits (batch_size, seq_len, vocab_size)
            h_final: Tuple of (final hidden state, final cell state)
        """
        # Embed characters
        embedded = self.dropout(self.embedding(x))
        
        # LSTM forward pass
        outputs, h_final = self.lstm(embedded, h_init)
        
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


class Seq2SeqLSTM(nn.Module):
    """
    Sequence-to-Sequence model with LSTM encoder and decoder.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embedding_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.encoder = LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        
        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.decoder = LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        
        # Output projection
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt):
        """
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
        Returns:
            logits: Output logits (batch_size, tgt_len, tgt_vocab_size)
        """
        # Encode
        src_embedded = self.dropout(self.src_embedding(src))
        _, (h_enc, c_enc) = self.encoder(src_embedded)
        
        # Decode
        tgt_embedded = self.dropout(self.tgt_embedding(tgt))
        dec_outputs, _ = self.decoder(tgt_embedded, (h_enc, c_enc))
        
        # Project to vocabulary
        logits = self.fc(dec_outputs)
        
        return logits
