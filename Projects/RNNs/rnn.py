import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    """
    Implements a basic Recurrent Neural Network cell.
    The simplest form of recurrent architecture for sequence processing.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden projection
        self.Wxh = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Hidden to hidden (recurrent) projection
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.Wxh.weight)
        nn.init.xavier_uniform_(self.Whh.weight)
        if self.Wxh.bias is not None:
            nn.init.zeros_(self.Wxh.bias)
            nn.init.zeros_(self.Whh.bias)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        # RNN update equation: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
        h_new = torch.tanh(self.Wxh(x) + self.Whh(h_prev))
        return h_new


class RNN(nn.Module):
    """
    Multi-layer Recurrent Neural Network.
    Processes sequences by maintaining a hidden state that evolves over time.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Stack of RNN cells
        self.cells = nn.ModuleList()
        
        # First layer
        self.cells.append(RNNCell(input_size, hidden_size, bias))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.cells.append(RNNCell(hidden_size, hidden_size, bias))
        
        # Dropout between layers
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(batch_size, self.hidden_size, device=device)
    
    def forward(self, x: torch.Tensor, 
                h_init: torch.Tensor = None) -> tuple:
        """
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size) if batch_first=True
            h_init: Initial hidden state (optional)
        Returns:
            outputs: All hidden states of shape (batch_size, seq_len, hidden_size)
            h_final: Final hidden state of shape (batch_size, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first for processing
        
        device = x.device
        
        # Initialize hidden states for all layers
        if h_init is None:
            hidden_states = [self.init_hidden(batch_size, device) for _ in range(self.num_layers)]
        else:
            hidden_states = list(h_init.unbind(0))
        
        # Store all outputs
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Input at time t
            
            # Pass through each layer
            for layer in range(self.num_layers):
                hidden_states[layer] = self.cells[layer](x_t, hidden_states[layer])
                x_t = hidden_states[layer]  # Output becomes input for next layer
                
                # Apply dropout between layers (not after last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            # Collect output from last layer
            outputs.append(hidden_states[-1].unsqueeze(1))
        
        # Concatenate all time steps
        outputs = torch.cat(outputs, dim=1)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        h_final = torch.stack(hidden_states, dim=0)
        
        return outputs, h_final


class ActivationFunctions:
    """Collection of activation functions commonly used in RNNs."""
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        """Tanh activation for RNN hidden states."""
        return torch.tanh(x)
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """ReLU activation."""
        return F.relu(x)
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation for gates."""
        return torch.sigmoid(x)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax for output layer."""
        return F.softmax(x, dim=dim)
