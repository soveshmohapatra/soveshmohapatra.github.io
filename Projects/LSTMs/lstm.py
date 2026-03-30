import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """
    Implements a Long Short-Term Memory (LSTM) cell.
    Solves the vanishing gradient problem through gated memory updates.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gates
        self.Wxi = nn.Linear(input_size, hidden_size, bias=bias)  # Input -> input gate
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden -> input gate
        
        # Forget gates
        self.Wxf = nn.Linear(input_size, hidden_size, bias=bias)  # Input -> forget gate
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden -> forget gate
        
        # Cell gates (candidate cell state)
        self.Wxc = nn.Linear(input_size, hidden_size, bias=bias)  # Input -> cell candidate
        self.Whc = nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden -> cell candidate
        
        # Output gates
        self.Wxo = nn.Linear(input_size, hidden_size, bias=bias)  # Input -> output gate
        self.Who = nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden -> output gate
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for weight in [self.Wxi, self.Whi, self.Wxf, self.Whf, 
                       self.Wxc, self.Whc, self.Wxo, self.Who]:
            nn.init.xavier_uniform_(weight.weight)
            if weight.bias is not None:
                nn.init.zeros_(weight.bias)
        
        # Initialize forget gate bias to 1.0 for better gradient flow
        if self.Wxf.bias is not None:
            nn.init.ones_(self.Wxf.bias)
        if self.Whf.bias is not None:
            nn.init.ones_(self.Whf.bias)
    
    def forward(self, x: torch.Tensor, 
                h_prev: torch.Tensor, 
                c_prev: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
            c_prev: Previous cell state of shape (batch_size, hidden_size)
        Returns:
            h_new: New hidden state
            c_new: New cell state
        """
        # Compute gates
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))  # Input gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))  # Forget gate
        g = torch.tanh(self.Wxc(x) + self.Whc(h_prev))     # Cell candidate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))  # Output gate
        
        # Update cell state: c_t = f * c_{t-1} + i * g
        c_new = f * c_prev + i * g
        
        # Update hidden state: h_t = o * tanh(c_t)
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """
    Multi-layer Long Short-Term Memory network.
    Processes sequences using gated recurrent units with cell state memory.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Stack of LSTM cells
        self.cells = nn.ModuleList()
        
        # First layer
        self.cells.append(LSTMCell(input_size, hidden_size, bias))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.cells.append(LSTMCell(hidden_size, hidden_size, bias))
        
        # Dropout between layers
        if dropout > 0 and num_layers > 1:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """Initialize hidden state and cell state with zeros."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c
    
    def forward(self, x: torch.Tensor, 
                h_init: tuple = None) -> tuple:
        """
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_size) if batch_first=True
            h_init: Tuple of (hidden_state, cell_state) initial states (optional)
        Returns:
            outputs: All hidden states of shape (batch_size, seq_len, hidden_size)
            h_final: Tuple of (final hidden state, final cell state)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first for processing
        
        device = x.device
        
        # Initialize states for all layers
        if h_init is None:
            hidden_states = [self.init_hidden(batch_size, device) for _ in range(self.num_layers)]
        else:
            hidden_states = list(zip(h_init[0].unbind(0), h_init[1].unbind(0)))
        
        # Store all outputs
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Input at time t
            
            # Pass through each layer
            for layer in range(self.num_layers):
                h_prev, c_prev = hidden_states[layer]
                h_new, c_new = self.cells[layer](x_t, h_prev, c_prev)
                hidden_states[layer] = (h_new, c_new)
                x_t = h_new  # Output becomes input for next layer
                
                # Apply dropout between layers (not after last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            # Collect output from last layer
            outputs.append(hidden_states[-1][0].unsqueeze(1))
        
        # Concatenate all time steps
        outputs = torch.cat(outputs, dim=1)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        h_final = (
            torch.stack([h for h, c in hidden_states], dim=0),
            torch.stack([c for h, c in hidden_states], dim=0)
        )
        
        return outputs, h_final


class ActivationFunctions:
    """Collection of activation functions commonly used in LSTMs."""
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation for gates."""
        return torch.sigmoid(x)
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        """Tanh activation for cell state."""
        return torch.tanh(x)
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """ReLU activation."""
        return F.relu(x)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Softmax for output layer."""
        return F.softmax(x, dim=dim)
