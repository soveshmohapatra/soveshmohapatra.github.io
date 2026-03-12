import torch
import torch.nn as nn
import unittest

class EchoStateNetwork(nn.Module):
    """
    A pure PyTorch implementation of an Echo State Network (ESN).
    The input and reservoir weights are randomly initialized and frozen.
    Only the readout (output) weights are trained using Ridge Regression.
    """
    def __init__(self, input_size, hidden_size, output_size, spectral_radius=0.9, sparsity=0.1, input_scaling=1.0, leaky_rate=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaky_rate = leaky_rate

        # 1. Initialize Input Weights (W_in)
        # Uniform [-input_scaling, input_scaling]
        self.W_in = nn.Parameter(
            (torch.rand(hidden_size, input_size) * 2 - 1) * input_scaling,
            requires_grad=False
        )

        # 2. Initialize Reservoir Weights (W_res)
        # Sparse random matrix
        W = torch.rand(hidden_size, hidden_size) - 0.5
        mask = torch.rand(hidden_size, hidden_size) < sparsity
        W = W * mask
        
        # 3. Scale W_res to achieve the desired Spectral Radius
        # Spectral radius = max absolute eigenvalue
        eigenvalues = torch.linalg.eigvals(W)
        current_sr = torch.max(torch.abs(eigenvalues)).item()
        if current_sr > 0:
            W = W * (spectral_radius / current_sr)
            
        self.W_res = nn.Parameter(W, requires_grad=False)

        # 4. Initialize Output Weights (W_out)
        # These will be learned via Ridge Regression, so we initialize to zeros
        # Note: We include an explicit bias inherently by adding a column of 1s to the states later, 
        # or we can keep a dedicated bias parameter. For simplicity in ridge regression, 
        # we'll expand the state matrix with 1s during fit.
        # W_out shape: (output_size, hidden_size + 1 (for bias))
        self.W_out = nn.Parameter(
            torch.zeros(output_size, hidden_size + 1 + input_size), # Optional: include input in readout
            requires_grad=False
        )

    def forward(self, inputs, initial_state=None):
        """
        Run the inputs through the reservoir to collect the internal states.
        inputs: (batch, seq_len, input_size) or (seq_len, input_size)
        Returns:
            outputs: (seq_len, output_size)
            states: (seq_len, hidden_size)
        Note: ESNs are typically trained on single long sequences, so we assume batch=1.
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0) # Add batch dim
            
        b, seq_len, _ = inputs.shape
        states = []
        
        if initial_state is None:
            state = torch.zeros(b, self.hidden_size, device=inputs.device)
        else:
            state = initial_state

        for t in range(seq_len):
            u_t = inputs[:, t, :] # (b, input_size)
            
            # The core ESN update equation
            # x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_in * u[t] + W_res * x[t-1])
            update = torch.tanh(u_t @ self.W_in.T + state @ self.W_res.T)
            state = (1 - self.leaky_rate) * state + self.leaky_rate * update
            states.append(state)
            
        # Stack states to (b, seq_len, hidden_size)
        states = torch.stack(states, dim=1)
        
        # To compute output, we typically concatenate [1; u[t]; x[t]]
        # (batch, seq_len, 1)
        bias = torch.ones(b, seq_len, 1, device=inputs.device)
        extended_states = torch.cat([bias, inputs, states], dim=-1)
        
        # Compute output using W_out
        outputs = extended_states @ self.W_out.T
        
        return outputs.squeeze(0), states.squeeze(0)

    def fit(self, inputs, targets, washout=100, ridge_lambda=1e-4):
        """
        Trains the W_out matrix using Ridge Regression (Tikhonov Regularization).
        washout: Number of initial timesteps to discard to allow the reservoir memory to warm up.
        """
        if inputs.dim() == 2:
             inputs = inputs.unsqueeze(0)
        if targets.dim() == 2:
             targets = targets.unsqueeze(0)
             
        b, seq_len, _ = inputs.shape
        target_seq_len = targets.shape[1]
        
        # 1. Harvest states by running data through reservoir
        # We don't care about outputs here, just the internal states
        _, states = self.forward(inputs) # states: (seq_len, hidden_size)
        inputs_squeezed = inputs.squeeze(0)
        targets_squeezed = targets.squeeze(0)
        
        # 2. Discard the washout period
        # If the input and target are offset (e.g. predicting next step), ensure lengths match
        X = states[washout:target_seq_len]
        U = inputs_squeezed[washout:target_seq_len]
        Y = targets_squeezed[washout:]
        
        if len(Y) != len(X):
            min_len = min(len(Y), len(X))
            X = X[:min_len]
            U = U[:min_len]
            Y = Y[:min_len]

        # 3. Construct extended state matrix (add bias and input)
        bias = torch.ones(X.shape[0], 1, device=X.device)
        extended_states = torch.cat([bias, U, X], dim=1) # Shape: (N, 1 + input_size + hidden_size)
        
        # 4. Solve Ridge Regression: W_out = Y^T x X (X^T x X + lambda * I)^-1
        # It's numerically more stable to solve (X^T X + lambda I) W^T = X^T Y
        S = extended_states
        identity = torch.eye(S.shape[1], device=S.device)
        
        # S^T * S
        STS = S.T @ S
        # S^T * Y
        STY = S.T @ Y
        
        # Solve the linear system
        W_out_T = torch.linalg.solve(STS + ridge_lambda * identity, STY)
        
        # Update weights (transpose back to match parameter shape)
        self.W_out.data = W_out_T.T


class TestESN(unittest.TestCase):
    def test_spectral_radius(self):
        esn = EchoStateNetwork(input_size=1, hidden_size=100, output_size=1, spectral_radius=0.8)
        eigenvalues = torch.linalg.eigvals(esn.W_res)
        actual_sr = torch.max(torch.abs(eigenvalues)).item()
        self.assertAlmostEqual(actual_sr, 0.8, places=4)
        
    def test_forward_shapes(self):
        esn = EchoStateNetwork(input_size=2, hidden_size=50, output_size=1)
        u = torch.randn(10, 2)
        out, states = esn(u)
        self.assertEqual(states.shape, (10, 50))
        self.assertEqual(out.shape, (10, 1))
        
    def test_fit_reduces_error(self):
        esn = EchoStateNetwork(input_size=1, hidden_size=100, output_size=1, spectral_radius=0.95)
        # Simple sine wave task
        t = torch.linspace(0, 20, 200).unsqueeze(1)
        u = torch.sin(t)
        # Target is shifted sine wave
        y = torch.sin(t + 0.1)
        
        # Error before training
        out_before, _ = esn(u)
        err_before = nn.functional.mse_loss(out_before[20:], y[20:])
        
        # Fit
        esn.fit(u, y, washout=20)
        
        # Error after training
        out_after, _ = esn(u)
        err_after = nn.functional.mse_loss(out_after[20:], y[20:])
        
        self.assertTrue(err_after < err_before)
        self.assertTrue(err_after < 1e-2)

if __name__ == '__main__':
    unittest.main()
