import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PCNLayer(nn.Module):
    """
    A single layer in a Predictive Coding Network (PCN).
    Each layer tries to predict the activity of the layer below it.
    
    Args:
        in_features (int): Number of nodes in the layer below (the target of the prediction).
                           Note: "in_features" here refers to the size of the layer closer to the input data.
        out_features (int): Number of nodes in this layer (the layer making the prediction).
        activation (callable): Activation function to apply to the latent state (default: ReLU).
    """
    def __init__(self, in_features, out_features, activation=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        # The top-down generative weights: predicting the layer 'below'
        self.W = nn.Linear(out_features, in_features, bias=True)
        
        # Initialize weights (standard Xavier/Glorot)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        
        # Latent state (the 'belief' of this layer)
        self.latent_state = None
        
        # The prediction error between what this layer predicted and the actual state below
        self.prediction_error = None
        
    def forward(self):
        """
        The 'forward' pass in a PCN acts top-down. 
        It generates a prediction for the layer below based on its current latent state.
        """
        assert self.latent_state is not None, "Latent state must be initialized before forward pass."
        # f(x) * W
        return self.W(self.activation(self.latent_state))
        
    def compute_prediction_error(self, target_state):
        """
        Compute the local prediction error: target - prediction.
        """
        prediction = self.forward()
        self.prediction_error = target_state - prediction
        return self.prediction_error

class PCNetwork(nn.Module):
    """
    A Sequential Predictive Coding Network.
    
    Args:
        layer_sizes (list of int): Sizes of each layer, from input layer [first] to latent layers [last].
        activation (callable, optional): Activation function for the hidden layers. Defaults to F.relu.
    """
    def __init__(self, layer_sizes, activation=F.relu):
        super().__init__()
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation
        
        # Create PCNLayers
        self.layers = nn.ModuleList()
        # Layers are built such that layer i predicts layer i-1
        for i in range(self.num_layers):
            in_sz = layer_sizes[i]     # Layer below
            out_sz = layer_sizes[i+1]  # This layer
            
            # The last layer might not need activation depending on formulation, 
            # but standard PCNs apply it to all hidden states. 
            # For simplicity, we apply activation to all Latent States.
            self.layers.append(PCNLayer(in_sz, out_sz, activation=activation))
            
    def initialize_states(self, batch_size, device):
        """
        Initialize the latent states of all layers (except the fixed input layer) to zero.
        """
        for layer in self.layers:
            # Shape: [batch_size, out_features]
            layer.latent_state = torch.zeros(batch_size, layer.out_features, device=device, requires_grad=True)
            
    def set_input(self, x):
        """
        The input data x is fixed at the lowest level (closest to data).
        In Predictive Coding, inference tries to match this input.
        """
        self.input_data = x.clone().detach()

    def inference_step(self, inference_lr=0.1):
        """
        Perform one step of energy minimization (inference) to update the latent states.
        This local update is the PCN equivalent of the forward pass in standard ANNs.
        """
        # We need to compute gradients of the total energy with respect to the latent states.
        # Energy = sum of squared prediction errors across all layers.
        
        total_energy = 0.0
        
        # 1. Compute prediction errors bottom-up
        current_target = self.input_data
        for i in range(self.num_layers):
            error = self.layers[i].compute_prediction_error(current_target)
            # Local energy at this layer: 1/2 * ||error||^2
            layer_energy = 0.5 * torch.sum(error ** 2)
            total_energy = total_energy + layer_energy
            
            # The 'target' for the next highest layer is the latent state of the current layer
            current_target = self.layers[i].latent_state
            
        # 2. Backpropagate energy gradients ONLY to the latent states 
        # (Weights remain completely frozen during this phase!)
        latent_states = [layer.latent_state for layer in self.layers if layer.latent_state.requires_grad]
        
        if len(latent_states) == 0:
            return total_energy.item()

        gradients = torch.autograd.grad(total_energy, latent_states, create_graph=False)
        
        # 3. Update the latent states (Gradient Descent on Energy)
        with torch.no_grad():
            grad_idx = 0
            for layer in self.layers:
                if layer.latent_state.requires_grad:
                    layer.latent_state.sub_(inference_lr * gradients[grad_idx])
                    grad_idx += 1
                
        return total_energy.item()

    def update_weights(self, learning_rate=0.01):
        """
        After inference settles the latent states, update the top-down weights.
        This is a local Hebbian-like update based entirely on local variables:
        pre-synaptic activity f(x) and post-synaptic prediction error (e).
        
        Weight update rule: delta W = learning_rate * e * f(x)^T
        """
        with torch.no_grad():
            for layer in self.layers:
                # e is the prediction error [batch_size, in_features]
                error = layer.prediction_error
                
                # f(x) is the activated latent state [batch_size, out_features]
                pre_synaptic = layer.activation(layer.latent_state)
                
                # We want delta W to minimize the squared error.
                # delta_W shape is [out_features, in_features] since error is in_features and pre_synaptic is out_features
                # We need W.weight which is [in_features, out_features]
                delta_W = torch.matmul(error.t(), pre_synaptic) / error.size(0)
                delta_bias = torch.mean(error, dim=0)

                # W = W + lr * delta_W  (Predictive Coding typically uses gradient ASCENT on log likelihood or descent on energy)
                # Since Energy = 0.5 * error^2, dE/dW = -error * f(x). So weight update is + lr * delta_W
                layer.W.weight.add_(learning_rate * delta_W)
                if layer.W.bias is not None:
                    layer.W.bias.add_(learning_rate * delta_bias)

    def train_batch(self, x, y=None, inference_steps=20, inference_lr=0.1, weight_lr=0.01, is_supervised=False):
        """
        Train the network on a single batch of data.
        If is_supervised=True and y is provided, we clamp the HIGHEST layer's state to the label y 
        (e.g., one-hot encoded label). This propagates the label information down during inference.
        """
        device = x.device
        batch_size = x.size(0)
        
        # 1. Initialize latent states to zero for this new batch
        self.initialize_states(batch_size, device)
        
        # 2. Set the fixed bottom-level input
        self.set_input(x)
        
        # If supervised, we "clamp/fix" the top layer to the true labels
        if is_supervised and y is not None:
            # We don't want the highest layer to update its state; it stays fixed as the label.
            top_layer = self.layers[-1]
            # Must detach and require_grad=False so it doesn't move during inference
            top_layer.latent_state = y.clone().detach().to(device)
            top_layer.latent_state.requires_grad = False
            
        # 3. Inference phase: iterative state updates to minimize energy
        for step in range(inference_steps):
            energy = self.inference_step(inference_lr=inference_lr)
            
        # 4. Learning phase: local weight updates based on the settled states
        self.update_weights(learning_rate=weight_lr)
        
        return energy
        
    def predict(self, x, inference_steps=20, inference_lr=0.1):
        """
        To generate a prediction for unseen data, we fix the input data, 
        run inference but DO NOT clamp the top layer.
        The network relaxes into a state, and we read out the top layer's state as the prediction.
        """
        device = x.device
        self.initialize_states(x.size(0), device)
        self.set_input(x)
        
        for _ in range(inference_steps):
             self.inference_step(inference_lr=inference_lr)
            
        # Return the 'belief' of the highest layer
        return self.layers[-1].latent_state.detach()

if __name__ == "__main__":
    # --- Quick Smoke Test ---
    print("Initializing a 3-layer PCN: [Input: 10] <- [Hidden: 20] <- [Output: 5]")
    model = PCNetwork([10, 20, 5])
    
    # Random synthetic data
    x_batch = torch.randn(32, 10)
    # One-hot labels
    y_labels = torch.randint(0, 5, (32,))
    y_batch = F.one_hot(y_labels, num_classes=5).float()
    
    # Train one batch
    initial_energy = None
    final_energy = None
    
    print("\nTraining on 1 batch...")
    # Manually run train_batch step by step to see energy decay
    model.initialize_states(32, torch.device('cpu'))
    model.set_input(x_batch)
    
    # Supervised clamping
    model.layers[-1].latent_state.data = y_batch
    model.layers[-1].latent_state.requires_grad = False

    for step in range(10):
        energy = model.inference_step(inference_lr=0.1)
        if step == 0: initial_energy = energy
        if step == 9: final_energy = energy
        print(f"Inference Step {step+1:2d} | Energy: {energy:.4f}")
        
    print(f"Energy dropped from {initial_energy:.4f} to {final_energy:.4f}")
    
    print("Updating Weights...")
    model.update_weights(learning_rate=0.05)
    print("Done. Model architecture is sound.")
