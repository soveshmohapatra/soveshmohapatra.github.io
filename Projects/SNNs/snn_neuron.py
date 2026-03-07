import torch
import torch.nn as nn
import unittest


# ---------------------------------------------------------------------------
# Surrogate Gradient (Fast Sigmoid)
# ---------------------------------------------------------------------------
# The Heaviside step function (spike) is non-differentiable at 0.
# During the backward pass, we substitute its gradient with an approximation
# called the "surrogate gradient". We use the fast sigmoid here:
#   sigma'(x) = 1 / (1 + k|x|)^2
# Reference: Neftci et al., 2019 (Surrogate Gradient Learning in SNNs)
# ---------------------------------------------------------------------------

class SurrogateHeaviside(torch.autograd.Function):
    """
    Custom autograd function implementing the Heaviside step as the forward
    pass and the fast sigmoid derivative as the backward surrogate gradient.
    """

    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        # Save the membrane potential for backward gradient computation
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        # Fire a spike (1.0) wherever potential exceeds threshold
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_potential,) = ctx.saved_tensors
        threshold = ctx.threshold
        # Fast sigmoid surrogate: d/dx [ 1 / (1 + |x - vth|) ] approx.
        # k=10 is a common sharpness factor
        k = 10.0
        shifted = membrane_potential - threshold
        surrogate = 1.0 / (1.0 + k * shifted.abs()) ** 2
        return grad_output * surrogate, None   # None for threshold (not a tensor grad)


def spike_function(membrane_potential: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convenience wrapper around SurrogateHeaviside."""
    return SurrogateHeaviside.apply(membrane_potential, threshold)


# ---------------------------------------------------------------------------
# Leaky Integrate-and-Fire (LIF) Neuron Layer
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """
    A minimal, educational implementation of a Leaky Integrate-and-Fire (LIF)
    neuron layer in pure PyTorch.

    The discrete-time LIF update rule per time-step t is:
        V[t] = decay * V[t-1] + I[t]      (integrate + leak)
        S[t] = Θ(V[t] - v_threshold)      (fire)
        V[t] = V[t] * (1 - S[t])          (reset: soft reset to 0)

    where:
        V[t]         : membrane potential at time t  (batch, num_neurons)
        decay        : membrane leak factor ∈ (0, 1), equivalent to exp(-dt/tau)
        I[t]         : synaptic input current at time t (from previous layer)
        S[t]         : output spike train (binary: 0 or 1)
        v_threshold  : firing threshold (default 1.0)

    Gradients flow through S[t] via the surrogate fast-sigmoid gradient.

    References:
    - Gerstner & Kistler, "Spiking Neuron Models" (2002)
    - Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks" (2019)
    """

    def __init__(
        self,
        num_neurons: int,
        decay: float = 0.9,
        threshold: float = 1.0,
    ):
        """
        Args:
            num_neurons:  Number of neurons (output dimensionality).
            decay:        Membrane potential leak factor ∈ (0, 1).
                          Higher value → slower leak → longer memory.
            threshold:    Membrane potential firing threshold.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        # Learnable decay per neuron, initialized to the given value
        # Sigmoid ensures it stays in (0, 1) during training
        self._decay_logit = nn.Parameter(
            torch.full((num_neurons,), _inv_sigmoid(decay))
        )

    @property
    def decay(self) -> torch.Tensor:
        """Return the effective per-neuron decay factor ∈ (0, 1)."""
        return torch.sigmoid(self._decay_logit)

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ):
        """
        Process one time-step of LIF dynamics.

        Args:
            current:   Synaptic input at this time step.  Shape: (batch, num_neurons)
            membrane:  Membrane potential from previous step. Shape: (batch, num_neurons)

        Returns:
            spike:         Output spike tensor (0/1).  Shape: (batch, num_neurons)
            new_membrane:  Updated membrane potential.  Shape: (batch, num_neurons)
        """
        # 1. Leak + integrate
        new_membrane = self.decay * membrane + current

        # 2. Fire: Heaviside with surrogate gradient
        spike = spike_function(new_membrane, self.threshold)

        # 3. Reset: subtract threshold from fired neurons (soft reset)
        new_membrane = new_membrane - spike * self.threshold

        return spike, new_membrane

    def init_membrane(self, batch_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Return a zero-initialized membrane potential tensor."""
        return torch.zeros(batch_size, self.num_neurons, device=device)


# ---------------------------------------------------------------------------
# 2-Layer SNN Classifier (for MNIST)
# ---------------------------------------------------------------------------

class SNNClassifier(nn.Module):
    """
    A 2-hidden-layer Spiking Neural Network for image classification.

    Architecture (for MNIST):
        Input (flat pixels) → Linear → LIF → Linear → LIF → Linear (readout)

    The readout layer accumulates output spikes over T timesteps.
    Final prediction = argmax of total spike count.

    The same static input is presented at every timestep (rate coding).
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        output_size: int = 10,
        decay: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Synaptic weight matrices (standard linear, no bias on LIF layers)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # LIF neuron layers
        self.lif1 = LIFNeuron(hidden_size, decay=decay, threshold=threshold)
        self.lif2 = LIFNeuron(hidden_size, decay=decay, threshold=threshold)

    def forward(self, x: torch.Tensor, num_steps: int = 25) -> torch.Tensor:
        """
        Args:
            x:         Input batch. Shape: (batch, input_size).
                       Flattened pixel values, already scaled to [0, 1].
            num_steps: Number of simulation timesteps T.

        Returns:
            logits:    Accumulated spike count over T steps.
                       Shape: (batch, output_size). Use for cross-entropy.
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize membrane potentials
        mem1 = self.lif1.init_membrane(batch_size, device)
        mem2 = self.lif2.init_membrane(batch_size, device)

        # Accumulate output spikes over T timesteps
        spike_accumulator = torch.zeros(batch_size, self.output_size, device=device)

        for _ in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Readout (accumulate raw logits, not spikes)
            spike_accumulator += self.fc_out(spk2)

        return spike_accumulator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _inv_sigmoid(y: float) -> float:
    """Compute logit (inverse sigmoid) so that sigmoid(logit(y)) == y."""
    import math
    return math.log(y / (1.0 - y))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestSurrogateHeaviside(unittest.TestCase):
    def test_forward_fires_above_threshold(self):
        v = torch.tensor([0.5, 1.0, 1.5, 2.0])
        s = spike_function(v, threshold=1.0)
        # Should fire at 1.0, 1.5, 2.0 (>= threshold)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.equal(s, expected))

    def test_forward_no_fire_below_threshold(self):
        v = torch.tensor([0.0, 0.5, 0.99])
        s = spike_function(v, threshold=1.0)
        expected = torch.zeros(3)
        self.assertTrue(torch.equal(s, expected))

    def test_backward_surrogate_is_nonzero(self):
        """Gradients should pass through the surrogate, not be zero."""
        v = torch.tensor([0.8, 1.0, 1.2], requires_grad=True)
        s = spike_function(v, threshold=1.0)
        loss = s.sum()
        loss.backward()
        # Surrogate gradient should be non-zero near the threshold
        self.assertIsNotNone(v.grad)
        self.assertTrue((v.grad != 0).any())


class TestLIFNeuron(unittest.TestCase):
    def setUp(self):
        self.layer = LIFNeuron(num_neurons=16, decay=0.9, threshold=1.0)
        self.batch = 8

    def test_output_shapes(self):
        current = torch.randn(self.batch, 16)
        membrane = self.layer.init_membrane(self.batch)
        spike, new_mem = self.layer(current, membrane)
        self.assertEqual(spike.shape, (self.batch, 16))
        self.assertEqual(new_mem.shape, (self.batch, 16))

    def test_spike_fires_when_threshold_exceeded(self):
        """Force a large current so all neurons must fire."""
        current = torch.full((self.batch, 16), 5.0)
        membrane = self.layer.init_membrane(self.batch)
        spike, _ = self.layer(current, membrane)
        self.assertTrue((spike == 1.0).all(), "All neurons should fire with large input")

    def test_no_spike_below_threshold(self):
        """With zero current and zero membrane, no neuron fires."""
        current = torch.zeros(self.batch, 16)
        membrane = self.layer.init_membrane(self.batch)
        spike, _ = self.layer(current, membrane)
        self.assertTrue((spike == 0.0).all(), "No neuron should fire with zero input")

    def test_voltage_reset_after_spike(self):
        """After a spike, the soft reset brings membrane strictly below threshold.
        With current=1.5 and zero initial membrane:
          integrate -> V = decay*0 + 1.5 = 1.5  -> spike fires
          soft reset -> V = 1.5 - 1.0 = 0.5    -> strictly below threshold (1.0)
        """
        current = torch.full((self.batch, 16), 1.5)  # Will fire (1.5 >= threshold=1.0)
        membrane = torch.zeros(self.batch, 16)
        spike, new_mem = self.layer(current, membrane)
        # All neurons should have fired
        self.assertTrue((spike == 1.0).all(), "All neurons should fire with current=1.5")
        # After soft reset, every membrane value should be strictly below threshold
        self.assertTrue(
            (new_mem < self.layer.threshold).all(),
            "Membrane should be strictly below threshold after soft reset (expected ~0.5)"
        )

    def test_decay_is_positive_and_lt_one(self):
        decay = self.layer.decay
        self.assertTrue((decay > 0).all())
        self.assertTrue((decay < 1).all())


class TestSNNClassifier(unittest.TestCase):
    def test_output_shape(self):
        model = SNNClassifier(input_size=784, hidden_size=64, output_size=10)
        x = torch.randn(4, 784)
        out = model(x, num_steps=5)
        self.assertEqual(out.shape, (4, 10))

    def test_parameter_count(self):
        model = SNNClassifier(input_size=784, hidden_size=256, output_size=10)
        n_params = count_parameters(model)
        print(f"\nSNNClassifier (hidden=256) parameter count: {n_params:,}")
        self.assertGreater(n_params, 0)

    def test_gradient_flows(self):
        """Ensure surrogate gradients allow backprop through the full model."""
        model = SNNClassifier(input_size=784, hidden_size=64, output_size=10)
        x = torch.randn(4, 784)
        out = model(x, num_steps=5)
        loss = out.sum()
        loss.backward()
        # Check that fc1 weight received gradients
        self.assertIsNotNone(model.fc1.weight.grad)
        self.assertTrue((model.fc1.weight.grad != 0).any(),
                        "fc1 should receive non-zero gradients via surrogate backward")


if __name__ == "__main__":
    unittest.main(verbosity=2)
