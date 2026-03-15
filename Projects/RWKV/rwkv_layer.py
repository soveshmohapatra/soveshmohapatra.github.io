import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class RWKVTimeMixer(nn.Module):
    """The core Time Mixing (TM) module in RWKV."""
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.time_decay = nn.Parameter(torch.ones(embed_dim))
        self.time_first = nn.Parameter(torch.ones(embed_dim))
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln = nn.LayerNorm(embed_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.time_decay, 0.0)
        nn.init.constant_(self.time_first, 1.0)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.receptance.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def _get_wkv_weights(self, B, C, device):
        time_decay = torch.sigmoid(self.time_decay).view(1, 1, C)
        time_first = torch.sigmoid(self.time_first).view(1, 1, C)
        return time_decay, time_first
    
    def forward_parallel(self, x):
        B, T, C = x.shape
        x = self.ln(x)
        k, v, r = self.key(x), self.value(x), self.receptance(x)
        time_decay, time_first = self._get_wkv_weights(B, C, x.device)
        kv = k * v
        wkv = self._compute_wkv_parallel(kv, time_decay, time_first, T)
        return self.output(torch.sigmoid(r) * wkv)
    
    def _compute_wkv_parallel(self, kv, time_decay, time_first, T):
        B, T, C = kv.shape
        time_decay = time_decay.expand(B, T, C)
        time_first = time_first.expand(B, T, C)
        time_indices = torch.arange(T, device=kv.device).view(1, T, 1)
        weighted_kv = kv * torch.pow(time_decay, time_indices)
        cumsum = torch.cumsum(weighted_kv, dim=1)
        wkv = torch.zeros_like(kv)
        wkv[:, 0] = time_first[:, 0] * kv[:, 0]
        if T > 1:
            wkv[:, 1:] = time_first[:, 1:] * kv[:, 1:] + time_decay[:, 1:] * cumsum[:, :-1]
        return wkv
    
    def forward_recurrent(self, x, state=None):
        if x.dim() == 2:  # (B, C) - single token
            B, C = x.shape
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            B, T, C = x.shape
            squeeze_output = False
        
        x = self.ln(x)
        time_decay, time_first = self._get_wkv_weights(B, C, x.device)
        wkv_state = torch.zeros(B, C, device=x.device) if state is None else state[0]
        outputs = []
        
        for t in range(x.shape[1]):
            x_t = x[:, t]
            k_t, v_t, r_t = self.key(x_t), self.value(x_t), self.receptance(x_t)
            wkv_t = time_first[:, 0] * k_t * v_t + time_decay[:, 0] * wkv_state
            outputs.append(self.output(torch.sigmoid(r_t) * wkv_t))
            wkv_state = wkv_t
        
        output = torch.stack(outputs, dim=1)
        if squeeze_output:
            output = output.squeeze(1)
        return output, (wkv_state,)
    
    def forward(self, x, state=None, use_recurrent=False):
        return self.forward_recurrent(x, state) if use_recurrent else (self.forward_parallel(x), None)


class RWKVChannelMixer(nn.Module):
    """Channel Mixing module with squared ReLU."""
    def __init__(self, embed_dim, expand_factor=4):
        super().__init__()
        self.hidden_dim = embed_dim * expand_factor
        self.key = nn.Linear(embed_dim, self.hidden_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        self.ln = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.receptance.weight)
        nn.init.xavier_uniform_(self.value.weight)
    
    def forward(self, x):
        x = self.ln(x)
        return torch.sigmoid(self.receptance(x)) * self.value(F.relu(self.key(x)) ** 2)


class RWKVBlock(nn.Module):
    """Complete RWKV block."""
    def __init__(self, embed_dim, num_heads=1, expand_factor=4):
        super().__init__()
        self.time_mixer = RWKVTimeMixer(embed_dim, num_heads)
        self.channel_mixer = RWKVChannelMixer(embed_dim, expand_factor)
    
    def forward(self, x, state=None, use_recurrent=False):
        if use_recurrent:
            tm_out, new_state = self.time_mixer(x, state, use_recurrent=True)
            x = x + tm_out
        else:
            tm_out, _ = self.time_mixer(x)
            x = x + tm_out
            new_state = None
        return x + self.channel_mixer(x), new_state


class RWKV(nn.Module):
    """Complete RWKV model for sequence modeling."""
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads=1, expand_factor=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([RWKVBlock(embed_dim, num_heads, expand_factor) for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.ones_(self.ln_out.weight)
        nn.init.zeros_(self.ln_out.bias)
    
    def forward(self, x, states=None, use_recurrent=False):
        """
        Args:
            x: Token indices. Either (B,) for single token or (B, T) for sequence.
        Returns:
            logits: (B, vocab_size) for single token, (B, T, vocab_size) for sequence.
        """
        was_1d = (x.dim() == 1)
        if was_1d:
            x = x.unsqueeze(1)
        
        x = self.embedding(x)
        states = [None] * self.num_layers if states is None else states
        new_states = []
        
        for i, block in enumerate(self.blocks):
            if use_recurrent:
                x, new_state = block(x, states[i], use_recurrent=True)
                new_states.append(new_state)
            else:
                x, _ = block(x)
        
        logits = self.head(self.ln_out(x))
        
        if was_1d:
            logits = logits.squeeze(1)
        
        return (logits, new_states) if use_recurrent else (logits, None)
    
    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens, temperature=1.0):
        self.eval()
        context = prompt_tokens.clone()
        states = None
        
        # Process prompt - use parallel mode for efficiency
        if context.shape[1] > 1:
            logits, _ = self(context, use_recurrent=False)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V) - get last token
            # Initialize state by running through recurrent mode
            for i, block in enumerate(self.blocks):
                # Get state from last token
                pass  # States will be initialized on first recurrent call
            states = None  # Simplified: re-init states for generation
        
        for _ in range(max_new_tokens):
            next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
            context = torch.cat([context, next_token], dim=1)
            # Pass single token (1D) for recurrent mode
            logits, states = self(next_token.squeeze(1), states, use_recurrent=True)
        
        return context


class TestRWKV(unittest.TestCase):
    def setUp(self):
        self.model = RWKV(vocab_size=100, embed_dim=64, num_layers=2)
        self.batch_size, self.seq_len = 4, 16
    
    def test_forward_parallel_shapes(self):
        x = torch.randint(0, 100, (self.batch_size, self.seq_len))
        logits, states = self.model(x, use_recurrent=False)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 100))
        self.assertIsNone(states)
    
    def test_forward_recurrent_shapes(self):
        x = torch.randint(0, 100, (self.batch_size,))
        logits, states = self.model(x, use_recurrent=True)
        self.assertEqual(logits.shape, (self.batch_size, 100))
        self.assertEqual(len(states), 2)
    
    def test_generation_shapes(self):
        prompt = torch.randint(0, 100, (self.batch_size, 8))
        with torch.no_grad():
            generated = self.model.generate(prompt, max_new_tokens=10)
        self.assertEqual(generated.shape, (self.batch_size, 18))
    
    def test_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nRWKV model parameters: {total_params:,}")
        self.assertGreater(total_params, 10000)
    
    def test_gradient_flow(self):
        x = torch.randint(0, 100, (self.batch_size, self.seq_len))
        target = torch.randint(0, 100, (self.batch_size, self.seq_len))
        logits, _ = self.model(x, use_recurrent=False)
        loss = F.cross_entropy(logits.view(-1, 100), target.view(-1))
        loss.backward()
        self.assertIsNotNone(self.model.embedding.weight.grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
