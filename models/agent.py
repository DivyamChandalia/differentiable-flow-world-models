import torch
import torch.nn as nn
from .common import SwiGLUMLP, init_linear_orthogonal
from torch.distributions import Normal
import numpy as np
import math
import torch.nn.functional as F

class DeepResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2, dropout_prob=0.0, norm_fn=nn.LayerNorm):
        super().__init__()
        self.input_proj = SwiGLUMLP(input_dim, hidden_dim, hidden_dim, dropout_prob, norm_fn)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                norm_fn(hidden_dim),
                SwiGLUMLP(hidden_dim, hidden_dim, hidden_dim, dropout_prob, norm_fn=None)
            )
            for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)
class Actor(nn.Module):
    def __init__(self, action_space, obs_dim, hidden_dim, chunk_size=15, num_blocks=2):
        super().__init__()
        self.action_dim = action_space.shape[0]
        self.chunk_size = chunk_size
        
        high = torch.tensor(action_space.high, dtype=torch.float32)
        low = torch.tensor(action_space.low, dtype=torch.float32)
        
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
        
        # FIX 1: Double the output dim to predict both mean and log_std dynamically
        self.mlp = DeepResNetMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * self.action_dim * self.chunk_size, 
            num_blocks=num_blocks
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init_linear_orthogonal(module, gain=np.sqrt(2))
                
        final_layer = self._get_final_layer()
        if final_layer is not None:
            # Keep the final layer initialization small so the network starts near origin
            init_linear_orthogonal(final_layer, gain=0.01)

    def _get_final_layer(self):
        modules = list(self.mlp.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def get_distribution(self, obs):
        out = self.mlp(obs)
        
        # Split the output into mean and log_std
        raw_mean, log_std = torch.chunk(out, 2, dim=-1)
        
        raw_mean = raw_mean.view(*obs.shape[:-1], self.chunk_size, self.action_dim)
        log_std = log_std.view(*obs.shape[:-1], self.chunk_size, self.action_dim)
        
        # FIX 2: -20 is too small for stable gradients. -5 or -10 is a safer floor.
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        
        return Normal(raw_mean, std)

    def forward(self, obs, action=None, deterministic=False):
        dist = self.get_distribution(obs)
        
        if action is not None:
            y = (action - self.action_bias) / self.action_scale
            y = torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6)
            x = torch.atanh(y)
            final_action = action
        else:
            if deterministic:
                x = dist.mean
            else:
                x = dist.rsample()
                
            y = torch.tanh(x)
            final_action = y * self.action_scale + self.action_bias

        # Calculate joint log probability
        log_prob = dist.log_prob(x)
        
        # Tanh Squashing Correction
        log_prob_correction = 2 * (math.log(2) - x - F.softplus(-2 * x))
        log_prob_correction = log_prob_correction + torch.log(self.action_scale)
        
        log_prob = log_prob - log_prob_correction
        
        # FIX 3: Sum across the action dimension to get the true joint log probability
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Return the analytical entropy of the base distribution to the loss function
        # instead of the sampled squashed entropy.
        analytical_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return final_action, log_prob, analytical_entropy
    
class Value(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_blocks=2):
        super().__init__()
        self.mlp = DeepResNetMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_blocks=num_blocks
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                init_linear_orthogonal(module, gain=np.sqrt(2))
                
        final_layer = self._get_final_layer()
        if final_layer is not None:
            init_linear_orthogonal(final_layer, gain=0.01)

    def _get_final_layer(self):
        modules = list(self.mlp.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        return linear_layers[-1] if linear_layers else None

    def forward(self, obs):
        value = self.mlp(obs)
        return value
        