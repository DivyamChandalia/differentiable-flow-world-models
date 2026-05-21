import torch
import torch.nn as nn
from .common import SwiGLUMLP, init_linear_orthogonal
from torch.distributions import Normal
import numpy as np
import math
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, action_space, obs_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_space.shape[0]
        
        high = torch.tensor(action_space.high, dtype=torch.float32)
        low = torch.tensor(action_space.low, dtype=torch.float32)
        
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
        
        self.mlp = SwiGLUMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim
        )
        
        self._initialize_weights()
        
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

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

    def get_distribution(self, obs):
        raw_mean = self.mlp(obs)
        
        log_std = self.log_std.expand_as(raw_mean)
        
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
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

        log_prob = dist.log_prob(x)
        
        log_prob_correction = 2 * (math.log(2) - x - F.softplus(-2 * x))
        log_prob_correction = log_prob_correction + torch.log(self.action_scale)
        
        log_prob = log_prob - log_prob_correction
        entropy = -log_prob
        
        return final_action, log_prob, entropy
    
class Value(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.mlp = SwiGLUMLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1
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
        