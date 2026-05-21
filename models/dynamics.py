import torch.nn as nn
from .common import ConditionalTransformer

class Dynamics(nn.Module):
    def __init__(self,
                 max_frames,
                 action_dim,
                 hidden_dim,
                 num_layers,
                 num_heads,
                 dropout=0.0,
                 ):
        super().__init__()
        self.transformer = ConditionalTransformer(
            depth=num_layers,
            dim=hidden_dim, 
            num_heads=num_heads, 
            cond_dim=action_dim,
            seq_len=max_frames
        )
        self.horizon = max_frames

    def forward(self, x, c, kv_cache=None):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        res = self.transformer(x, c, kv_cache=kv_cache)
        if kv_cache is not None:
            out, new_kv_cache = res
            return x + out, new_kv_cache
        else:
            return x + res
            
