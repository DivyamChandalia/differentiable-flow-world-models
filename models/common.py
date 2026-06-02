import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (RoPE).
    dim: head dimension
    end: maximum sequence length
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to queries and keys.
    xq, xk shapes: (Batch, Num_Heads, Seq_Len, Head_Dim)
    freqs_cis shape: (Seq_Len, Head_Dim // 2)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def init_linear_orthogonal(module, gain=1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class SwiGLUMLP(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim=None, 
        dropout_prob=0.0, 
        norm_fn=nn.LayerNorm
    ):
        super().__init__()

        self.up_proj = nn.Linear(input_dim, hidden_dim * 2, bias=False)
        self.swiglu = SwiGLU()

        self.norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()

        self.down_proj = nn.Linear(hidden_dim, output_dim or input_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.swiglu(x)
        x = self.norm(x)
        x = self.dropout(x)        
        x = self.down_proj(x)
        return x



class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            
        new_kv_cache = (k, v)
        
        is_causal = (T > 1) if self.causal else False
        
        x = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        
        return x, new_kv_cache

class AdaLNTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4.0, causal=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        
        self.attn = SelfAttention(dim, num_heads, causal=causal)
        
        mlp_hidden_dim = int(dim * mlp_ratio * (2/3))
        self.mlp = SwiGLUMLP(dim, mlp_hidden_dim) 

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )

        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(self, x, c, freqs_cis, kv_cache=None):
        (shift_msa, scale_msa, gate_msa, 
         shift_mlp, scale_mlp, gate_mlp) = self.adaln_modulation(c).chunk(6, dim=-1)

        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        
        attn_out, new_kv_cache = self.attn(
            x_norm1, 
            freqs_cis=freqs_cis, 
            kv_cache=kv_cache
        )
        
        x = x + gate_msa * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        
        x = x + gate_mlp * mlp_out

        return x, new_kv_cache

class ConditionalTransformer(nn.Module):
    def __init__(self, depth, dim, num_heads, cond_dim, seq_len, window_size=None, causal=True):
        super().__init__()
        self.window_size = window_size
        self.head_dim = dim // num_heads
        
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(dim, num_heads, cond_dim, causal=causal) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, c, kv_cache=None):
        start_pos = 0
        if kv_cache is not None and len(kv_cache) > 0:
            start_pos = kv_cache[0][0].size(2)
            
        seq_len = x.size(1)
        
        if self.window_size is not None and start_pos + seq_len > self.window_size:
            raise ValueError(f"Total sequence length ({start_pos + seq_len}) exceeds window_size ({self.window_size})")
        
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
            
        new_kv_cache = [] if kv_cache is not None else None
        
        for i, block in enumerate(self.blocks):
            block_cache = kv_cache[i] if (kv_cache is not None and len(kv_cache) > 0) else None
            
            x, updated_cache = block(
                x, c, 
                freqs_cis=freqs_cis, 
                kv_cache=block_cache
            )
            
            if new_kv_cache is not None:
                new_kv_cache.append(updated_cache)
        
        x = self.norm(x)

        if new_kv_cache is not None:
            return x, new_kv_cache
        return x