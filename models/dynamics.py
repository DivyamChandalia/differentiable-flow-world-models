import torch
import torch.nn as nn
from .common import ConditionalTransformer, SinusoidalTimeEmbedding


class FlowMatchingDynamics(nn.Module):
    """Conditional Flow Matching dynamics model.

    Instead of predicting next-states via a single transformer pass with
    placeholder inputs, this model learns a velocity field v_θ(z_t, t | ctx, a)
    and integrates it from t=0 (noise) to t=1 (predicted states) using Euler
    steps.  The entire ODE solve is differentiable, enabling analytical
    gradients for actor training.

    Training:
        loss, pred_states = model(ctx_states, action_emb, target_states)

    Inference (imagination):
        pred_states = model.generate(ctx_states, action_emb)
    """

    def __init__(
        self,
        max_frames,
        action_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout=0.0,
        causal=False,
        num_euler_steps=6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_euler_steps = num_euler_steps
        self.horizon = max_frames

        # Time embedding: scalar t → hidden_dim vector
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # Projection that fuses the action embedding with the time embedding
        # before feeding into AdaLN conditioning.
        self.cond_fuse = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )

        # Learnable conditioning token for context positions
        # (context states are clean — they don't need action conditioning,
        #  but AdaLN requires a conditioning vector per position)
        self.ctx_cond_token = nn.Parameter(torch.zeros(1, 1, action_dim))
        nn.init.normal_(self.ctx_cond_token, std=0.02)

        # Velocity network — same architecture as the old dynamics transformer.
        # Input:  [ctx_states ; z_t]   (B, C + H, D)
        # Cond:   [ctx_cond   ; fused] (B, C + H, action_dim)
        # Output: predicted velocity for the z_t positions.
        self.velocity_net = ConditionalTransformer(
            depth=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            cond_dim=action_dim,
            seq_len=max_frames,
            causal=causal,
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def compute_velocity(self, z_t, t, ctx_states, future_action_emb):
        """Evaluate the velocity field v_θ(z_t, t | ctx, actions).

        Args:
            z_t:               (B, H, D)        noisy / interpolated future states
            t:                 (B,) or (B, 1)    flow timestep in [0, 1]
            ctx_states:        (B, C, D)         real context frames (clean)
            future_action_emb: (B, H, action_dim) action embeddings for 
                               predicted future steps

        Returns:
            velocity:          (B, H, D)  predicted velocity at positions of z_t
        """
        B, H, D = z_t.shape
        C = ctx_states.size(1)

        # Concat context (clean) + noisy future tokens → transformer input
        x = torch.cat([ctx_states, z_t], dim=1)  # (B, C + H, D)

        # Fuse time embedding into the future action embeddings
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, H, -1)  # (B, H, hidden_dim)

        fused = self.cond_fuse(
            torch.cat([future_action_emb, t_emb_expanded], dim=-1)
        )  # (B, H, action_dim)

        # Expand learnable context conditioning to batch
        ctx_cond = self.ctx_cond_token.expand(B, C, -1)  # (B, C, action_dim)

        # Full conditioning: [ctx_cond ; fused_future]
        cond = torch.cat([ctx_cond, fused], dim=1)  # (B, C + H, action_dim)

        # Run through transformer velocity network (no KV cache)
        out = self.velocity_net(x, cond, kv_cache=None)  # (B, C + H, D)

        # Extract velocity predictions for the future positions only
        velocity = out[:, C:]  # (B, H, D)

        return velocity

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, ctx_states, future_action_emb, target_states):
        """Training mode: compute velocity matching loss.

        Args:
            ctx_states:        (B, C, D)          real encoded context frames
            future_action_emb: (B, H, action_dim) action embeddings for future steps
            target_states:     (B, H, D)          ground-truth next states

        Returns:
            velocity_loss: scalar  MSE velocity matching loss
        """
        B, H, D = target_states.shape

        # Sample noise and flow time
        z_0 = torch.randn_like(target_states)            # (B, H, D)
        t = torch.rand(B, device=target_states.device)   # (B,)

        # Optimal transport interpolation: z_t = (1-t)*z_0 + t*z_1
        t_expanded = t[:, None, None]                     # (B, 1, 1)
        z_t = (1.0 - t_expanded) * z_0 + t_expanded * target_states

        # Target velocity: v* = z_1 - z_0
        velocity_target = target_states - z_0

        # Predicted velocity
        velocity_pred = self.compute_velocity(z_t, t, ctx_states, future_action_emb)

        # Velocity matching loss
        velocity_loss = torch.nn.functional.mse_loss(velocity_pred, velocity_target)

        return velocity_loss

    # ------------------------------------------------------------------
    # Inference / Imagination
    # ------------------------------------------------------------------

    def generate(self, ctx_states, future_action_emb, num_steps=None, noise=None):
        """Euler integration from t=0 (noise) to t=1 (predicted states).

        Fully differentiable — gradients flow through all Euler steps.

        Args:
            ctx_states:        (B, C, D)          real context frames
            future_action_emb: (B, H, action_dim) action embeddings for future steps
            num_steps:         int, optional       override default Euler steps
            noise:             (B, H, D), optional fixed noise tensor

        Returns:
            pred_states: (B, H, D)  generated future latent states
        """
        return self._euler_solve(ctx_states, future_action_emb, num_steps=num_steps, noise=noise)

    # ------------------------------------------------------------------
    # Internal Euler solver
    # ------------------------------------------------------------------

    def _euler_solve(self, ctx_states, future_action_emb, num_steps=None, noise=None):
        """Run K-step Euler integration of the learned velocity field.

        z_{k+1} = z_k + dt * v_θ(z_k, t_k, ctx, actions)

        Args:
            ctx_states:        (B, C, D)
            future_action_emb: (B, H, action_dim)
            num_steps:         int
            noise:             (B, H, D)

        Returns:
            z: (B, H, D) — predicted states at t=1
        """
        K = num_steps or self.num_euler_steps
        dt = 1.0 / K

        H = future_action_emb.size(1)

        if noise is not None:
            z = noise
        else:
            z = torch.randn(
                ctx_states.size(0), H, ctx_states.size(2),
                device=ctx_states.device, dtype=ctx_states.dtype,
            )

        for k in range(K):
            t_k = torch.full(
                (z.size(0),), k * dt,
                device=z.device, dtype=z.dtype,
            )
            v = self.compute_velocity(z, t_k, ctx_states, future_action_emb)
            z = z + dt * v

        return z
