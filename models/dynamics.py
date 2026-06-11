import torch
import torch.nn as nn
from .common import ConditionalTransformer, SinusoidalTimeEmbedding


class FlowMatchingDynamics(nn.Module):
    """State-to-State Conditional Flow Matching dynamics model.

    Learns a velocity field v_θ(z_t, t | ctx, a) that transports the last
    known context state to each future target state in parallel.  The ODE
    interpolates from t=0 (source = last context state + noise) to t=1
    (predicted/target states) using Euler steps.  The entire solve is
    differentiable, enabling analytical gradients for actor training.

    Key design decisions vs. the previous version:
      • Input projection fuses [z_t, action_emb, t_emb] → hidden_dim so
        each future position has unique content from the very first layer.
      • A dedicated velocity head (MLP) maps transformer output → velocity
        without being gated by AdaLN — avoids zero-init gate starvation.
      • Source noise (σ = 0.25 by default) breaks the symmetry of identical
        broadcast source states across the prediction horizon.
      • Uniform time sampling t ~ U(0, 1) avoids the bias of logit-normal.

    Training:
        loss = model(ctx_states, action_emb, target_states, source_states)

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
        source_noise_sigma=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_euler_steps = num_euler_steps
        self.source_noise_sigma = source_noise_sigma
        self.horizon = max_frames

        # ── Input projection ────────────────────────────────────────────
        # Fuses [z_t ‖ action_emb ‖ t_emb] → hidden_dim so each future
        # position has unique content before the transformer sees it.
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable position embedding for future tokens — breaks symmetry
        # when all positions start from the same broadcast source state.
        self.horizon_pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.normal_(self.horizon_pos_embed, std=0.02)

        # ── Time embedding ──────────────────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)

        # ── Additive conditioning ───────────────────────────────────────
        # Fuse action + time into a conditioning vector for explicit
        # additive injection at each transformer layer.
        self.cond_proj = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )

        # Learnable conditioning token for context positions
        # (context states are clean — they don't need action conditioning,
        #  but the transformer requires a conditioning vector per position)
        self.ctx_cond_token = nn.Parameter(torch.zeros(1, 1, action_dim))
        nn.init.normal_(self.ctx_cond_token, std=0.02)

        # ── Velocity network backbone (additive conditioning) ──────────
        self.velocity_net = ConditionalTransformer(
            depth=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            cond_dim=action_dim,
            seq_len=max_frames,
            causal=causal,
            block_type='additive',
        )

        # ── Velocity output head ────────────────────────────────────────
        # Dedicated MLP that maps transformer output → velocity.
        # NOT gated by AdaLN — produces non-trivial output from init.
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Small-but-nonzero init for stable start
        nn.init.xavier_uniform_(self.velocity_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.velocity_head[-1].bias)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def compute_velocity(self, z_t, t, ctx_states, future_action_emb):
        """Evaluate the velocity field v_θ(z_t, t | ctx, actions).

        Args:
            z_t:               (B, H, D)          interpolated future states
            t:                 (B,) or (B, H)     flow timestep in [0, 1]
                               (B,) during Euler inference — same t for all positions.
                               (B, H) during training — per-position t sampling.
            ctx_states:        (B, C, D)           real context frames (clean)
            future_action_emb: (B, H, action_dim)  action embeddings for
                               predicted future steps

        Returns:
            velocity:          (B, H, D)  predicted velocity at positions of z_t
        """
        B, H, D = z_t.shape
        C = ctx_states.size(1)

        # ── Per-position time embedding ─────────────────────────────────
        # Handle both (B,) shared time and (B, H) per-position time.
        if t.dim() == 1:
            # Euler inference: same t for all positions → broadcast
            t_expanded = t.unsqueeze(1).expand(-1, H)  # (B, H)
        else:
            t_expanded = t  # already (B, H)

        # Embed each position's time independently
        t_flat = t_expanded.reshape(-1)                      # (B*H,)
        t_emb = self.time_embed(t_flat).reshape(B, H, -1)    # (B, H, hidden_dim)

        # ── Build per-position future input ─────────────────────────────
        # Fuse [z_t ‖ action_emb ‖ t_emb] → hidden_dim
        future_input = torch.cat([z_t, future_action_emb, t_emb], dim=-1)
        future_tokens = self.input_proj(future_input)  # (B, H, hidden_dim)

        # Add horizon positional embeddings
        future_tokens = future_tokens + self.horizon_pos_embed[:, :H, :]

        # ── Concat context + future → transformer ──────────────────────
        x = torch.cat([ctx_states, future_tokens], dim=1)  # (B, C + H, D)

        # ── Additive conditioning ──────────────────────────────────────
        cond_input = torch.cat([future_action_emb, t_emb], dim=-1)
        future_cond = self.cond_proj(cond_input)  # (B, H, action_dim)

        ctx_cond = self.ctx_cond_token.expand(B, C, -1)  # (B, C, action_dim)
        cond = torch.cat([ctx_cond, future_cond], dim=1)  # (B, C + H, action_dim)

        # ── Transformer backbone ────────────────────────────────────────
        out = self.velocity_net(x, cond, kv_cache=None)  # (B, C + H, D)

        # ── Velocity head (future positions only) ──────────────────────
        future_out = out[:, C:]  # (B, H, D)
        velocity = self.velocity_head(future_out)  # (B, H, D)

        return velocity

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, ctx_states, future_action_emb, target_states, source_states):
        """Training mode: compute velocity matching loss.

        State-to-state flow matching: interpolates between source_states
        (last context state + noise) and target_states (ground-truth
        next states).

        Args:
            ctx_states:        (B, C, D)          real encoded context frames
            future_action_emb: (B, H, action_dim) action embeddings for future steps
            target_states:     (B, H, D)          ground-truth next states
            source_states:     (B, H, D)          source states (last ctx broadcast)

        Returns:
            velocity_loss: scalar  MSE velocity matching loss
        """
        B, H, D = target_states.shape

        # Warm-start: perturb the broadcast source with noise to provide
        # per-position diversity while keeping the source informative.
        noise = torch.randn_like(source_states)
        z_0 = source_states + self.source_noise_sigma * noise  # (B, H, D)

        # Per-position uniform time sampling — each position gets its own
        # t ∈ [0, 1], exposing the model to diverse interpolation scenarios.
        # This breaks the symmetry where all positions see identical z_t at
        # the same t, forcing the model to use action conditioning to
        # differentiate positions rather than relying on z_t alone.
        t = torch.rand(B, H, device=target_states.device)  # (B, H)

        # OT interpolation: z_t = (1-t)*z_0 + t*z_1, per position
        t_expanded = t.unsqueeze(-1)                          # (B, H, 1)
        z_t = (1.0 - t_expanded) * z_0 + t_expanded * target_states

        # Target velocity adapts to the noisy source: v* = z_1 - z_0
        velocity_target = target_states - z_0

        # Predicted velocity (per-position t passed through)
        velocity_pred = self.compute_velocity(z_t, t, ctx_states, future_action_emb)

        # Velocity matching loss
        velocity_loss = torch.nn.functional.mse_loss(velocity_pred, velocity_target)

        return velocity_loss

    # ------------------------------------------------------------------
    # Inference / Imagination
    # ------------------------------------------------------------------

    def generate(self, ctx_states, future_action_emb, num_steps=None, source=None):
        """Euler integration from t=0 (source state) to t=1 (predicted states).

        Fully differentiable — gradients flow through all Euler steps.
        If no source is provided, the last context state is broadcast
        across the prediction horizon.

        Args:
            ctx_states:        (B, C, D)          real context frames
            future_action_emb: (B, H, action_dim) action embeddings for future steps
            num_steps:         int, optional       override default Euler steps
            source:            (B, H, D), optional explicit source tensor

        Returns:
            pred_states: (B, H, D)  generated future latent states
        """
        return self._euler_solve(ctx_states, future_action_emb, num_steps=num_steps, source=source)

    # ------------------------------------------------------------------
    # Internal Euler solver
    # ------------------------------------------------------------------

    def _euler_solve(self, ctx_states, future_action_emb, num_steps=None, source=None):
        """Run K-step Euler integration of the learned velocity field.

        z_{k+1} = z_k + dt * v_θ(z_k, t_k, ctx, actions)

        Starts from the source state (default: last context state broadcast
        across the prediction horizon) with added noise to match training.

        Args:
            ctx_states:        (B, C, D)
            future_action_emb: (B, H, action_dim)
            num_steps:         int
            source:            (B, H, D), optional explicit source tensor

        Returns:
            z: (B, H, D) — predicted states at t=1
        """
        K = num_steps or self.num_euler_steps
        dt = 1.0 / K

        H = future_action_emb.size(1)

        if source is not None:
            z = source
        else:
            # Default: broadcast last context state across the horizon
            z = ctx_states[:, -1:].expand(-1, H, -1).clone()

        # Warm-start noise — matches training distribution
        z = z + self.source_noise_sigma * torch.randn_like(z)

        for k in range(K):
            t_k = torch.full(
                (z.size(0),), k * dt,
                device=z.device, dtype=z.dtype,
            )
            v = self.compute_velocity(z, t_k, ctx_states, future_action_emb)
            z = z + dt * v

        return z
