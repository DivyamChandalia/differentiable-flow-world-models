import torch
import torch.nn as nn
from .common import ConditionalTransformer, SinusoidalTimeEmbedding


class FlowMatchingDynamics(nn.Module):
    """State-to-State Conditional Flow Matching dynamics model.

    Learns a velocity field v_θ(z_t, t | ctx, a) that transports the last
    known context state to each future target state in parallel.  The ODE
    interpolates from t=0 (source = last context state, broadcast) to t=1
    (predicted/target states) using Euler steps.  The entire solve is
    differentiable, enabling analytical gradients for actor training.

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
        source_noise_sigma=0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_euler_steps = num_euler_steps
        self.source_noise_sigma = source_noise_sigma
        self.horizon = max_frames

        # Learnable position embedding for future tokens — breaks symmetry
        # when all positions start from the same broadcast source state.
        self.horizon_pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.normal_(self.horizon_pos_embed, std=0.02)

        # Project action embeddings directly into the state space for
        # content-level injection into z_t.  AdaLN alone is too weak to
        # differentiate identical hidden states; this gives each position
        # unique content based on its driving action.
        self.action_to_state = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.action_to_state[-1].weight)
        nn.init.zeros_(self.action_to_state[-1].bias)

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
            z_t:               (B, H, D)        interpolated future states
            t:                 (B,) or (B, 1)    flow timestep in [0, 1]
            ctx_states:        (B, C, D)         real context frames (clean)
            future_action_emb: (B, H, action_dim) action embeddings for 
                               predicted future steps

        Returns:
            velocity:          (B, H, D)  predicted velocity at positions of z_t
        """
        B, H, D = z_t.shape
        C = ctx_states.size(1)

        # Inject horizon position embedding + action content so the
        # transformer can distinguish future positions even when the
        # underlying state content is identical (t ≈ 0 / early Euler).
        z_t = z_t + self.horizon_pos_embed[:, :H, :] + self.action_to_state(future_action_emb)

        # Concat context (clean) + future tokens → transformer input
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

    def forward(self, ctx_states, future_action_emb, target_states, source_states):
        """Training mode: compute velocity matching loss.

        State-to-state flow matching: interpolates between source_states
        (last context state, broadcast) and target_states (ground-truth
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

        # Logit-normal time sampling (SD3 / Rectified Flow):
        # Concentrates samples around the mean with tails at t≈0 and t≈1.
        u = torch.randn(B, device=target_states.device) * 1.0 + (-0.5)
        t = torch.sigmoid(u)                              # (B,)

        # OT interpolation: z_t = (1-t)*z_0 + t*z_1
        t_expanded = t[:, None, None]                     # (B, 1, 1)
        z_t = (1.0 - t_expanded) * z_0 + t_expanded * target_states

        # Target velocity adapts to the noisy source: v* = z_1 - z_0
        velocity_target = target_states - z_0

        # Predicted velocity
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
        across the prediction horizon).

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
