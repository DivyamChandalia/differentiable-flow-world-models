import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(1))
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, x):
        x_flat = x.detach().reshape(-1, x.size(-1))
        batch_mean = x_flat.mean(dim=0)
        batch_var = x_flat.var(dim=0, unbiased=False)
        batch_count = x_flat.size(0)

        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.count.fill_(batch_count)
        else:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta.square() * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(tot_count)

    def normalize(self, x):
        std = torch.sqrt(self.var + self.epsilon)
        return (x - self.mean) / std

    def denormalize(self, x):
        std = torch.sqrt(self.var + self.epsilon)
        return x * std + self.mean


class World(nn.Module):
    def __init__(self, vision, dynamics_ar, dynamics_flow, action, reward, termination, horizon=15, decoder=None):
        """Dual-dynamics world model.

        Args:
            vision:        VisionEncoder — image → latent
            dynamics_ar:   Dynamics (autoregressive) — trains encoder, reward, termination
            dynamics_flow: FlowMatchingDynamics — fast parallel imagination for actor
            action:        ActionEncoder — raw action → latent
            reward:        Reward head
            termination:   Termination head
            horizon:       max sequence length for KV cache sliding window
            decoder:       optional VisionDecoder for reconstruction debugging
        """
        super().__init__()
        self.vision = vision
        self.dynamics_ar = dynamics_ar
        self.dynamics_flow = dynamics_flow
        self.action = action
        self.reward = reward
        self.termination = termination
        self.decoder = decoder
        self.horizon = horizon

        # Latent standardization tracker for flow matching
        self.latent_rms = RunningMeanStd(dim=vision.latent_dim if hasattr(vision, 'latent_dim') else 256)

        # KV cache for autoregressive step_world
        self.current_kv_cache = None
        self.latest_state = None

    def forward(self, actions, observations, ctx_frames=None, run_euler=False,
                flow_detach_encoder=True, flow_distill_from_ar=False,
                flow_standardize_latents=True, flow_cfg_dropout=0.0,
                flow_cfg_scale=1.0):
        '''
        Training forward pass with dual dynamics.

        Autoregressive dynamics (dynamics_ar):
            - Predicts next states from context + placeholders (parallel)
            - Trains encoder, reward, termination via gradient flow
            - Returns pred_next_state for MSE dynamics loss

        Flow matching dynamics (dynamics_flow):
            - Learns velocity field for fast parallel imagination
            - Optionally detached from encoder to prevent collapse
            - Returns velocity_loss

        Args:
            actions:             (B, T-1, Action space)
            observations:        (B, T, C, H, W)
            ctx_frames:          int — number of real frames used as context
            run_euler:           bool — compute Euler predicted states for visualization
            flow_detach_encoder: bool — if True, stop-grad encoder outputs for flow model
            flow_distill_from_ar: bool — if True, flow model trains on AR predictions
                                         instead of encoder outputs
            flow_standardize_latents: bool — if True, standardize latents for flow matching

        Returns:
            states:            (B, T, D)     all encoded states
            target_state:      (B, T-1, D)   ground-truth next states (encoder outputs)
            pred_next_state:   (B, T-1, D)   autoregressive predicted states
            pred_rewards:      (B, T-1, 1)   predicted rewards (from AR predictions)
            pred_terminals:    (B, T-1, 1)   predicted terminals (from AR predictions)
            velocity_loss:     scalar         CFM velocity matching loss
            euler_full:        (B, T-1, D)   Euler-predicted states (or None)
        '''

        B, T_act, AS = actions.shape
        action_embedding = self.action(actions.reshape(B*T_act, AS)).reshape(B, T_act, -1)

        B, T_obs, C, H, W = observations.shape
        states = self.vision(observations.reshape(B*T_obs, C, H, W))

        states = states.reshape(B, T_obs, -1)
        current_states = states[:, :-1]   # (B, T-1, D) = (B, T_act, D)
        target_state = states[:, 1:]      # (B, T-1, D) = (B, T_act, D)

        # Update latent running statistics for flow matching
        if self.training:
            self.latent_rms.update(states)

        # =====================================================================
        # Autoregressive dynamics — teacher-forced training
        # With causal attention, position i sees states 0..i and predicts
        # state i+1 via the residual: pred[i] = current_states[i] + transformer_out[i]
        # No placeholders needed — the causal mask prevents future leakage.
        # =====================================================================
        pred_next_state = self.dynamics_ar(current_states, action_embedding)

        # Reward/term heads train on autoregressive predictions — gradients
        # flow through the dynamics model as per the user's request
        pred_rewards = self.reward(pred_next_state)
        pred_terminals = self.termination(pred_next_state)

        # =====================================================================
        # Flow matching dynamics — velocity loss for the imagination model
        # =====================================================================
        flow_ctx_frames = ctx_frames if (ctx_frames is not None and ctx_frames < T_act) else 1

        if flow_distill_from_ar:
            # Train flow model to match AR predictions (distillation)
            flow_future_targets = pred_next_state[:, flow_ctx_frames - 1:].detach()
        else:
            # Train flow model on encoder outputs
            flow_future_targets = target_state[:, flow_ctx_frames - 1:]

        flow_ctx_states = current_states[:, :flow_ctx_frames]
        flow_future_act = action_embedding[:, flow_ctx_frames - 1:]

        if flow_detach_encoder:
            # Stop-grad: flow model doesn't influence encoder learning
            flow_ctx_states = flow_ctx_states.detach()
            flow_future_targets = flow_future_targets.detach()
            flow_future_act = flow_future_act.detach()

        flow_source = flow_ctx_states[:, -1:].expand_as(flow_future_targets)

        if flow_standardize_latents:
            flow_ctx_states_norm = self.latent_rms.normalize(flow_ctx_states)
            flow_future_targets_norm = self.latent_rms.normalize(flow_future_targets)
            flow_source_norm = self.latent_rms.normalize(flow_source)
            velocity_loss = self.dynamics_flow(
                flow_ctx_states_norm, flow_future_act, flow_future_targets_norm, flow_source_norm,
                cfg_dropout=flow_cfg_dropout
            )
        else:
            velocity_loss = self.dynamics_flow(
                flow_ctx_states, flow_future_act, flow_future_targets, flow_source,
                cfg_dropout=flow_cfg_dropout
            )

        # =====================================================================
        # Euler solve for reconstruction visualization (optional, no grad)
        # =====================================================================
        euler_full = None
        if run_euler:
            with torch.no_grad():
                euler_ctx = current_states[:, :flow_ctx_frames].detach()
                euler_act = action_embedding[:, flow_ctx_frames - 1:].detach()
                if flow_standardize_latents:
                    euler_ctx_norm = self.latent_rms.normalize(euler_ctx)
                    euler_pred_norm = self.dynamics_flow.generate(
                        euler_ctx_norm, euler_act, cfg_scale=flow_cfg_scale
                    )  # (B, H, D)
                    euler_pred = self.latent_rms.denormalize(euler_pred_norm)
                else:
                    euler_pred = self.dynamics_flow.generate(
                        euler_ctx, euler_act, cfg_scale=flow_cfg_scale
                    )  # (B, H, D)
                # Pad to full T_act length for visualization / loss
                if flow_ctx_frames > 1:
                    euler_full = torch.cat([current_states[:, 1:flow_ctx_frames], euler_pred], dim=1)
                else:
                    euler_full = euler_pred
                euler_full = euler_full[:, :T_act]

        return states, target_state, pred_next_state, pred_rewards, pred_terminals, velocity_loss, euler_full

    def reset_cache(self):
        """Clears the KV cache before a new sequence rollout."""
        self.current_kv_cache = None
        self.latest_state = None

    def step_world(self, actions, start_observations=None, start_states=None):
        '''
        Single-step world prediction for environment rollouts.
        Uses autoregressive dynamics with KV cache for sequential inference.

        actions: (B, Seq, Action space)
        start_observations: (B, T, C, H, W) - For environment rollouts
        start_states: (B, Seq, Latent) - For imagination rollouts
        '''
        action_embedding = self.action(actions)

        if self.current_kv_cache is None:
            if start_observations is not None:
                B, T, C, H, W = start_observations.shape
                state = self.vision(start_observations.reshape(B*T, C, H, W)).reshape(B, T, -1)
            elif start_states is not None:
                state = start_states
            else:
                raise ValueError("start_observations or start_states required for initial rollout")
            
            if state.size(1) > self.horizon:
                state = state[:, -self.horizon:]
                action_embedding = action_embedding[:, -self.horizon:]

            out_states, self.current_kv_cache = self.dynamics_ar(state, action_embedding, kv_cache=[])
            self.latest_state = out_states[:, -1:]

        else:
            curr_action = action_embedding[:, -1:]

            if self.current_kv_cache is not None:
                # Sliding window: evict the oldest token if we are at the horizon limit.
                cache_len = self.current_kv_cache[0][0].size(2)
                if cache_len >= self.horizon:
                    self.current_kv_cache = [
                        (k[:, :, 1:, :], v[:, :, 1:, :])
                        for k, v in self.current_kv_cache
                    ]

            out_states, self.current_kv_cache = self.dynamics_ar(
                self.latest_state,
                curr_action,
                kv_cache=self.current_kv_cache
            )
            self.latest_state = out_states[:, -1:]

        reward = self.reward(self.latest_state.squeeze(1))
        terminal = self.termination(self.latest_state.squeeze(1))

        return self.latest_state, reward, terminal

    def generate_chunk(self, actions, start_states, flow_standardize_latents=True):
        '''
        Parallel chunk generation via flow matching Euler integration.
        Fully differentiable — gradients flow through all Euler steps.
        Used for actor imagination / training.

        actions: (B, T_act, Action space) - where T_act = ctx - 1 + chunk_size
        start_states: (B, ctx, Latent)
        flow_standardize_latents: bool — whether to standardize inputs to the flow model
        
        Returns:
            all_states_seq: (chunk_size + 1, B, D)  — includes start state
            rewards:        (chunk_size, B, 1)
            terminals:      (chunk_size, B, 1)
        '''
        B, T_act, AS = actions.shape
        action_embedding = self.action(actions.reshape(B * T_act, AS)).reshape(B, T_act, -1)
        
        ctx = start_states.size(1)
        chunk_size = T_act - ctx + 1
        
        # actions layout: [ctx-1 context transition actions | chunk_size future actions]
        future_action_emb = action_embedding[:, ctx - 1:]  # (B, chunk_size, act_dim)
        
        # Flow matching: generate chunk_size future states from context
        if flow_standardize_latents:
            ctx_states_norm = self.latent_rms.normalize(start_states)
            imagined_states_norm = self.dynamics_flow.generate(
                ctx_states=ctx_states_norm,
                future_action_emb=future_action_emb,
                num_steps=self.dynamics_flow.num_euler_steps,
            )  # (B, chunk_size, D)
            imagined_states = self.latent_rms.denormalize(imagined_states_norm)
        else:
            imagined_states = self.dynamics_flow.generate(
                ctx_states=start_states,
                future_action_emb=future_action_emb,
                num_steps=self.dynamics_flow.num_euler_steps,
            )  # (B, chunk_size, D)
        
        # Prepend the starting state to get chunk_size + 1 states
        all_states = torch.cat([start_states[:, -1:], imagined_states], dim=1)  # (B, chunk_size + 1, D)
        
        # Transpose to (Seq, B, D) format
        all_states_seq = all_states.transpose(0, 1)       # (chunk_size + 1, B, D)
        imagined_states_seq = imagined_states.transpose(0, 1)  # (chunk_size, B, D)
        
        # Predict rewards and terminals
        rewards = self.reward(imagined_states_seq)       # (chunk_size, B, 1)
        terminals = self.termination(imagined_states_seq) # (chunk_size, B, 1)
        
        return all_states_seq, rewards, terminals

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def reset(self):
        """Helper to cleanly reset the environment state."""
        self.reset_cache()

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from .world_helpers import VisionEncoder, ActionEncoder, Reward, Termination
    from .dynamics import Dynamics, FlowMatchingDynamics
    from .agent import Actor, Value
    from gymnasium import spaces
    import numpy as np

    latent_dim = 256
    hidden_dim = 256
    horizon = 16
    batch_size = 4
    img_size = (64, 64)
    ctx_frames = 3
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    vision = VisionEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    action_encoder = ActionEncoder(action_space=action_space, hidden_dim=hidden_dim, output_dim=latent_dim)
    
    dynamics_ar = Dynamics(
        max_frames=horizon,
        action_dim=latent_dim,
        hidden_dim=latent_dim,
        num_layers=6,
        num_heads=4,
        causal=False,  # Non-causal for parallel training
    )
    dynamics_flow = FlowMatchingDynamics(
        max_frames=horizon + ctx_frames,
        action_dim=latent_dim,
        hidden_dim=latent_dim,
        num_layers=6,
        num_heads=4,
        num_euler_steps=6,
    )
    
    reward = Reward(obs_dim=latent_dim, hidden_dim=hidden_dim)
    termination = Termination(obs_dim=latent_dim, hidden_dim=hidden_dim)
    value = Value(obs_dim=latent_dim, hidden_dim=hidden_dim)
    actor = Actor(action_space=action_space, obs_dim=latent_dim, hidden_dim=hidden_dim)

    world = World(
        vision=vision,
        dynamics_ar=dynamics_ar,
        dynamics_flow=dynamics_flow,
        action=action_encoder,
        reward=reward,
        termination=termination,
    )

    world_params = (
        list(vision.parameters()) + 
        list(action_encoder.parameters()) + 
        list(dynamics_ar.parameters()) +
        list(dynamics_flow.parameters()) +
        list(reward.parameters()) +
        list(termination.parameters())
    )
    world_optimizer = optim.Adam(world_params, lr=1e-4)

    print("--- Training World Model (Dual Dynamics) ---")
    num_train_steps = 10
    
    for step in range(num_train_steps):
        batch_obs = torch.randn(batch_size, horizon, 3, img_size[0], img_size[1])
        batch_actions = torch.randn(batch_size, horizon - 1, action_space.shape[0])
        batch_rewards_gt = torch.randn(batch_size, horizon - 1, 1)
        batch_terminals_gt = torch.zeros(batch_size, horizon - 1, 1)

        world_optimizer.zero_grad()
        
        states, target_latents, pred_latents, pred_rewards, pred_terminals, vel_loss, _ = world(
            batch_actions, batch_obs, ctx_frames=ctx_frames
        )
        
        dyn_loss = F.mse_loss(pred_latents, target_latents.detach())
        reward_loss = F.mse_loss(pred_rewards, batch_rewards_gt)
        term_loss = F.binary_cross_entropy_with_logits(pred_terminals, batch_terminals_gt)
        
        world_loss = dyn_loss + vel_loss + reward_loss + term_loss
        
        world_loss.backward()
        world_optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step} | Total Loss: {world_loss.item():.4f} "
                  f"(AR Dyn: {dyn_loss.item():.4f}, Flow Vel: {vel_loss.item():.4f}, "
                  f"Reward: {reward_loss.item():.4f}, Term: {term_loss.item():.4f})")


    print("\n--- Imagination Rollout (Flow Matching Chunk Generation) ---")
    imagination_horizon = 15
    
    world.freeze()
    
    start_obs = torch.randn(batch_size, ctx_frames, 3, img_size[0], img_size[1])
    with torch.no_grad():
        ctx_states = vision(start_obs.reshape(-1, 3, img_size[0], img_size[1])).reshape(batch_size, ctx_frames, -1)
    
    dummy_actions = torch.randn(batch_size, ctx_frames - 1 + imagination_horizon, action_space.shape[0])
    
    all_states, rewards, terminals = world.generate_chunk(dummy_actions, ctx_states)
    
    print(f"Successfully generated a trajectory of {imagination_horizon} steps via flow matching.")
    print(f"States: {all_states.shape}, Rewards: {rewards.shape}, Terminals: {terminals.shape}")