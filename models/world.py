import torch
import torch.nn as nn

class World(nn.Module):
    def __init__(self, vision, dynamics, action, reward, termination, horizon=15, decoder=None):
        super().__init__()
        self.vision = vision
        self.dynamics = dynamics
        self.action = action
        self.reward = reward
        self.termination = termination
        self.decoder = decoder
        self.latest_state = None
        self.horizon = horizon

    def forward(self, actions, observations, ctx_frames=None, run_euler=False):
        '''
        Training forward pass with flow matching dynamics.

        action: (B, T-1, Action space)
        observations: (B, T, C, H, W)
        ctx_frames: int — number of real frames used as context
        run_euler: bool — whether to compute Euler predicted next states (slow, for debugging/logging)

        Returns:
            states:        (B, T, D)     all encoded states
            target_state:  (B, T-1, D)   ground-truth next states (encoder outputs)
            euler_full:    (B, T-1, D)   Euler-predicted states (or None if run_euler=False)
            rewards:       (B, T-1, 1)   predicted rewards (from target states)
            terminals:     (B, T-1, 1)   predicted terminals (from target states)
            velocity_loss: scalar        CFM velocity matching loss
        '''

        B, T_act, AS = actions.shape
        action_embedding = self.action(actions.reshape(B*T_act, AS)).reshape(B, T_act, -1)

        B, T_obs, C, H, W = observations.shape
        states = self.vision(observations.reshape(B*T_obs, C, H, W))

        states = states.reshape(B, T_obs, -1)
        current_states = states[:, :-1]   # (B, T-1, D) = (B, T_act, D)
        target_state = states[:, 1:]      # (B, T-1, D) = (B, T_act, D)

        # --- Flow matching dynamics ---
        if ctx_frames is not None and ctx_frames < T_act:
            ctx_states = current_states[:, :ctx_frames]            # (B, ctx, D)
            future_targets = target_state[:, ctx_frames - 1:]      # (B, H, D)
            future_action_emb = action_embedding[:, ctx_frames - 1:]  # (B, H, act_dim)
        else:
            ctx_frames = 1
            ctx_states = current_states[:, :1]
            future_targets = target_state
            future_action_emb = action_embedding

        # Source for state-to-state flow matching: last context state,
        # broadcast across the future prediction horizon
        source_states = ctx_states[:, -1:].expand_as(future_targets)

        # Velocity matching loss (trains the velocity network).
        # No detach — flow matching gradients flow into the vision encoder,
        # encouraging dynamics-aware representations.
        velocity_loss = self.dynamics(
            ctx_states, future_action_emb, future_targets, source_states
        )

        # Reward/term heads train on TARGET states (encoder outputs) — always
        # meaningful representations, unlike Euler-predicted states which are
        # garbage early in training.
        rewards = self.reward(target_state)
        terminals = self.termination(target_state)

        # Euler solve for reconstruction visualization and loss (no grad needed)
        euler_full = None
        if run_euler:
            with torch.no_grad():
                euler_pred = self.dynamics.generate(
                    ctx_states, future_action_emb
                )  # (B, H, D)
                # Pad to full T_act length for visualization / loss
                if ctx_frames > 1:
                    euler_full = torch.cat([current_states[:, 1:ctx_frames], euler_pred], dim=1)
                else:
                    euler_full = euler_pred
                euler_full = euler_full[:, :T_act]

        return states, target_state, euler_full, rewards, terminals, velocity_loss
    
    def reset_cache(self):
        """Kept for API compatibility. Flow matching is non-autoregressive."""
        self.latest_state = None

    def step_world(self, actions, start_observations=None, start_states=None):
        '''
        Single-step world prediction for environment rollouts.
        Uses flow matching generate() with a 1-step prediction horizon.

        actions: (B, Seq, Action space)
        start_observations: (B, T, C, H, W) - For environment rollouts
        start_states: (B, Seq, Latent) - For imagination rollouts
        '''
        action_embedding = self.action(actions)

        if self.latest_state is not None:
            # Use latest_state as context (B, 1, D), predict one step forward
            ctx = self.latest_state  # (B, 1, D)
            future_act = action_embedding[:, -1:]  # (B, 1, act_dim)

            pred = self.dynamics.generate(
                ctx_states=ctx,
                future_action_emb=future_act,
                num_steps=self.dynamics.num_euler_steps,
            )  # (B, 1, D)
            self.latest_state = pred
        else:
            # First call: encode observations/states and store context
            if start_observations is not None:
                B, T, C, H, W = start_observations.shape
                state = self.vision(start_observations.reshape(B*T, C, H, W)).reshape(B, T, -1)
            elif start_states is not None:
                state = start_states
            else:
                raise ValueError("start_observations or start_states required for initial step")
            self.latest_state = state[:, -1:]
            
        reward = self.reward(self.latest_state.squeeze(1))
        terminal = self.termination(self.latest_state.squeeze(1))

        return self.latest_state, reward, terminal

    def generate_chunk(self, actions, start_states):
        '''
        Parallel chunk generation via flow matching Euler integration.
        Fully differentiable — gradients flow through all Euler steps.

        actions: (B, T_act, Action space) - where T_act = ctx - 1 + chunk_size
        start_states: (B, ctx, Latent)
        
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
        # The dynamics model only needs the future action embeddings
        future_action_emb = action_embedding[:, ctx - 1:]  # (B, chunk_size, act_dim)
        
        # Flow matching: generate chunk_size future states from context
        imagined_states = self.dynamics.generate(
            ctx_states=start_states,
            future_action_emb=future_action_emb,
            num_steps=self.dynamics.num_euler_steps,
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
    from .dynamics import FlowMatchingDynamics
    from .agent import Actor, Value
    from gymnasium import spaces
    import numpy as np

    latent_dim = 256
    hidden_dim = 256
    horizon = 16
    batch_size = 4
    img_size = (64, 64)
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    vision = VisionEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    action_encoder = ActionEncoder(action_space=action_space, hidden_dim=hidden_dim, output_dim=latent_dim)
    dynamics = FlowMatchingDynamics(
        max_frames=horizon + 10,
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
        dynamics=dynamics,
        action=action_encoder,
        reward=reward,
        termination=termination
    )

    world_params = (
        list(vision.parameters()) + 
        list(action_encoder.parameters()) + 
        list(dynamics.parameters()) + 
        list(reward.parameters()) +
        list(termination.parameters())
    )
    world_optimizer = optim.Adam(world_params, lr=1e-4)

    print("--- Training World Model (Batched) ---")
    num_train_steps = 10
    
    for step in range(num_train_steps):
        batch_obs = torch.randn(batch_size, horizon, 3, img_size[0], img_size[1])
        batch_actions = torch.randn(batch_size, horizon - 1, action_space.shape[0])
        batch_rewards_gt = torch.randn(batch_size, horizon - 1, 1)
        batch_terminals_gt = torch.zeros(batch_size, horizon - 1, 1)

        world_optimizer.zero_grad()
        
        states, target_latents, pred_latents, pred_rewards, pred_terminals, vel_loss = world(
            batch_actions, batch_obs, ctx_frames=3
        )
        
        reward_loss = F.mse_loss(pred_rewards, batch_rewards_gt)
        term_loss = F.binary_cross_entropy_with_logits(pred_terminals, batch_terminals_gt)
        
        world_loss = vel_loss + reward_loss + term_loss
        
        world_loss.backward()
        world_optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step} | Total Loss: {world_loss.item():.4f} "
                  f"(Velocity: {vel_loss.item():.4f}, Reward: {reward_loss.item():.4f}, Term: {term_loss.item():.4f})")


    print("\n--- Imagination Rollout (Flow Matching Chunk Generation) ---")
    imagination_horizon = 15
    ctx_frames = 3
    
    world.freeze()
    
    start_obs = torch.randn(batch_size, ctx_frames, 3, img_size[0], img_size[1])
    with torch.no_grad():
        ctx_states = vision(start_obs.reshape(-1, 3, img_size[0], img_size[1])).reshape(batch_size, ctx_frames, -1)
    
    dummy_actions = torch.randn(batch_size, ctx_frames - 1 + imagination_horizon, action_space.shape[0])
    
    all_states, rewards, terminals = world.generate_chunk(dummy_actions, ctx_states)
    
    print(f"Successfully generated a trajectory of {imagination_horizon} steps via flow matching.")
    print(f"States: {all_states.shape}, Rewards: {rewards.shape}, Terminals: {terminals.shape}")