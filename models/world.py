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

    def forward(self, actions, observations):
        '''
        action: (B, T-1, Action space)
        observations: (B, T, C, H, W)
        '''

        B, T_act, AS = actions.shape
        action_embedding = self.action(actions.reshape(B*T_act, AS)).reshape(B, T_act, -1)

        B, T_obs, C, H, W = observations.shape
        states = self.vision(observations.reshape(B*T_obs, C, H, W))

        states = states.reshape(B, T_obs, -1)
        current_states = states[:, :-1] 
        target_state = states[:, 1:]   

        next_state = self.dynamics(current_states, action_embedding)
        
        rewards = self.reward(next_state)
        terminals = self.termination(next_state)

        return states, target_state, next_state, rewards, terminals
    
    def reset_cache(self):
        """Clears the KV cache before a new sequence rollout."""
        self.current_kv_cache = None
        self.latest_state = None

    def step_world(self, actions, start_observations=None, start_states=None):
        '''
        actions: (B, Seq, Action space)
        start_observations: (B, 3, C, H, W) - For environment rollouts
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

            out_states, self.current_kv_cache = self.dynamics(state, action_embedding, kv_cache=[])
            self.latest_state = out_states[:, -1:]

        else:
            curr_action = action_embedding[:, -1:]

            # # Stop gradients of states and KV cache flowing back in time
            # self.latest_state = self.latest_state.detach()
            if self.current_kv_cache is not None:
                self.current_kv_cache = [(k.detach(), v.detach()) for k, v in self.current_kv_cache]

                # Sliding window: evict the oldest token if we are at the horizon limit.
                # This keeps start_pos + 1 <= self.horizon so RoPE freqs_cis never goes
                # out of bounds regardless of how many context frames primed the cache.
                cache_len = self.current_kv_cache[0][0].size(2)  # tokens already in cache
                if cache_len >= self.horizon:
                    self.current_kv_cache = [
                        (k[:, :, 1:, :], v[:, :, 1:, :])
                        for k, v in self.current_kv_cache
                    ]

            out_states, self.current_kv_cache = self.dynamics(
                self.latest_state,
                curr_action,
                kv_cache=self.current_kv_cache
            )
            self.latest_state = out_states[:, -1:]

        reward = self.reward(self.latest_state.squeeze(1))
        terminal = self.termination(self.latest_state.squeeze(1))

        return self.latest_state, reward, terminal

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
    from .dynamics import Dynamics
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
    dynamics = Dynamics(
        max_frames=horizon,
        action_dim=latent_dim,
        hidden_dim=latent_dim,
        num_layers=6,
        num_heads=4
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
        
        states, target_latents, pred_latents, pred_rewards, pred_terminals = world(batch_actions, batch_obs)
        
        latent_loss = F.mse_loss(pred_latents, target_latents)
        reward_loss = F.mse_loss(pred_rewards, batch_rewards_gt)
        term_loss = F.binary_cross_entropy_with_logits(pred_terminals, batch_terminals_gt)
        
        world_loss = latent_loss + reward_loss + term_loss
        
        world_loss.backward()
        world_optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step} | Total Loss: {world_loss.item():.4f} "
                  f"(Latent: {latent_loss.item():.4f}, Reward: {reward_loss.item():.4f}, Term: {term_loss.item():.4f})")


    print("\n--- Imagination Rollout (Actor/Value Training Phase) ---")
    imagination_horizon = 15
    
    world.reset()
    world.freeze()
    
    start_obs = torch.randn(batch_size, 1, 3, img_size[0], img_size[1])
    dummy_start_act = torch.zeros(batch_size, 1, action_space.shape[0])
    
    _ = world.step_world(dummy_start_act, start_observations=start_obs)
    
    imagined_rewards = []
    imagined_terminals = []
    imagined_values = []
    log_probs = []
    
    for t in range(imagination_horizon):
        current_latent = world.latest_state.squeeze(1).detach()
        
        action, log_prob, _ = actor(current_latent)
        
        action_seq = action.unsqueeze(1)
        _, predicted_reward, predicted_terminal = world.step_world(action_seq)
        
        state_value = value(world.latest_state.squeeze(1))
        
        imagined_rewards.append(predicted_reward)
        imagined_terminals.append(predicted_terminal)
        imagined_values.append(state_value)
        log_probs.append(log_prob)
        
    print(f"Successfully imagined a trajectory of {imagination_horizon} steps.")
    print(f"Collected Shapes -> Rewards: {imagined_rewards[0].shape}, Terminals: {imagined_terminals[0].shape}, Values: {imagined_values[0].shape}")