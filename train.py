import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from envs.helpers import make_env, SubprocVecEnv
from losses import SIGReg, WeakSIGReg, ValueLoss, ActorLoss, ReturnEMA
from models.agent import Actor, Value
from models.dynamics import Dynamics, FlowMatchingDynamics
from models.world_helpers import VisionEncoder, DINOv3VisionEncoder, ActionEncoder, Reward, VisionDecoder, Termination
import torchvision
from models.world import World
from models.augmentations import augment_obs, time_symmetry_aug
import os
from datetime import datetime
from models.common import symlog, symexp
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageDraw
from torch.distributions import Normal

@dataclass
class Config:
    domain: str = 'cartpole'
    task: str = 'swingup'
    seed: int = 42
    image_size: int = 64
    num_envs: int = 5
    
    latent_dim: int = 256
    hidden_dim: int = 256

    vision_encoder: str = "dinov3"  # "cnn" or "dinov3"
    dino_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_freeze_backbone: bool = True
    
    # Autoregressive dynamics (for encoder/reward/termination training)
    dyn_num_layers: int = 6
    dyn_num_heads: int = 4
    dyn_causal: bool = True  # Causal: teacher-forced AR, mask prevents seeing future states
    dyn_block_type: str = 'adaln'
    
    # Flow matching dynamics (for actor imagination)
    flow_num_layers: int = 3
    flow_num_heads: int = 4
    flow_causal: bool = False   # Causal sequence mask: prevents future target leakage during training
    flow_num_euler_steps: int = 4
    flow_source_noise_sigma: float = 0.0
    flow_loss_weight: float = 0.1
    flow_vel_cos_weight: float = 0.02
    flow_distill_from_ar: bool = False  # If True, flow trains on AR predictions instead of encoder outputs
    flow_use_cfg: bool = False           # Master switch to enable Classifier-Free Guidance (CFG)
    flow_cfg_dropout: float = 0.15      # Probability of dropping context during flow training (CFG)
    flow_cfg_scale: float = 1.0         # CFG extrapolation scale during actor imagination
    flow_standardize_latents: bool = True # Standardize latents to standard normal for flow model
    flow_adjoint_method: str = "none"   # Adjoint method ('none' or 'torchdiffeq')
    flow_solver: str = "euler"          # Solver name ('euler', 'rk4', or 'dopri5')
    flow_solver_rtol: float = 1e-5      # Relative tolerance for adaptive solvers (dopri5)
    flow_solver_atol: float = 1e-7      # Absolute tolerance for adaptive solvers (dopri5)
    flow_sim_temperature: float = 0.1   # Temperature for cosine-similarity softmax (reward, value & termination suppression)
    flow_floor_quantile: float = 0.0    # Quantile of observed values used as suppression floor (0.0 = minimum)

    # Augmentation / View Consistency
    flow_use_shift: bool = True             # Enable spatial shift augmentation (random_shift, pad=3)
    flow_use_color: bool = False             # Enable sequence-consistent brightness & contrast jitter
    flow_sensor_noise: float = 0.000        # Std-dev of independent Gaussian sensor noise
    flow_use_consistency: bool = False      # Enable view consistency loss between two augmented views
    flow_consistency_weight: float = 0.001   # Weight of the view consistency loss term
    flow_training_method: str = 'cfm'       # Dynamics objective: 'cfm', 'euler', or 'both'
    flow_use_time_sym: bool = False           # Apply time-reversal aug (50 % of steps): reverses frames & negates actions

    imagination_mode: str = 'flow'      # 'flow' or 'ar'
    world_backend: tuple = ('flow',)  # Active backends ('flow', 'ar')
    world_training: tuple = ('flow',) # Backends active for training reward/termination heads
    
    max_frames: int = 6
    world_horizon: int = 3
    imagination_ctx_frames: int = 3   # minimum real frames fed as context before imagining

    batch_size: int = 96  
    train_steps: int = 100
    num_epochs: int = 500
    
    world_agent_ratio: float = 1.0  
    
    world_lr: float = 1e-3
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    grad_clip_norm: float = 100.0
    use_amp: bool = True 
    amp_dtype: str = 'bfloat16' 
    value_target_tau: float = 0.02
    
    use_sigreg: bool = True
    sigreg_weight: float = 0.01
    weak_sigreg: bool = True     # if True, use WeakSIGReg (Frobenius cov loss) instead of SIGReg
    entropy_scale: float = 6e-3
    use_return_ema: bool = False
    use_advantage: bool = False
    reinforce: bool = False
    discount: float = 0.99
    return_lambda: float = 0.99
    actor_num_blocks: int = 4
    value_num_blocks: int = 4
    use_symlog: bool = True
    use_termination: bool = True        # Whether to use the termination head
    env_terminate_on_limit: bool = True  # Whether to terminate Cartpole episodes when hitting track limits
    freeze_world: bool = False           # Whether to freeze the world model after bootstrapping

    beta_real_value: float = 1.0
    beta_imag_value: float = 1.0
    beta_value_dynamics: float = 0.03

    # Reward model training weights
    beta_reward_real: float = 1.0
    beta_reward_dynamics: float = 0.03

    buffer_capacity: int = 50
    post_bootstrap_buffer_capacity: int = 50
    prefill_episodes: int = 50
    world_bootstrap_steps: int = 100  # extra WM-only gradient steps run after prefill, before training loop
    
    recon_debug: bool = True
    recon_train: bool = False
    
    # Diagnostics
    diag_enabled: bool = True          # master switch for actor diagnostics
    diag_every_n_steps: int = 250      # run diagnostics every N agent steps
    diag_landscape_grid: int = 21      # grid resolution for 2D landscape (11x11 = 121 evals)
    diag_landscape_range: float = 1.0  # perturbation range in param space
    diag_fd_check: bool = True         # finite-difference gradient check
    diag_fd_params: int = 20           # number of params for FD check
    
    log_dir: str = ""

    def __post_init__(self):
        if not self.flow_use_cfg:
            self.flow_cfg_dropout = 0.0
            self.flow_cfg_scale = 1.0
            
        if not self.log_dir:
            algo = "reinforce" if self.reinforce else "analytical"
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            parts = [
                f"{self.domain}_{self.task}",
                algo,
                f"ent{self.entropy_scale}",
                f"alr{self.actor_lr}",
                f"wlr{self.world_lr}",
                f"vlr{self.value_lr}",
                f"wh{self.world_horizon}",
                f"ah{self.imagination_ctx_frames}",
                f"sig{self.sigreg_weight}"
            ]
            
            if self.use_sigreg:
                if self.weak_sigreg:
                    parts.append("weaksig")
                else:
                    parts.append("sigreg")
            else:
                parts.append("nosig")
                
            if self.use_return_ema:
                parts.append("ema")
            if self.use_advantage:
                parts.append("adv")
                
            parts.append(f"seed{self.seed}")
            parts.append(current_time)
            
            run_name = "_".join(parts)
            self.log_dir = os.path.join('./runs', run_name)


class EpisodeReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
    def set_capacity(self, new_capacity):
        self.capacity = new_capacity
        self.episodes = deque(list(self.episodes)[-new_capacity:], maxlen=new_capacity)
        
    def add_episode(self, obs, actions, rewards, terminals):
        self.episodes.append({
            'obs': np.array(obs, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'terminals': np.array(terminals, dtype=np.float32)
        })

    def sample_sequences(self, batch_size, seq_len):
        batch_obs, batch_actions, batch_rewards, batch_terminals = [], [], [], []
        
        valid_episodes = [ep for ep in self.episodes if len(ep['obs']) >= seq_len]
        if not valid_episodes:
            return None
            
        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            start_idx = random.randint(0, len(ep['obs']) - seq_len)
            
            batch_obs.append(ep['obs'][start_idx : start_idx + seq_len])
            batch_actions.append(ep['actions'][start_idx : start_idx + seq_len - 1])
            batch_rewards.append(ep['rewards'][start_idx : start_idx + seq_len])
            batch_terminals.append(ep['terminals'][start_idx : start_idx + seq_len])
            
        obs_tensor = torch.as_tensor(np.stack(batch_obs), dtype=torch.float32) / 255.0
        act_tensor = torch.as_tensor(np.stack(batch_actions), dtype=torch.float32)
        rew_tensor = torch.as_tensor(np.stack(batch_rewards), dtype=torch.float32).unsqueeze(-1)
        term_tensor = torch.as_tensor(np.stack(batch_terminals), dtype=torch.float32).unsqueeze(-1)
        
        return obs_tensor, act_tensor, rew_tensor, term_tensor


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} with {cfg.num_envs} parallel environments.")
        print(f"TensorBoard log directory: {cfg.log_dir}")
        self.set_seed(cfg.seed)
        
        env_fns = [
            partial(
                make_env, 
                cfg.domain, 
                cfg.task, 
                cfg.seed + i, 
                cfg.image_size, 
                cfg.image_size,
                terminate_on_limit=cfg.env_terminate_on_limit
            ) 
            for i in range(cfg.num_envs)
        ]
        self.envs = SubprocVecEnv(env_fns)
        self.action_dim = self.envs.get_action_space().shape[0]
        
        self.val_env = make_env(
            cfg.domain, 
            cfg.task, 
            cfg.seed * 100, 
            cfg.image_size, 
            cfg.image_size,
            terminate_on_limit=cfg.env_terminate_on_limit
        )
        
        if cfg.vision_encoder == "dinov3":
            vision = DINOv3VisionEncoder(
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                model_name=cfg.dino_model_name,
                freeze_backbone=cfg.dino_freeze_backbone,
            )
        elif cfg.vision_encoder == "cnn":
            vision = VisionEncoder(
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
            )
        else:
            raise ValueError(f"Unknown vision encoder: {cfg.vision_encoder}")
        action_enc = ActionEncoder(action_space=self.envs.get_action_space(), hidden_dim=cfg.hidden_dim, output_dim=cfg.latent_dim)
        # Check configuration validity
        if cfg.imagination_mode == 'ar' and 'ar' not in cfg.world_backend:
            raise ValueError("imagination_mode cannot be 'ar' when 'ar' is not in world_backend.")
        if cfg.imagination_mode == 'flow' and 'flow' not in cfg.world_backend:
            raise ValueError("imagination_mode cannot be 'flow' when 'flow' is not in world_backend.")
        for train_backend in cfg.world_training:
            if train_backend not in cfg.world_backend:
                raise ValueError(f"world_training backend '{train_backend}' is not active in world_backend: {cfg.world_backend}")

        if 'ar' in cfg.world_backend:
            dynamics_ar = Dynamics(
                max_frames=cfg.max_frames + cfg.imagination_ctx_frames,
                action_dim=cfg.latent_dim, hidden_dim=cfg.latent_dim,
                num_layers=cfg.dyn_num_layers, num_heads=cfg.dyn_num_heads,
                causal=cfg.dyn_causal,
                block_type=cfg.dyn_block_type,
            )
        else:
            dynamics_ar = None

        if 'flow' in cfg.world_backend:
            dynamics_flow = FlowMatchingDynamics(
                max_frames=cfg.max_frames + cfg.imagination_ctx_frames,
                action_dim=cfg.latent_dim, hidden_dim=cfg.latent_dim,
                num_layers=cfg.flow_num_layers, num_heads=cfg.flow_num_heads,
                causal=cfg.flow_causal, num_euler_steps=cfg.flow_num_euler_steps,
                source_noise_sigma=cfg.flow_source_noise_sigma,
                adjoint_method=cfg.flow_adjoint_method,
                solver=cfg.flow_solver,
                rtol=cfg.flow_solver_rtol,
                atol=cfg.flow_solver_atol,
                standardize_latents=cfg.flow_standardize_latents,
                vel_cos_weight=cfg.flow_vel_cos_weight,
            )
        else:
            dynamics_flow = None

        reward_enc = Reward(obs_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        if cfg.use_termination:
            termination_enc = Termination(obs_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        else:
            termination_enc = None
        
        if cfg.recon_debug:
            decoder = VisionDecoder(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        else:
            decoder = None
            
        self.world = World(
            vision, dynamics_ar, dynamics_flow, action_enc, reward_enc, termination_enc, 
            self.cfg.world_horizon, decoder=decoder
        ).to(self.device)
        self.actor = Actor(action_space=self.envs.get_action_space(), obs_dim=cfg.imagination_ctx_frames * cfg.latent_dim, hidden_dim=cfg.hidden_dim, chunk_size=cfg.world_horizon, num_blocks=cfg.actor_num_blocks).to(self.device)
        self.value_model = Value(obs_dim=cfg.imagination_ctx_frames * cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.value_num_blocks).to(self.device)
        self.target_value_model = Value(obs_dim=cfg.imagination_ctx_frames * cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.value_num_blocks).to(self.device)
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        self.target_value_model.eval()
        for p in self.target_value_model.parameters():
            p.requires_grad = False
        
        if cfg.weak_sigreg:
            self.sig_reg = WeakSIGReg().to(self.device)
        else:
            self.sig_reg = SIGReg().to(self.device)
        self.value_loss_fn = ValueLoss(discount=cfg.discount, lambda_=cfg.return_lambda, batch_first=False, use_symlog=cfg.use_symlog)
        self.actor_loss_fn = ActorLoss(discount=cfg.discount, batch_first=False, entropy_scale=cfg.entropy_scale, reinforce=cfg.reinforce)
        self.return_ema = ReturnEMA(decay=0.99).to(self.device) if cfg.use_return_ema else None
        
        self.world_opt = optim.Adam(
            [
                param
                for param in self.world.parameters()
                if param.requires_grad
            ],
            lr=cfg.world_lr,
        )
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.value_opt = optim.Adam(self.value_model.parameters(), lr=cfg.value_lr)
        
        self.use_amp = cfg.use_amp and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        self.buffer = EpisodeReplayBuffer(capacity=cfg.buffer_capacity)
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.global_step = 0
        self.agent_step = 0
        self.world_step = 0

        # Diagnostics
        if cfg.diag_enabled:
            from diagnostics import LossLandscapeVisualizer, GradientFlowAnalyzer, ActionDistributionMonitor, FlowDiagnostics, ARDiagnostics
            self.landscape_viz = LossLandscapeVisualizer(self)
            self.grad_analyzer = GradientFlowAnalyzer(self)
            self.action_monitor = ActionDistributionMonitor(self)
            self.flow_diagnostics = FlowDiagnostics(self) if 'flow' in cfg.world_backend else None
            self.ar_diagnostics = ARDiagnostics(self) if 'ar' in cfg.world_backend else None
            self.diag_output_dir = os.path.join(cfg.log_dir, 'diagnostics')

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def draw_action_indicator(self, frame, action_array):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        num_actions = len(action_array)
        bar_width = 180
        bar_left = (256 - bar_width) // 2
        bar_right = bar_left + bar_width
        bar_center = 256 // 2
        
        bar_thickness = 4
        dot_radius = 6
        
        for d in range(num_actions):
            y_center = 230 - d * 16
            
            # Draw background bar
            draw.rounded_rectangle(
                [bar_left, y_center - bar_thickness // 2, bar_right, y_center + bar_thickness // 2],
                radius=bar_thickness // 2,
                fill=(100, 100, 100)
            )
            
            # Draw center tick
            draw.rectangle(
                [bar_center - 1, y_center - 4, bar_center + 1, y_center + 4],
                fill=(255, 255, 255)
            )
            
            # Calculate dot position
            act_val = np.clip(action_array[d], -1.0, 1.0)
            x_dot = bar_center + int(act_val * (bar_width // 2))
            
            # Draw dot (Cyan indicator with black outline)
            draw.ellipse(
                [x_dot - dot_radius - 1, y_center - dot_radius - 1, x_dot + dot_radius + 1, y_center + dot_radius + 1],
                fill=(0, 0, 0)
            )
            draw.ellipse(
                [x_dot - dot_radius, y_center - dot_radius, x_dot + dot_radius, y_center + dot_radius],
                fill=(0, 220, 255)
            )
            
            # Draw label
            label = f"a{d}"
            label_x, label_y = bar_left - 20, y_center - 6
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                draw.text((label_x + dx, label_y + dy), label, fill=(0, 0, 0))
            draw.text((label_x, label_y), label, fill=(255, 255, 255))
            
        return np.array(img)

    def _compute_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    def update_value_target(self):
        with torch.no_grad():
            for p, p_target in zip(self.value_model.parameters(), self.target_value_model.parameters()):
                p_target.copy_(self.cfg.value_target_tau * p + (1.0 - self.cfg.value_target_tau) * p_target)
    
    def _get_history_windows(self, states, horizon, context=None):
        # states shape: (Seq, B, D)
        Seq, B, D = states.shape
        if context is not None:
            ctx_states = context.transpose(0, 1) # (ctx, B, D)
            # Detect if the last context state is already prepended to states
            is_start_prepended = torch.allclose(
                states[0].float(),
                ctx_states[-1].to(device=states.device, dtype=states.dtype).float(),
                atol=1e-4
            )
            if is_start_prepended:
                ctx_feed = ctx_states[-horizon:-1] if horizon > 1 else ctx_states[:0]
            else:
                ctx_feed = ctx_states[-(horizon - 1):] if horizon > 1 else ctx_states[:0]
            padded = torch.cat([ctx_feed, states], dim=0)
        else:
            s0 = states[0:1]
            padded = torch.cat([s0.repeat(horizon - 1, 1, 1), states], dim=0)
        windows = padded.unfold(dimension=0, size=horizon, step=1) # (Seq, B, D, horizon)
        return windows.permute(0, 1, 3, 2).reshape(Seq, B, horizon * D)
    
    
    @torch.no_grad()
    def rollout(self, env, deterministic=False, max_steps=250, render=False, add_to_buffer=False):
        """Unified rollout method that supports both single and vectorized environments."""
        self.world.eval()
        self.actor.eval()
        
        is_vectorized = hasattr(env, 'n_envs') or hasattr(env, 'num_envs')
        num_envs = env.n_envs if is_vectorized else 1
        
        reset_res = env.reset()
        if is_vectorized:
            obs_batch = reset_res
        else:
            obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
            obs_batch = np.expand_dims(obs, axis=0)
            
        ep_obs = [[obs_batch[i]] for i in range(num_envs)]
        ep_actions = [[] for _ in range(num_envs)]
        ep_rewards = [[0.0] for _ in range(num_envs)]
        ep_terminals = [[0.0] for _ in range(num_envs)]
        ep_returns = np.zeros(num_envs)
        active_envs = np.ones(num_envs, dtype=bool)
        
        actor_history = deque(maxlen=self.cfg.imagination_ctx_frames)
        frames = []
        step = 0
        
        while step < max_steps and active_envs.any():
            if render and not is_vectorized:
                frame = env.render(height=256, width=256)
                
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device) / 255.0
            
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                z_t = self.world.vision(obs_tensor)
                
                # Maintain actor history and pad if necessary
                actor_history.append(z_t)
                history_list = list(actor_history)
                while len(history_list) < self.cfg.imagination_ctx_frames:
                    history_list.insert(0, history_list[0])
                
                actor_input = torch.stack(history_list, dim=1).reshape(num_envs, -1)
                
                actions, _, _ = self.actor(actor_input, deterministic=deterministic)
                actions = actions[:, 0]
                
            actions_np = actions.cpu().numpy()
            if not is_vectorized:
                # Squeeze the batch dimension for a single environment step, clip actions like in evaluate
                action_np = actions_np.squeeze(0)
                action_np = np.clip(action_np, -1.0, 1.0)
                actions_np[0] = action_np
                
            if render and not is_vectorized:
                frame = self.draw_action_indicator(frame, actions_np[0])
                frames.append(frame)
                
            if not is_vectorized:
                res = env.step(action_np)
                if len(res) == 4:
                    next_obs, reward, done, info = res
                    term = done
                else:
                    next_obs, reward, term, trunc, info = res
                    done = term or trunc
                    
                next_obs_batch = np.expand_dims(next_obs, axis=0)
                rewards = np.array([reward])
                dones = np.array([done])
                infos = [{**info, 'terminated': term}]
            else:
                next_obs_batch, rewards, dones, infos = env.step(actions_np)
                
            for i in range(num_envs):
                if not active_envs[i]:
                    continue
                
                term_val = float(infos[i].get('terminated', False))
                
                if dones[i]:
                    actual_next_obs = infos[i].get('terminal_observation', next_obs_batch[i])
                    ep_obs[i].append(actual_next_obs)
                    active_envs[i] = False
                else:
                    ep_obs[i].append(next_obs_batch[i])
                    
                ep_rewards[i].append(rewards[i])
                ep_returns[i] += rewards[i]
                ep_actions[i].append(actions_np[i])
                ep_terminals[i].append(term_val)
                
            obs_batch = next_obs_batch
            step += 1
            
        if add_to_buffer:
            for i in range(num_envs):
                self.buffer.add_episode(ep_obs[i], ep_actions[i], ep_rewards[i], ep_terminals[i])
                
        self.world.train()
        self.actor.train()
        
        if render and not is_vectorized:
            return np.mean(ep_returns), step, frames
        return np.mean(ep_returns), step

    def collect_episodes(self, max_steps=250):
        """Collects trajectories from all parallel environments simultaneously."""
        mean_return, step = self.rollout(
            env=self.envs, 
            deterministic=False, 
            max_steps=max_steps, 
            add_to_buffer=True
        )
        return mean_return, step

    def evaluate(self, epoch, max_steps=250):
        """Runs a single deterministic episode to evaluate and log a video."""
        mean_return, ep_len, frames = self.rollout(
            env=self.val_env, 
            deterministic=True, 
            max_steps=max_steps, 
            render=True,
            add_to_buffer=False
        )
        self.writer.add_scalar('Validation/Return', mean_return, epoch)
        self.writer.add_scalar('Validation/Episode_Length', ep_len, epoch)
        
        video_array = np.array(frames)
        video_tensor = torch.tensor(video_array, dtype=torch.uint8).permute(0, 3, 1, 2).unsqueeze(0)
        self.writer.add_video('Validation/Video', video_tensor, epoch, fps=30)
        
        return mean_return

    def train_world(self, batch, skip_value=False):
        obs_batch, act_batch, rew_batch, term_batch = [b.to(self.device, non_blocking=True) for b in batch]

        # Clean supervision labels — these are NEVER flipped/modified by
        # time-symmetry augmentation so they always align with clean_states.
        if self.cfg.use_symlog:
            clean_sym_rew_batch = symlog(rew_batch)
        else:
            clean_sym_rew_batch = rew_batch
        clean_term_batch = term_batch

        is_wm_frozen = self.cfg.freeze_world and (self.world_step >= self.cfg.world_bootstrap_steps)
        if is_wm_frozen:
            self.world.freeze()
        else:
            self.world.unfreeze()
        self.value_model.train()

        log_recon = self.cfg.recon_debug and (self.world_step % self.cfg.train_steps == 0)

        B, T_obs = obs_batch.shape[:2]
        T_act = act_batch.shape[1]
        AS = act_batch.shape[2]

        assert T_obs == T_act + 1, (
            f"Expected one more observation than actions, "
            f"got observations={T_obs}, actions={T_act}"
        )

        if self.cfg.recon_train:
            clean_states = self.world.vision(
                obs_batch.flatten(0, 1)
            ).reshape(B, T_obs, -1)
        else:
            with torch.no_grad():
                clean_states = self.world.vision(
                    obs_batch.flatten(0, 1)
                ).reshape(B, T_obs, -1)

        # -----------------------------------------------------------------------
        # Time-symmetry augmentation (50 % of steps)
        # Applied to the OBSERVATIONS and ACTIONS used for dynamics only.
        # Rewards / terminals are NOT modified (see clean_* labels above).
        # -----------------------------------------------------------------------
        apply_time_sym = (
            self.cfg.flow_use_time_sym
            and torch.rand((), device=self.device).item() < 0.5
        )
        if apply_time_sym:
            aug_obs_base, aug_act = time_symmetry_aug(obs_batch, act_batch)
        else:
            aug_obs_base, aug_act = obs_batch, act_batch

        # -----------------------------------------------------------------------
        # Augment observations for flow matching training
        # If view consistency is enabled we need two independent augmented views.
        # -----------------------------------------------------------------------
        need_aug = (
            self.cfg.flow_use_shift
            or self.cfg.flow_use_color
            or self.cfg.flow_sensor_noise > 0.0
        )
        if need_aug:
            augmented_obs_1 = augment_obs(
                aug_obs_base,
                use_shift=self.cfg.flow_use_shift,
                use_color=self.cfg.flow_use_color,
                noise_std=self.cfg.flow_sensor_noise,
            )
            if self.cfg.flow_use_consistency:
                augmented_obs_2 = augment_obs(
                    aug_obs_base,
                    use_shift=self.cfg.flow_use_shift,
                    use_color=self.cfg.flow_use_color,
                    noise_std=self.cfg.flow_sensor_noise,
                )
            else:
                augmented_obs_2 = None
        else:
            augmented_obs_1 = aug_obs_base
            augmented_obs_2 = None

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            (
                raw_states, target_state,
                pred_next_state, pred_rewards, pred_terminals,
                velocity_loss, euler_full, ar_full,
                pred_rewards_flow, pred_terminals_flow, t, z_pred_flow,
                cfm_loss, euler_loss,
            ) = self.world(
                aug_act, augmented_obs_1,
                ctx_frames=self.cfg.imagination_ctx_frames, run_euler=log_recon,
                flow_distill_from_ar=self.cfg.flow_distill_from_ar,
                flow_standardize_latents=self.cfg.flow_standardize_latents,
                flow_cfg_dropout=self.cfg.flow_cfg_dropout,
                flow_cfg_scale=self.cfg.flow_cfg_scale,
                run_flow=('flow' in self.cfg.world_backend),
                world_training=self.cfg.world_training,
                flow_training_method=self.cfg.flow_training_method,
            )

            # ------------------------------------------------------------------
            # View Consistency Loss — vision-encoder-only second pass
            # We only run self.world.vision on the second augmented view to get
            # states_2, then compute MSE against states from the first pass.
            # This avoids re-running the full world model a second time.
            # ------------------------------------------------------------------
            consistency_loss = torch.tensor(0.0, device=self.device)
            if self.cfg.flow_use_consistency and augmented_obs_2 is not None:
                B2, T2, C2, H2, W2 = augmented_obs_2.shape
                states_2 = self.world.vision(
                    augmented_obs_2.reshape(B2 * T2, C2, H2, W2)
                ).reshape(B2, T2, -1)
                # states from first pass (raw_states) vs second pass (states_2)
                consistency_loss = F.mse_loss(raw_states, states_2)
            
            recon_loss = None
            recon_pred_loss = None
            recon_pred_loss_ar = None
            if self.cfg.recon_debug:
                if self.cfg.recon_train:
                    recon_obs = self.world.decoder(clean_states)
                else:
                    recon_obs = self.world.decoder(clean_states.detach())

                recon_loss = F.mse_loss(recon_obs.float(), obs_batch.float())
                total_recon_loss = recon_loss

                if log_recon and euler_full is not None:
                    if self.cfg.recon_train:
                        recon_pred_obs = self.world.decoder(euler_full)
                    else:
                        recon_pred_obs = self.world.decoder(euler_full.detach())
                    recon_pred_loss = F.mse_loss(recon_pred_obs.float(), obs_batch[:, 1:].float())

                if log_recon and ar_full is not None:
                    if self.cfg.recon_train:
                        recon_pred_obs_ar = self.world.decoder(ar_full)
                    else:
                        recon_pred_obs_ar = self.world.decoder(ar_full.detach())
                    recon_pred_loss_ar = F.mse_loss(recon_pred_obs_ar.float(), obs_batch[:, 1:].float())
            
        # Autoregressive dynamics loss (MSE to detached encoder targets)
        if 'ar' in self.cfg.world_backend and pred_next_state is not None:
            ar_dyn_loss = F.mse_loss(pred_next_state.float(), target_state.detach().float())
        else:
            ar_dyn_loss = torch.tensor(0.0, device=self.device)
        
        # Flow matching velocity loss
        if 'flow' in self.cfg.world_backend:
            flow_loss = self.cfg.flow_loss_weight * velocity_loss
        else:
            flow_loss = torch.tensor(0.0, device=self.device)
            cfm_loss  = torch.tensor(0.0, device=self.device)
            euler_loss = torch.tensor(0.0, device=self.device)
        
        # View consistency loss
        if self.cfg.flow_use_consistency and consistency_loss.item() > 0.0:
            flow_loss = flow_loss + self.cfg.flow_consistency_weight * consistency_loss
        
        # Combined dynamics loss (AR + flow)
        dyn_loss = ar_dyn_loss + flow_loss
        
        loss_rew_real = None
        loss_rew_ar_tf = None
        loss_rew_flow = None
        loss_term_flow = None

        # Reward training -- clean states / clean labels for supervision.
        # Time-symmetry augmentation does NOT touch these labels.
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            pred_rew_real = self.world.reward(clean_states[:, 1:])
            reward_targets = clean_sym_rew_batch[:, 1:]
            loss_rew_real = F.mse_loss(pred_rew_real.float(), reward_targets.float())

            rew_loss = self.cfg.beta_reward_real * loss_rew_real

            if 'ar' in self.cfg.world_training and pred_next_state is not None:
                pred_rew_ar_tf = self.world.reward(pred_next_state)
                loss_rew_ar_tf = F.mse_loss(pred_rew_ar_tf.float(), reward_targets.float())
                rew_loss = rew_loss + self.cfg.beta_reward_dynamics * loss_rew_ar_tf

            # Flow-generated endpoints have valid aligned replay labels.
            # z_pred_flow[:, h] predicts the state after the context at horizon h,
            # which aligns with observations [start : start + H].
            if 'flow' in self.cfg.world_training and pred_rewards_flow is not None:
                start = self.cfg.imagination_ctx_frames
                H = pred_rewards_flow.shape[1]
                flow_reward_targets = clean_sym_rew_batch[:, start:start + H]
                assert pred_rewards_flow.shape == flow_reward_targets.shape, (
                    f"Flow reward shape mismatch: pred={pred_rewards_flow.shape}, "
                    f"target={flow_reward_targets.shape}"
                )
                loss_rew_flow = F.mse_loss(pred_rewards_flow.float(), flow_reward_targets.float())
                rew_loss = rew_loss + self.cfg.beta_reward_dynamics * loss_rew_flow

        # Termination training -- clean states / clean labels for supervision.
        if self.cfg.use_termination:
            term_loss = torch.tensor(0.0, device=self.device)
            if 'ar' in self.cfg.world_training and pred_terminals is not None:
                term_loss = term_loss + F.binary_cross_entropy_with_logits(
                    pred_terminals.float(), clean_term_batch[:, 1:].float())
            else:
                pred_term_real = self.world.termination(clean_states[:, 1:])
                term_loss = term_loss + F.binary_cross_entropy_with_logits(
                    pred_term_real.float(), clean_term_batch[:, 1:].float())

            if 'flow' in self.cfg.world_training and pred_terminals_flow is not None:
                start = self.cfg.imagination_ctx_frames
                H = pred_terminals_flow.shape[1]
                flow_terminal_targets = clean_term_batch[:, start:start + H]
                assert pred_terminals_flow.shape == flow_terminal_targets.shape, (
                    f"Flow terminal shape mismatch: pred={pred_terminals_flow.shape}, "
                    f"target={flow_terminal_targets.shape}"
                )
                loss_term_flow = F.binary_cross_entropy_with_logits(
                    pred_terminals_flow.float(), flow_terminal_targets.float())
                term_loss = term_loss + loss_term_flow
        else:
            term_loss = torch.tensor(0.0, device=self.device)
        
        if self.cfg.use_sigreg:
            sig_loss = self.sig_reg(raw_states.transpose(0, 1).float())
        else:
            sig_loss = torch.tensor(0.0, device=self.device)
            
        # --- Value Grounding Losses ---
        value_real_loss = torch.tensor(0.0, device=self.device)
        v_loss = torch.tensor(0.0, device=self.device)
        if not skip_value:
            ctx = self.cfg.imagination_ctx_frames
            B_val, T_val, D_val = clean_states.shape
            clean_states_T = clean_states.detach().transpose(0, 1)  # (T, B, D)
            history_windows_real = self._get_history_windows(clean_states_T, ctx)

            real_values = self.value_model(history_windows_real)

            with torch.no_grad():
                target_real_values = self.target_value_model(history_windows_real)
                r_real_seq_T = clean_sym_rew_batch.transpose(0, 1)  # (T, B, 1)
                real_terminals_T = clean_term_batch.transpose(0, 1)  # (T, B, 1)
                pcont = 1.0 - real_terminals_T[1:].float()

                real_lambda_targets = self.value_loss_fn._compute_lambda_returns(
                    values=target_real_values,
                    rewards=r_real_seq_T[1:],
                    pcont=pcont,
                )

            value_real_loss = F.smooth_l1_loss(real_values[:-1], real_lambda_targets.detach())
            v_loss = self.cfg.beta_real_value * value_real_loss
        
        if self.cfg.recon_debug:
            wm_loss = dyn_loss + rew_loss + term_loss + total_recon_loss
        else:
            wm_loss = dyn_loss + rew_loss + term_loss
            
        if self.cfg.use_sigreg:
            wm_loss = wm_loss + self.cfg.sigreg_weight * sig_loss
            
        # Compute individual grad norms for world model losses
        scale_factor = self.scaler.get_scale()
        
        # AR Dynamics Loss
        if not is_wm_frozen and 'ar' in self.cfg.world_backend:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(ar_dyn_loss).backward(retain_graph=True)
            ar_dyn_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            ar_dyn_grad_norm = 0.0
            
        # Flow Velocity Loss
        if not is_wm_frozen and 'flow' in self.cfg.world_backend:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(flow_loss).backward(retain_graph=True)
            flow_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            flow_grad_norm = 0.0
            
        # Reward Loss
        if not is_wm_frozen:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(rew_loss).backward(retain_graph=True)
            rew_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            rew_grad_norm = 0.0
            
        # Termination Loss
        if not is_wm_frozen and self.cfg.use_termination:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(term_loss).backward(retain_graph=True)
            term_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            term_grad_norm = 0.0
            
        # SIGReg Loss
        if not is_wm_frozen and self.cfg.use_sigreg:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(self.cfg.sigreg_weight * sig_loss).backward(retain_graph=True)
            sig_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            sig_grad_norm = 0.0
            
        # Reconstruction Loss
        if not is_wm_frozen and self.cfg.recon_debug:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(total_recon_loss).backward(retain_graph=True)
            recon_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            recon_grad_norm = 0.0
            
        # Value Grounding Loss Grad Norm
        if not skip_value:
            self.value_opt.zero_grad(set_to_none=True)
            self.scaler.scale(v_loss).backward(retain_graph=True)
            v_grad_norm = self._compute_grad_norm(self.value_model) / scale_factor
        else:
            v_grad_norm = 0.0
            
        # Combined backward
        if not is_wm_frozen:
            self.world_opt.zero_grad(set_to_none=True)
            
            total_loss = wm_loss + v_loss
            if skip_value:
                self.scaler.scale(wm_loss).backward()
            else:
                self.value_opt.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()
            
            self.scaler.unscale_(self.world_opt)
            grad_norm = self._compute_grad_norm(self.world)
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.grad_clip_norm)
            
            self.scaler.step(self.world_opt)
            if not skip_value:
                self.scaler.unscale_(self.value_opt)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.cfg.grad_clip_norm)
                self.scaler.step(self.value_opt)
            self.scaler.update()
        else:
            total_loss = wm_loss + v_loss
            if not skip_value:
                self.value_opt.zero_grad(set_to_none=True)
                self.scaler.scale(v_loss).backward()
                self.scaler.unscale_(self.value_opt)
                grad_norm = 0.0
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.cfg.grad_clip_norm)
                self.scaler.step(self.value_opt)
            else:
                grad_norm = 0.0
            self.scaler.update()

        self.writer.add_scalar('Loss/World_Total', total_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_WM_Only', wm_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_AR_Dynamics', ar_dyn_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Flow_Velocity', flow_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Flow_CFM', (self.cfg.flow_loss_weight * cfm_loss).item(), self.world_step)
        self.writer.add_scalar('Loss/World_Flow_Euler', (self.cfg.flow_loss_weight * euler_loss).item(), self.world_step)
        if self.cfg.flow_use_consistency:
            self.writer.add_scalar('Loss/World_Flow_Consistency', consistency_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Dynamics', dyn_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Reward', rew_loss.item(), self.world_step)
        if loss_rew_real is not None:
            self.writer.add_scalar('Loss/World_Reward_Real', loss_rew_real.item(), self.world_step)
        if loss_rew_ar_tf is not None:
            self.writer.add_scalar('Loss/World_Reward_AR', loss_rew_ar_tf.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Termination', term_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_SIGReg', sig_loss.item(), self.world_step)
        
        # Log value grounding losses to TensorBoard
        self.writer.add_scalar('Loss/Agent_Value_Real', value_real_loss.item(), self.world_step)
            
        self.writer.add_scalar('GradNorm/World_SIGReg', sig_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Termination', term_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_AR_Dynamics', ar_dyn_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Flow_Velocity', flow_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/Agent_Value', v_grad_norm, self.world_step)
        
        if self.cfg.recon_debug:
            self.writer.add_scalar('Loss/World_Reconstruction', recon_loss.item(), self.world_step)
            if recon_pred_loss is not None:
                self.writer.add_scalar('Loss/World_Reconstruction_Pred', recon_pred_loss.item(), self.world_step)
            if recon_pred_loss_ar is not None:
                self.writer.add_scalar('Loss/World_Reconstruction_Pred_AR', recon_pred_loss_ar.item(), self.world_step)
            self.writer.add_scalar('GradNorm/World_Reconstruction', recon_grad_norm, self.world_step)
            
        self.writer.add_scalar('GradNorm/World', grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Dynamics', ar_dyn_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Reward', rew_grad_norm, self.world_step)
        
        if self.cfg.recon_debug and (self.world_step % self.cfg.train_steps == 0):
            self.visualize_reconstruction(obs_batch, clean_states, act_batch)

        self.world_step += 1

        return clean_states.detach(), act_batch.detach(), rew_batch.detach(), clean_term_batch.detach(), wm_loss.item()

    @torch.no_grad()
    def visualize_reconstruction(self, obs_batch, raw_states, act_batch):
        idx = random.randint(0, obs_batch.size(0) - 1)
        ctx = self.cfg.imagination_ctx_frames
        
        # 1. Decode raw_states for reconstruction of encoded/clean states
        recon_obs = self.world.decoder(raw_states[idx:idx+1].detach())
        actual = obs_batch[idx, ctx:].detach().cpu()
        encoded = recon_obs[0, ctx:].detach().cpu()
        
        # 2. Autoregressive predictions (teacher-forced, causal masking)
        ctx_states = raw_states[idx:idx+1, :ctx]  # (1, ctx, D)
        actions_idx = act_batch[idx:idx+1, ctx-1:]  # (1, T-ctx, AS)
        AS = actions_idx.shape[-1]
        action_embedding = self.world.action(actions_idx.reshape(-1, AS)).reshape(1, actions_idx.size(1), -1)
        
        # Open-loop AR dynamics (fair comparison with flow matching)
        T_act = act_batch.size(1)
        
        if self.world.dynamics_ar is not None:
            all_action_emb = self.world.action(act_batch[idx:idx+1].reshape(-1, AS)).reshape(1, T_act, -1)
            ar_open_loop = []
            curr_seq = raw_states[idx:idx+1, :ctx]  # (1, ctx, D)
            
            for i in range(T_act - ctx + 1):
                act_seq = all_action_emb[:, :ctx + i]
                pred = self.world.dynamics_ar(curr_seq, act_seq)
                next_state = pred[:, -1:]  # (1, 1, D)
                ar_open_loop.append(next_state)
                curr_seq = torch.cat([curr_seq, next_state], dim=1)
                
            ar_aligned_states = torch.cat(ar_open_loop, dim=1) # (1, H, D)
            recon_ar_pred_obs = self.world.decoder(ar_aligned_states.detach())
            ar_aligned = recon_ar_pred_obs[0].detach().cpu()
        else:
            ar_aligned_states = None
            ar_aligned = None
        
        # 3. Flow matching Euler predictions
        if self.cfg.imagination_mode == 'flow' and self.world.dynamics_flow is not None:
            if self.cfg.flow_standardize_latents:
                ctx_states_norm = self.world.latent_rms.normalize(ctx_states)
                euler_pred_norm = self.world.dynamics_flow.generate(
                    ctx_states_norm, action_embedding, cfg_scale=self.cfg.flow_cfg_scale
                )
                euler_pred = self.world.latent_rms.denormalize(euler_pred_norm)
            else:
                euler_pred = self.world.dynamics_flow.generate(
                    ctx_states, action_embedding, cfg_scale=self.cfg.flow_cfg_scale
                )
            
            # Align/pad to match predicted next states format
            if ctx > 1:
                flow_pred_next_state = torch.cat([raw_states[idx:idx+1, 1:ctx], euler_pred], dim=1)
            else:
                flow_pred_next_state = euler_pred
                
            recon_flow_pred_obs = self.world.decoder(flow_pred_next_state.detach())
            flow_aligned = recon_flow_pred_obs[0, ctx-1:].detach().cpu()

            # Flow matching value predictions
            flow_windows = self._get_history_windows(euler_pred.transpose(0, 1), self.cfg.imagination_ctx_frames, context=ctx_states)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                flow_vals_raw = self.value_model(flow_windows)
                if self.cfg.use_symlog:
                    flow_vals_raw = symexp(flow_vals_raw)
                flow_values = flow_vals_raw.squeeze(-1).squeeze(-1).float().cpu().numpy()
        else:
            euler_pred = None
            flow_aligned = None
            flow_values = None

        # 4. Calculate predicted values for encoded and predicted states
        states_idx = raw_states[idx:idx+1].transpose(0, 1)
        enc_windows = self._get_history_windows(states_idx, self.cfg.imagination_ctx_frames, context=None)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            enc_vals_raw = self.value_model(enc_windows)
            if self.cfg.use_symlog:
                enc_vals_raw = symexp(enc_vals_raw)
            enc_values = enc_vals_raw.squeeze(-1).squeeze(-1).float().cpu().numpy()[ctx:]

        # AR model value predictions
        if ar_aligned_states is not None:
            ar_windows = self._get_history_windows(ar_aligned_states.transpose(0, 1), self.cfg.imagination_ctx_frames, context=ctx_states)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                ar_vals_raw = self.value_model(ar_windows)
                if self.cfg.use_symlog:
                    ar_vals_raw = symexp(ar_vals_raw)
                ar_values = ar_vals_raw.squeeze(-1).squeeze(-1).float().cpu().numpy()
        else:
            ar_values = None

        # Draw values on encoded frames
        encoded_with_val = []
        for t in range(encoded.size(0)):
            val = enc_values[t]
            frame = encoded[t].float()
            img_pil = to_pil_image(frame)
            draw = ImageDraw.Draw(img_pil)
            text = f"v:{val:.2f}"
            x, y = 2, 2
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                draw.text((x + dx, y + dy), text, fill="black")
            draw.text((x, y), text, fill="white")
            encoded_with_val.append(to_tensor(img_pil))
        encoded_with_val = torch.stack(encoded_with_val, dim=0)

        def annotate_frames(frames_cpu, mode_text, mode_color, values_array=None):
            """Add mode label to bottom of each frame and value to top left."""
            annotated = []
            for t in range(frames_cpu.size(0)):
                frame = frames_cpu[t].float()
                img_pil = to_pil_image(frame)
                draw = ImageDraw.Draw(img_pil)
                w, h = img_pil.size
                
                # Bottom mode label
                x_m, y_m = 2, h - 12
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                    draw.text((x_m + dx, y_m + dy), mode_text, fill="black")
                draw.text((x_m, y_m), mode_text, fill=mode_color)
                
                # Top value label if provided
                if values_array is not None:
                    val = values_array[t]
                    val_text = f"v:{val:.2f}"
                    x_v, y_v = 2, 2
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                        draw.text((x_v + dx, y_v + dy), val_text, fill="black")
                    draw.text((x_v, y_v), val_text, fill="white")
                    
                annotated.append(to_tensor(img_pil))
            return torch.stack(annotated, dim=0)
        
        grid_actual = torchvision.utils.make_grid(actual, nrow=actual.size(0), normalize=False)
        grid_encoded = torchvision.utils.make_grid(encoded_with_val, nrow=encoded_with_val.size(0), normalize=False)
        
        grid_elements = [grid_actual, grid_encoded]
        
        if ar_aligned is not None:
            ar_annotated = annotate_frames(ar_aligned, "AR", "cyan", values_array=ar_values)
            grid_ar = torchvision.utils.make_grid(ar_annotated, nrow=ar_annotated.size(0), normalize=False)
            grid_elements.append(grid_ar)
            
        if flow_aligned is not None:
            flow_annotated = annotate_frames(flow_aligned, "FLOW", "red", values_array=flow_values)
            grid_flow = torchvision.utils.make_grid(flow_annotated, nrow=flow_annotated.size(0), normalize=False)
            grid_elements.append(grid_flow)
            
        combined_grid = torch.cat(grid_elements, dim=1)
        grid_name = "Reconstruction/Actual_vs_Encoded"
        if ar_aligned is not None:
            grid_name += "_vs_AR"
        if flow_aligned is not None:
            grid_name += "_vs_Flow"
            
        self.writer.add_image(grid_name, combined_grid, self.world_step)

        # Plot predicted values comparison
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 4))
            timesteps = np.arange(len(enc_values))
            ax.plot(timesteps, enc_values, label='Encoded (GT)', color='green', marker='o')
            if ar_values is not None:
                ax.plot(timesteps, ar_values, label='AR Imagination', color='cyan', marker='s')
            if flow_values is not None:
                ax.plot(timesteps, flow_values, label='FLOW Imagination', color='red', marker='^')
            ax.set_xlabel('Imagination Steps')
            ax.set_ylabel('Predicted Value')
            ax.set_title(f'Value Predictions Comparison (Step {self.world_step})')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            
            self.writer.add_figure('Reconstruction/Value_Predictions', fig, self.world_step)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to log value predictions plot: {e}")
        
        # --- Velocity field diagnostics ---
        if self.cfg.imagination_mode == 'flow' and self.world.dynamics_flow is not None and euler_pred is not None:
            target_states_all = raw_states[idx:idx+1, 1:]
            future_targets = target_states_all[:, ctx-1:]
            source = ctx_states[:, -1:].expand_as(euler_pred)
            v_gt = future_targets - source
            
            v_at_0 = self.world.dynamics_flow.compute_velocity(
                source.clone(), torch.zeros(1, device=source.device), ctx_states, action_embedding
            )
            v_at_05 = self.world.dynamics_flow.compute_velocity(
                0.5 * source + 0.5 * future_targets,
                torch.full((1,), 0.5, device=source.device), ctx_states, action_embedding
            )
            v_at_09 = self.world.dynamics_flow.compute_velocity(
                0.1 * source + 0.9 * future_targets,
                torch.full((1,), 0.9, device=source.device), ctx_states, action_embedding
            )
            
            mse_t0 = (v_at_0 - v_gt).pow(2).mean().item()
            mse_t05 = (v_at_05 - v_gt).pow(2).mean().item()
            mse_t09 = (v_at_09 - v_gt).pow(2).mean().item()
            self.writer.add_scalar('Diag/Velocity_MSE_t0', mse_t0, self.world_step)
            self.writer.add_scalar('Diag/Velocity_MSE_t05', mse_t05, self.world_step)
            self.writer.add_scalar('Diag/Velocity_MSE_t09', mse_t09, self.world_step)
            
            cos_t0 = torch.nn.functional.cosine_similarity(
                v_at_0.reshape(1, -1), v_gt.reshape(1, -1), dim=1
            ).item()
            self.writer.add_scalar('Diag/Velocity_Cosine_t0', cos_t0, self.world_step)
            
            euler_pos_var = euler_pred.var(dim=1).mean().item()
            target_pos_var = future_targets.var(dim=1).mean().item()
            self.writer.add_scalar('Diag/Euler_PositionVariance', euler_pos_var, self.world_step)
            self.writer.add_scalar('Diag/Target_PositionVariance', target_pos_var, self.world_step)
            
            euler_mse = (euler_pred - future_targets).pow(2).mean().item()
            self.writer.add_scalar('Diag/Euler_MSE_to_Target', euler_mse, self.world_step)

    def train_agent(self, start_states, real_actions, real_rewards, real_terminals):
        """
        Updates Actor and Value models using differentiable imagination.
        Imagination is seeded with a window of `imagination_ctx_frames` consecutive real
        latent frames + their actual rollout actions so the dynamics KV cache has
        genuine temporal context before any imagined step.
        Analytical gradients flow from the Value/Reward signal back into the Actor.

        Args:
            start_states: (B, T, D)   — raw encoded states from the world model forward pass.
            real_actions: (B, T-1, A) — raw actions from the replay buffer that caused
                                        those state transitions.
            real_rewards: (B, T, 1)   - real rewards from the replay buffer.
            real_terminals: (B, T, 1) - real terminals from the replay buffer.
        """
        self.world.freeze()

        ctx = self.cfg.imagination_ctx_frames  # e.g. 3
        B_wm, T, D = start_states.shape
        A = real_actions.size(-1)                # raw action dim

        # -----------------------------------------------------------------
        # Sample random context windows of length `ctx` from the real batch.
        # States:  s_t ... s_{t+ctx-1}            → (batch_size, ctx, D)
        # Actions: a_t ... a_{t+ctx-2}            → (batch_size, ctx-1, A)
        #   (ctx-1 real actions that caused those state transitions)
        # -----------------------------------------------------------------
        max_start = T - ctx  # last valid start index
        if max_start < 0:
            # Fallback: pad with first frame if sequence is shorter than ctx
            pad_s = start_states[:, :1].expand(B_wm, ctx - T, D)
            start_states = torch.cat([pad_s, start_states], dim=1)
            # Pad actions with zeros at the front
            pad_a = real_actions[:, :1].expand(B_wm, ctx - T, A) * 0
            real_actions = torch.cat([pad_a, real_actions], dim=1)
            # Pad rewards and terminals as well
            pad_r = real_rewards[:, :1].expand(B_wm, ctx - T, 1) * 0
            real_rewards = torch.cat([pad_r, real_rewards], dim=1)
            pad_term = real_terminals[:, :1].expand(B_wm, ctx - T, 1) * 0
            real_terminals = torch.cat([pad_term, real_terminals], dim=1)
            max_start = 0

        # Draw batch_size random (batch_item, start_idx) pairs
        batch_indices = torch.randint(0, B_wm, (self.cfg.batch_size,))
        time_indices  = torch.randint(0, max_start + 1, (self.cfg.batch_size,))

        # ctx_windows:       (batch_size, ctx,   D) — real reference frames
        # ctx_action_windows:(batch_size, ctx-1, A) — real actions between those frames
        ctx_windows = torch.stack([
            start_states[b, t : t + ctx]
            for b, t in zip(batch_indices.tolist(), time_indices.tolist())
        ], dim=0)  # (batch_size, ctx, D)

        ctx_action_windows = torch.stack([
            real_actions[b, t : t + ctx - 1]          # ctx-1 real actions
            for b, t in zip(batch_indices.tolist(), time_indices.tolist())
        ], dim=0)  # (batch_size, ctx-1, A)

        self.world.reset_cache()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):

            # Seed the actor history with the real context frames
            history_list = list(ctx_windows.unbind(dim=1))
            while len(history_list) < self.cfg.imagination_ctx_frames:
                history_list.insert(0, history_list[0])
            history_list = history_list[-self.cfg.imagination_ctx_frames:]

            actor_input = torch.stack(history_list, dim=1).reshape(self.cfg.batch_size, -1)

            # Generate squashed action chunk, its squashed log probability, and analytical entropy
            action_chunk_scaled, squashed_log_prob, analytical_entropy = self.actor(actor_input)

            ctx_actions_real = ctx_action_windows.to(device=self.device, dtype=action_chunk_scaled.dtype)
            
            if self.cfg.reinforce:
                action_chunk_scaled_for_wm = action_chunk_scaled.detach()
            else:
                action_chunk_scaled_for_wm = action_chunk_scaled

            actions_full = torch.cat([ctx_actions_real, action_chunk_scaled_for_wm], dim=1) # (batch_size, ctx - 1 + H, A)

            # 1-step parallel generation pass
            imagined_states, imagined_rewards, imagined_terminals = self.world.generate_chunk(
                actions=actions_full,
                start_states=ctx_windows,
                flow_standardize_latents=self.cfg.flow_standardize_latents,
                flow_cfg_scale=self.cfg.flow_cfg_scale,
                imagination_mode=self.cfg.imagination_mode,
            )

            history_windows = self._get_history_windows(imagined_states, self.cfg.imagination_ctx_frames, context=ctx_windows)

            imagined_values = self.value_model(history_windows)
            
            # EMA values — differentiable w.r.t. inputs (gradient flows through
            # imagined_states to actor), but target model params are frozen.
            ema_imagined_values = self.target_value_model(history_windows)
                
            pred_term_probs = torch.sigmoid(imagined_terminals)
            pcont = (1.0 - pred_term_probs).detach()
            
            value_imag_loss, targets_v = self.value_loss_fn(
                imagined_values, 
                imagined_rewards, 
                pcont=pcont, 
                target_values=ema_imagined_values.detach()  # fully detached for value bootstrap
            )

            # --- 1. Combined Value Loss ---
            v_loss = self.cfg.beta_imag_value * value_imag_loss

            # Compute actor targets using the EMA value model (more stable estimates)
            targets_a = self.value_loss_fn._compute_lambda_returns(
                ema_imagined_values,
                imagined_rewards,
                pcont
            )

            # Optionally compute advantages (returns - value baseline) for contrastive signal
            if self.cfg.use_advantage or self.cfg.reinforce:
                actor_targets = targets_a - imagined_values[:-1]
            else:
                actor_targets = targets_a
            
            # Optionally apply EMA percentile normalization
            if self.return_ema is not None:
                if self.cfg.reinforce:
                    # For REINFORCE, update with returns but normalize advantages by dividing by scale (preserving sign)
                    self.return_ema.update(targets_a)
                    scale = (self.return_ema.high - self.return_ema.low).clamp(min=1e-2)
                    normalized_targets = actor_targets / scale
                else:
                    self.return_ema.update(actor_targets)
                    normalized_targets = self.return_ema.normalize(actor_targets)
            else:
                normalized_targets = actor_targets

            # Determine whether to use analytical or squashed entropy
            if self.cfg.reinforce:
                entropy_to_use = analytical_entropy.transpose(0, 1)
            else:
                entropy_to_use = -squashed_log_prob.transpose(0, 1)
            log_prob = squashed_log_prob.transpose(0, 1)

            a_loss, a_target_loss, a_entropy_loss, entropy = self.actor_loss_fn(
                normalized_targets, 
                entropy_to_use, 
                log_prob=log_prob, 
                pcont=pcont
            )

        self.value_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)

        # Backward value loss first (updates value_model weights)
        self.scaler.scale(v_loss).backward(retain_graph=True)

        self.actor_opt.zero_grad(set_to_none=True)

        # Temporarily freeze value model parameters to prevent actor loss from updating them
        for p in self.value_model.parameters():
            p.requires_grad = False

        # Compute individual grad norms for actor loss components
        scale_factor = self.scaler.get_scale()

        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(a_target_loss).backward(retain_graph=True)
        a_target_grad_norm = self._compute_grad_norm(self.actor) / scale_factor

        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(a_entropy_loss).backward(retain_graph=True)
        a_entropy_grad_norm = self._compute_grad_norm(self.actor) / scale_factor

        # Combined backward for actual parameter update
        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(a_loss).backward()

        # Restore requires_grad on value model parameters
        for p in self.value_model.parameters():
            p.requires_grad = True

        self.scaler.unscale_(self.value_opt)
        self.scaler.unscale_(self.actor_opt)

        v_grad_norm = self._compute_grad_norm(self.value_model)
        a_grad_norm = self._compute_grad_norm(self.actor)

        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)

        self.scaler.step(self.value_opt)
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        self.writer.add_scalar('Loss/Agent_Value', v_loss.item(), self.agent_step)
        self.writer.add_scalar('Loss/Agent_Value_Imag', value_imag_loss.item(), self.agent_step)
        self.writer.add_scalar('Loss/Agent_Actor', a_loss.item(), self.agent_step)
        self.writer.add_scalar('Loss/Agent_Actor_Target', a_target_loss.item(), self.agent_step)
        self.writer.add_scalar('Loss/Agent_Actor_Entropy', a_entropy_loss.item(), self.agent_step)
        self.writer.add_scalar('Loss/Agent_Entropy', entropy.item(), self.agent_step)
        self.writer.add_scalar('GradNorm/Agent_Value', v_grad_norm, self.agent_step)
        self.writer.add_scalar('GradNorm/Agent_Actor', a_grad_norm, self.agent_step)
        self.writer.add_scalar('GradNorm/Agent_Actor_Target', a_target_grad_norm, self.agent_step)
        self.writer.add_scalar('GradNorm/Agent_Actor_Entropy', a_entropy_grad_norm, self.agent_step)
        if self.return_ema is not None:
            self.writer.add_scalar('ReturnEMA/Low', self.return_ema.low.item(), self.agent_step)
            self.writer.add_scalar('ReturnEMA/High', self.return_ema.high.item(), self.agent_step)
            self.writer.add_scalar('ReturnEMA/Scale', (self.return_ema.high - self.return_ema.low).clamp(min=1e-2).item(), self.agent_step)
        # Run diagnostics periodically
        if self.cfg.diag_enabled and self.agent_step % self.cfg.diag_every_n_steps == 0 and self.agent_step > 0:
            self._run_inline_diagnostics(start_states, real_actions)

        self.agent_step += 1

        return v_loss.item(), a_loss.item()

    def _run_inline_diagnostics(self, start_states, real_actions):
        """Run actor diagnostics and log results to TensorBoard + save plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        step = self.agent_step
        save_dir = os.path.join(self.diag_output_dir, f'step_{step:06d}')
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[Diagnostics] Running at agent_step={step}...")

        was_training_actor = self.actor.training
        was_training_value = self.value_model.training
        self.actor.eval()
        self.value_model.eval()

        try:
            # 0. Flow matching diagnostics (PCA trajectories and velocity fields)
            if self.flow_diagnostics is not None:
                flow_data = self.flow_diagnostics.collect_flow_data(start_states, real_actions)
                fig_traj = self.flow_diagnostics.plot_flow_trajectories(flow_data, save_dir=save_dir)
                if fig_traj is not None:
                    self.writer.add_figure('Diag/Flow_Matching/Trajectories', fig_traj, step)
                    plt.close(fig_traj)
                self.writer.add_scalar('Diag/Flow_Matching/Cosine_Sim_t0', flow_data['cos_sim'], step)
                self.writer.add_scalar('Diag/Flow_Matching/Dist_to_Target_Mean', flow_data['dist_to_target'], step)

                # Flow matching action impact diagnostics
                flow_impact_data = self.flow_diagnostics.collect_flow_action_impact(start_states, real_actions)
                fig_flow_impact = self.flow_diagnostics.plot_flow_action_impact(flow_impact_data, save_dir=save_dir)
                if fig_flow_impact is not None:
                    self.writer.add_figure('Diag/Flow_Matching/Action_Impact', fig_flow_impact, step)
                    plt.close(fig_flow_impact)
                self.writer.add_scalar('Diag/Flow_Matching/Zero_Action_MSE_Final', flow_impact_data['mse_zero'][-1], step)
                self.writer.add_scalar('Diag/Flow_Matching/Rand_Action_MSE_Final', flow_impact_data['mse_rand'][-1], step)
                self.writer.add_scalar('Diag/Flow_Matching/Opp_Action_MSE_Final', flow_impact_data['mse_opp'][-1], step)

            # 1. Action distribution stats (fast)
            action_stats = self.action_monitor.collect_action_stats(start_states, real_actions)
            fig_health, fig_hist, fig_profile = self.action_monitor.plot_action_stats(action_stats, save_dir=save_dir)
            if fig_health is not None:
                self.writer.add_figure('Diag/Action/Health', fig_health, step)
                plt.close(fig_health)
            if fig_hist is not None:
                self.writer.add_figure('Diag/Action/PreTanh_Hist', fig_hist, step)
                plt.close(fig_hist)
            if fig_profile is not None:
                self.writer.add_figure('Diag/Action/Profile', fig_profile, step)
                plt.close(fig_profile)

            self.writer.add_scalar('Diag/Action/PreTanh_MeanAbs_Max', float(action_stats['pre_tanh_mean_abs'].max()), step)
            self.writer.add_scalar('Diag/Action/PostTanh_MeanAbs_Max', float(action_stats['post_tanh_mean_abs'].max()), step)
            self.writer.add_scalar('Diag/Action/Std_Min', float(action_stats['pre_tanh_std'].min()), step)
            self.writer.add_scalar('Diag/Action/AnalyticalEntropy_Mean', float(action_stats['analytical_entropy'].mean()), step)

            # 2. Loss landscape (moderate cost)
            landscape_data = self.landscape_viz.compute_landscape(
                start_states, real_actions,
                grid_size=self.cfg.diag_landscape_grid,
                range_scale=self.cfg.diag_landscape_range
            )
            landscape_figs = self.landscape_viz.plot_landscape(landscape_data, save_dir=save_dir)
            if landscape_figs is not None:
                for fig_name, fig in landscape_figs.items():
                    self.writer.add_figure(f'Diag/Loss_Landscape/Landscape_{fig_name.title()}', fig, step)
                    plt.close(fig)

            total = landscape_data['total_loss']
            center = total[self.cfg.diag_landscape_grid // 2, self.cfg.diag_landscape_grid // 2]
            self.writer.add_scalar('Diag/Loss_Landscape/CenterLoss', center, step)
            self.writer.add_scalar('Diag/Loss_Landscape/Std', float(total.std()), step)
            self.writer.add_scalar('Diag/Loss_Landscape/Range', float(total.max() - total.min()), step)

            # 3. Gradient slice (moderate cost)
            slice_data = self.landscape_viz.compute_gradient_slice(
                start_states, real_actions, num_points=21,
                range_scale=self.cfg.diag_landscape_range
            )
            fig_slice = self.landscape_viz.plot_gradient_slice(slice_data, save_dir=save_dir)
            if fig_slice is not None:
                self.writer.add_figure('Diag/Gradients/Slice', fig_slice, step)
                plt.close(fig_slice)
            if slice_data:
                self.writer.add_scalar('Diag/Gradients/GradNorm_Actor', slice_data['grad_norm'], step)

            # 4. Per-layer gradient norms
            self.landscape_viz._run_actor_forward_with_grad(start_states, real_actions)
            norms = self.grad_analyzer.per_layer_grad_norms()
            fig_norms = self.grad_analyzer.plot_grad_norms(norms, save_dir=save_dir)
            if fig_norms is not None:
                self.writer.add_figure('Diag/Gradients/Layer_Grad_Norms', fig_norms, step)
                plt.close(fig_norms)

            vanishing = sum(1 for v in norms.values() if v < 1e-7)
            self.writer.add_scalar('Diag/Gradients/VanishingLayers', vanishing, step)

            # 5. Finite-difference check (expensive, optional)
            if self.cfg.diag_fd_check:
                fd_data = self.grad_analyzer.finite_difference_check(
                    start_states, real_actions, num_params=self.cfg.diag_fd_params
                )
                fig_fd = self.grad_analyzer.plot_fd_comparison(fd_data, save_dir=save_dir)
                if fig_fd is not None:
                    self.writer.add_figure('Diag/FD_Check/Check', fig_fd, step)
                    plt.close(fig_fd)
                self.writer.add_scalar('Diag/FD_Check/CosineSim', fd_data['cosine_similarity'], step)
                self.writer.add_scalar('Diag/FD_Check/RelError', fd_data['mean_relative_error'], step)

            # 6. Autoregressive action impact diagnostics
            if self.ar_diagnostics is not None:
                ar_data = self.ar_diagnostics.collect_ar_data(start_states, real_actions)
                fig_ar = self.ar_diagnostics.plot_ar_trajectories(ar_data, save_dir=save_dir)
                if fig_ar is not None:
                    self.writer.add_figure('Diag/AR_Dynamics/Action_Impact', fig_ar, step)
                    plt.close(fig_ar)
                self.writer.add_scalar('Diag/AR_Dynamics/Zero_Action_MSE_Final', ar_data['mse_zero'][-1], step)
                self.writer.add_scalar('Diag/AR_Dynamics/Rand_Action_MSE_Final', ar_data['mse_rand'][-1], step)
                self.writer.add_scalar('Diag/AR_Dynamics/Opp_Action_MSE_Final', ar_data['mse_opp'][-1], step)
                self.writer.add_scalar('Diag/AR_Dynamics/Dist_to_Target', ar_data['dist_to_target'], step)

            print(f"[Diagnostics] Done. Plots saved to {save_dir}")

        except Exception as e:
            import traceback
            print(f"[Diagnostics] Error: {e}")
            traceback.print_exc()
        finally:
            if was_training_actor:
                self.actor.train()
            if was_training_value:
                self.value_model.train()

    def run(self):
        try:
            print(f"Pre-filling buffer with {self.cfg.prefill_episodes} episodes...")
            pbar = tqdm(total=self.cfg.prefill_episodes, desc="Filling Buffer", unit="ep")
            
            while len(self.buffer.episodes) < self.cfg.prefill_episodes:
                self.collect_episodes(max_steps=250)
                pbar.n = min(len(self.buffer.episodes), self.cfg.prefill_episodes)
                pbar.refresh()
                
            pbar.close()
            print("Buffer is full! Starting training...")

            # ── World-model bootstrap ────────────────────────────────────────
            if self.cfg.world_bootstrap_steps > 0:
                print(f"Bootstrapping world model for {self.cfg.world_bootstrap_steps} steps...")
                for bs_step in tqdm(range(self.cfg.world_bootstrap_steps),
                                    desc="WM Bootstrap", unit="step"):
                    batch = self.buffer.sample_sequences(self.cfg.batch_size, self.cfg.max_frames)
                    if batch is None:
                        continue
                    self.train_world(batch, skip_value=True)
                print("World model bootstrap complete.")
            # ────────────────────────────────────────────────────────────────

            # Post-bootstrapping buffer capacity adjustment
            print(f"Post-bootstrapping: changing buffer capacity from {self.buffer.capacity} to {self.cfg.post_bootstrap_buffer_capacity}.")
            self.buffer.set_capacity(self.cfg.post_bootstrap_buffer_capacity)

            agent_updates_per_world = int(1.0 / self.cfg.world_agent_ratio) if self.cfg.world_agent_ratio < 1 else 1
            world_update_freq = int(self.cfg.world_agent_ratio) if self.cfg.world_agent_ratio >= 1 else 1

            for epoch in range(self.cfg.num_epochs):
                ep_ret_avg, ep_len_avg = self.collect_episodes(max_steps=250)
                
                self.writer.add_scalar('Environment/Avg_Episode_Return', ep_ret_avg, epoch)
                self.writer.add_scalar('Environment/Avg_Episode_Length', ep_len_avg, epoch)
                
                val_ret = self.evaluate(epoch, max_steps=250)
                
                avg_wm_loss, avg_v_loss, avg_a_loss = [], [], []
                
                for step in range(self.cfg.train_steps):
                    batch = self.buffer.sample_sequences(self.cfg.batch_size, self.cfg.max_frames)
                    if batch is None: continue
                    
                    clean_states, real_actions, real_rewards, real_terminals, wm_loss = self.train_world(batch)
                    avg_wm_loss.append(wm_loss)

                    if step % world_update_freq == 0:
                        for _ in range(agent_updates_per_world):
                            # clean_states:  (B, T,   D) — real encoded states
                            # real_actions:  (B, T-1, A) — real actions from the replay buffer
                            v_loss, a_loss = self.train_agent(
                                clean_states, real_actions, real_rewards, real_terminals
                            )
                            avg_v_loss.append(v_loss)
                            avg_a_loss.append(a_loss)

                        # One target-network update after all value updates
                        # for this iteration (real V + imagined V).
                        self.update_value_target()
                    
                    self.global_step += 1
                    
                wm_str = f"{np.mean(avg_wm_loss):.3f}" if avg_wm_loss else "N/A"
                v_str = f"{np.mean(avg_v_loss):.3f}" if avg_v_loss else "N/A"
                a_str = f"{np.mean(avg_a_loss):.3f}" if avg_a_loss else "N/A"
                    
                print(f"Epoch {epoch+1:03d} | Train Ret: {ep_ret_avg:.1f} | Val Ret: {val_ret:.1f} | "
                      f"WM: {wm_str} | V: {v_str} | A: {a_str}")
                
        finally:
            self.envs.close()
            self.val_env.close()
            self.writer.close()

if __name__ == '__main__':

    config = Config(
        domain='cartpole',
        task='swingup'
    )
    
    trainer = Trainer(config)
    trainer.run()