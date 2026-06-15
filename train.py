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
from models.world_helpers import VisionEncoder, ActionEncoder, Reward, VisionDecoder, Termination
import torchvision
from models.world import World
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
    
    # Autoregressive dynamics (for encoder/reward/termination training)
    dyn_num_layers: int = 6
    dyn_num_heads: int = 4
    dyn_causal: bool = True  # Causal: teacher-forced AR, mask prevents seeing future states
    
    # Flow matching dynamics (for actor imagination)
    flow_num_layers: int = 6
    flow_num_heads: int = 4
    flow_causal: bool = True   # Causal sequence mask: prevents future target leakage during training
    flow_num_euler_steps: int = 6
    flow_source_noise_sigma: float = 0.05
    flow_loss_weight: float = 1.0
    flow_detach_encoder: bool = True    # Stop-grad encoder outputs for flow model
    flow_distill_from_ar: bool = False  # If True, flow trains on AR predictions instead of encoder outputs
    flow_cfg_dropout: float = 0.15      # Probability of dropping context during flow training (CFG)
    flow_cfg_scale: float = 3.0         # CFG extrapolation scale during actor imagination
    flow_standardize_latents: bool = True # Standardize latents to standard normal for flow model
    flow_adjoint_method: str = "none"   # Adjoint method ('none', 'custom', or 'torchdiffeq')
    
    max_frames: int = 18
    world_horizon: int = 15
    agent_horizon: int = 3
    batch_size: int = 96  
    train_steps: int = 100
    num_epochs: int = 500
    
    imagination_ctx_frames: int = 3   # minimum real frames fed as context before imagining

    world_agent_ratio: float = 1.0  
    
    world_lr: float = 1e-3
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    grad_clip_norm: float = 100.0
    use_amp: bool = True 
    amp_dtype: str = 'bfloat16' 
    value_target_tau: float = 0.02
    
    use_sigreg: bool = True
    sigreg_weight: float = 0.1
    weak_sigreg: bool = True     # if True, use WeakSIGReg (Frobenius cov loss) instead of SIGReg
    entropy_scale: float = 3e-3
    use_return_ema: bool = True
    use_advantage: bool = False
    reinforce: bool = False
    discount: float = 0.99
    actor_num_blocks: int = 4
    value_num_blocks: int = 4
    use_symlog: bool = True

    buffer_capacity: int = 150
    prefill_episodes: int = 50
    world_bootstrap_steps: int = 500  # extra WM-only gradient steps run after prefill, before training loop
    
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
                f"ah{self.agent_horizon}",
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
            partial(make_env, cfg.domain, cfg.task, cfg.seed + i, cfg.image_size, cfg.image_size) 
            for i in range(cfg.num_envs)
        ]
        self.envs = SubprocVecEnv(env_fns)
        self.action_dim = self.envs.get_action_space().shape[0]
        
        self.val_env = make_env(cfg.domain, cfg.task, cfg.seed * 100, cfg.image_size, cfg.image_size)
        
        vision = VisionEncoder(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        action_enc = ActionEncoder(action_space=self.envs.get_action_space(), hidden_dim=cfg.hidden_dim, output_dim=cfg.latent_dim)
        dynamics_ar = Dynamics(
            max_frames=cfg.max_frames + cfg.imagination_ctx_frames,
            action_dim=cfg.latent_dim, hidden_dim=cfg.latent_dim,
            num_layers=cfg.dyn_num_layers, num_heads=cfg.dyn_num_heads,
            causal=cfg.dyn_causal,
        )
        dynamics_flow = FlowMatchingDynamics(
            max_frames=cfg.max_frames + cfg.imagination_ctx_frames,
            action_dim=cfg.latent_dim, hidden_dim=cfg.latent_dim,
            num_layers=cfg.flow_num_layers, num_heads=cfg.flow_num_heads,
            causal=cfg.flow_causal, num_euler_steps=cfg.flow_num_euler_steps,
            source_noise_sigma=cfg.flow_source_noise_sigma,
            adjoint_method=cfg.flow_adjoint_method,
        )
        reward_enc = Reward(obs_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        termination_enc = Termination(obs_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        
        if cfg.recon_debug:
            decoder = VisionDecoder(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
        else:
            decoder = None
            
        self.world = World(vision, dynamics_ar, dynamics_flow, action_enc, reward_enc, termination_enc, self.cfg.world_horizon, decoder=decoder).to(self.device)
        self.actor = Actor(action_space=self.envs.get_action_space(), obs_dim=cfg.agent_horizon * cfg.latent_dim, hidden_dim=cfg.hidden_dim, chunk_size=cfg.world_horizon, num_blocks=cfg.actor_num_blocks).to(self.device)
        self.value_model = Value(obs_dim=cfg.agent_horizon * cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.value_num_blocks).to(self.device)
        self.target_value_model = Value(obs_dim=cfg.agent_horizon * cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_blocks=cfg.value_num_blocks).to(self.device)
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        self.target_value_model.eval()
        for p in self.target_value_model.parameters():
            p.requires_grad = False
        
        if cfg.weak_sigreg:
            self.sig_reg = WeakSIGReg().to(self.device)
        else:
            self.sig_reg = SIGReg().to(self.device)
        self.value_loss_fn = ValueLoss(discount=cfg.discount, batch_first=False, use_symlog=cfg.use_symlog)
        self.actor_loss_fn = ActorLoss(discount=cfg.discount, batch_first=False, entropy_scale=cfg.entropy_scale, reinforce=cfg.reinforce)
        self.return_ema = ReturnEMA(decay=0.99).to(self.device) if cfg.use_return_ema else None
        
        self.world_opt = optim.Adam(self.world.parameters(), lr=cfg.world_lr)
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
            from diagnostics import LossLandscapeVisualizer, GradientFlowAnalyzer, ActionDistributionMonitor, FlowDiagnostics
            self.landscape_viz = LossLandscapeVisualizer(self)
            self.grad_analyzer = GradientFlowAnalyzer(self)
            self.action_monitor = ActionDistributionMonitor(self)
            self.flow_diagnostics = FlowDiagnostics(self)
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
            ctx_feed = ctx_states[-horizon:-1] # (horizon - 1, B, D)
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
        
        actor_history = deque(maxlen=self.cfg.agent_horizon)
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
                while len(history_list) < self.cfg.agent_horizon:
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

    def train_world(self, batch):
        obs_batch, act_batch, rew_batch, term_batch = [b.to(self.device, non_blocking=True) for b in batch]
        if self.cfg.use_symlog:
            sym_rew_batch = symlog(rew_batch)
        else:
            sym_rew_batch = rew_batch
        
        self.world.unfreeze()
        
        log_recon = self.cfg.recon_debug and (self.world_step % self.cfg.train_steps == 0)
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            raw_states, target_state, pred_next_state, pred_rewards, pred_terminals, velocity_loss, euler_full = self.world(
                act_batch, obs_batch, ctx_frames=self.cfg.imagination_ctx_frames, run_euler=log_recon,
                flow_detach_encoder=self.cfg.flow_detach_encoder,
                flow_distill_from_ar=self.cfg.flow_distill_from_ar,
                flow_standardize_latents=self.cfg.flow_standardize_latents,
                flow_cfg_dropout=self.cfg.flow_cfg_dropout,
                flow_cfg_scale=self.cfg.flow_cfg_scale,
            )
            
            recon_loss = None
            recon_pred_loss = None
            if self.cfg.recon_debug:
                if self.cfg.recon_train:
                    recon_obs = self.world.decoder(raw_states)
                else:
                    recon_obs = self.world.decoder(raw_states.detach())
                    
                recon_loss = F.mse_loss(recon_obs.float(), obs_batch.float())
                total_recon_loss = recon_loss
                
                if log_recon and pred_next_state is not None:
                    if self.cfg.recon_train:
                        recon_pred_obs = self.world.decoder(pred_next_state)
                    else:
                        recon_pred_obs = self.world.decoder(pred_next_state.detach())
                    recon_pred_loss = F.mse_loss(recon_pred_obs.float(), obs_batch[:, 1:].float())
            
        # Autoregressive dynamics loss (MSE to detached encoder targets)
        ar_dyn_loss = F.mse_loss(pred_next_state.float(), target_state.detach().float())
        
        # Flow matching velocity loss (separate, optionally detached from encoder)
        flow_loss = self.cfg.flow_loss_weight * velocity_loss
        
        # Combined dynamics loss (AR + flow)
        dyn_loss = ar_dyn_loss + flow_loss
        
        rew_loss = F.mse_loss(pred_rewards.float(), sym_rew_batch[:, 1:].float())
        term_loss = F.binary_cross_entropy_with_logits(pred_terminals.float(), term_batch[:, 1:].float())
        
        if self.cfg.use_sigreg:
            sig_loss = self.sig_reg(raw_states.transpose(0, 1).float())
        else:
            sig_loss = torch.tensor(0.0, device=self.device)
        
        if self.cfg.recon_debug:
            wm_loss = dyn_loss + rew_loss + term_loss + total_recon_loss
        else:
            wm_loss = dyn_loss + rew_loss + term_loss
            
        if self.cfg.use_sigreg:
            wm_loss = wm_loss + self.cfg.sigreg_weight * sig_loss
        
        # Compute individual grad norms for world model losses
        scale_factor = self.scaler.get_scale()
        
        # AR Dynamics Loss
        self.world_opt.zero_grad(set_to_none=True)
        self.scaler.scale(ar_dyn_loss).backward(retain_graph=True)
        ar_dyn_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        
        # Flow Velocity Loss
        self.world_opt.zero_grad(set_to_none=True)
        self.scaler.scale(flow_loss).backward(retain_graph=True)
        flow_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        
        # Reward Loss
        self.world_opt.zero_grad(set_to_none=True)
        self.scaler.scale(rew_loss).backward(retain_graph=True)
        rew_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        
        # Termination Loss
        self.world_opt.zero_grad(set_to_none=True)
        self.scaler.scale(term_loss).backward(retain_graph=True)
        term_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        
        # SIGReg Loss
        if self.cfg.use_sigreg:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(self.cfg.sigreg_weight * sig_loss).backward(retain_graph=True)
            sig_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            sig_grad_norm = 0.0
            
        # Reconstruction Loss
        if self.cfg.recon_debug:
            self.world_opt.zero_grad(set_to_none=True)
            self.scaler.scale(total_recon_loss).backward(retain_graph=True)
            recon_grad_norm = self._compute_grad_norm(self.world) / scale_factor
        else:
            recon_grad_norm = 0.0
            
        # Combined backward
        self.world_opt.zero_grad(set_to_none=True)
        self.scaler.scale(wm_loss).backward()
        self.scaler.unscale_(self.world_opt)
        grad_norm = self._compute_grad_norm(self.world)
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.grad_clip_norm)
        
        self.scaler.step(self.world_opt)
        self.scaler.update()
        
        self.writer.add_scalar('Loss/World_Total', wm_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_AR_Dynamics', ar_dyn_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Flow_Velocity', flow_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Dynamics', dyn_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Reward', rew_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_Termination', term_loss.item(), self.world_step)
        self.writer.add_scalar('Loss/World_SIGReg', sig_loss.item(), self.world_step)
        self.writer.add_scalar('GradNorm/World_SIGReg', sig_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Termination', term_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_AR_Dynamics', ar_dyn_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Flow_Velocity', flow_grad_norm, self.world_step)
        
        if self.cfg.recon_debug:
            self.writer.add_scalar('Loss/World_Reconstruction', recon_loss.item(), self.world_step)
            if recon_pred_loss is not None:
                self.writer.add_scalar('Loss/World_Reconstruction_Pred', recon_pred_loss.item(), self.world_step)
            self.writer.add_scalar('GradNorm/World_Reconstruction', recon_grad_norm, self.world_step)
            
        self.writer.add_scalar('GradNorm/World', grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Dynamics', ar_dyn_grad_norm, self.world_step)
        self.writer.add_scalar('GradNorm/World_Reward', rew_grad_norm, self.world_step)
        
        if self.cfg.recon_debug and (self.world_step % self.cfg.train_steps == 0):
            self.visualize_reconstruction(obs_batch, raw_states, act_batch)
            
        self.world_step += 1
        
        return raw_states.detach(), act_batch.detach(), wm_loss.item()

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
        
        # AR dynamics: feed all real states with causal masking (matches training forward)
        T_act = act_batch.size(1)
        current_states_idx = raw_states[idx:idx+1, :-1]  # (1, T_act, D) — all current states
        all_action_emb = self.world.action(act_batch[idx:idx+1].reshape(-1, AS)).reshape(1, T_act, -1)
        ar_pred = self.world.dynamics_ar(current_states_idx, all_action_emb)  # (1, T_act, D)
        
        # Decode AR predictions
        recon_ar_pred_obs = self.world.decoder(ar_pred.detach())
        ar_aligned = recon_ar_pred_obs[0, ctx-1:].detach().cpu()
        
        # 3. Flow matching Euler predictions
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

        # 4. Calculate predicted values for encoded and predicted states
        states_idx = raw_states[idx:idx+1].transpose(0, 1)
        enc_windows = self._get_history_windows(states_idx, self.cfg.agent_horizon, context=None)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            enc_vals_raw = self.value_model(enc_windows)
            if self.cfg.use_symlog:
                enc_vals_raw = symexp(enc_vals_raw)
            enc_values = enc_vals_raw.squeeze(-1).squeeze(-1).float().cpu().numpy()[ctx:]

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

        def annotate_frames(frames_cpu, mode_text, mode_color):
            """Add mode label to bottom of each frame."""
            annotated = []
            for t in range(frames_cpu.size(0)):
                frame = frames_cpu[t].float()
                img_pil = to_pil_image(frame)
                draw = ImageDraw.Draw(img_pil)
                w, h = img_pil.size
                x_m, y_m = 2, h - 12
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                    draw.text((x_m + dx, y_m + dy), mode_text, fill="black")
                draw.text((x_m, y_m), mode_text, fill=mode_color)
                annotated.append(to_tensor(img_pil))
            return torch.stack(annotated, dim=0)
        
        ar_annotated = annotate_frames(ar_aligned, "AR", "cyan")
        flow_annotated = annotate_frames(flow_aligned, "FLOW", "red")
        
        grid_actual = torchvision.utils.make_grid(actual, nrow=actual.size(0), normalize=False)
        grid_encoded = torchvision.utils.make_grid(encoded_with_val, nrow=encoded_with_val.size(0), normalize=False)
        grid_ar = torchvision.utils.make_grid(ar_annotated, nrow=ar_annotated.size(0), normalize=False)
        grid_flow = torchvision.utils.make_grid(flow_annotated, nrow=flow_annotated.size(0), normalize=False)
        
        combined_grid = torch.cat([grid_actual, grid_encoded, grid_ar, grid_flow], dim=1)
        self.writer.add_image('Reconstruction/Actual_vs_Encoded_vs_AR_vs_Flow', combined_grid, self.world_step)
        
        # --- Velocity field diagnostics ---
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

    def train_agent(self, start_states, real_actions):
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
            while len(history_list) < self.cfg.agent_horizon:
                history_list.insert(0, history_list[0])
            history_list = history_list[-self.cfg.agent_horizon:]

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
            )

            history_windows = self._get_history_windows(imagined_states, self.cfg.agent_horizon, context=ctx_windows)

            imagined_values = self.value_model(history_windows)
            
            # EMA values — differentiable w.r.t. inputs (gradient flows through
            # imagined_states to actor), but target model params are frozen.
            ema_imagined_values = self.target_value_model(history_windows)
                
            pred_term_probs = torch.sigmoid(imagined_terminals)
            pcont = (1.0 - pred_term_probs).detach()
            
            v_loss, targets_v = self.value_loss_fn(
                imagined_values, 
                imagined_rewards, 
                pcont=pcont, 
                target_values=ema_imagined_values.detach()  # fully detached for value bootstrap
            )

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

        self.update_value_target()

        self.writer.add_scalar('Loss/Agent_Value', v_loss.item(), self.agent_step)
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
            flow_data = self.flow_diagnostics.collect_flow_data(start_states, real_actions)
            fig_traj = self.flow_diagnostics.plot_flow_trajectories(flow_data, save_dir=save_dir)
            if fig_traj is not None:
                self.writer.add_figure('Diag/Flow_Trajectories', fig_traj, step)
            self.writer.add_scalar('Diag/Flow_Cosine_Sim_t0', flow_data['cos_sim'], step)
            self.writer.add_scalar('Diag/Flow_Dist_to_Target_Mean', flow_data['dist_to_target'], step)

            # 1. Action distribution stats (fast)
            action_stats = self.action_monitor.collect_action_stats(start_states, real_actions)
            self.action_monitor.plot_action_stats(action_stats, save_dir=save_dir)

            self.writer.add_scalar('Diag/PreTanh_MeanAbs_Max', float(action_stats['pre_tanh_mean_abs'].max()), step)
            self.writer.add_scalar('Diag/PostTanh_MeanAbs_Max', float(action_stats['post_tanh_mean_abs'].max()), step)
            self.writer.add_scalar('Diag/ActionStd_Min', float(action_stats['pre_tanh_std'].min()), step)
            self.writer.add_scalar('Diag/AnalyticalEntropy_Mean', float(action_stats['analytical_entropy'].mean()), step)

            # 2. Loss landscape (moderate cost)
            landscape_data = self.landscape_viz.compute_landscape(
                start_states, real_actions,
                grid_size=self.cfg.diag_landscape_grid,
                range_scale=self.cfg.diag_landscape_range
            )
            self.landscape_viz.plot_landscape(landscape_data, save_dir=save_dir)

            total = landscape_data['total_loss']
            center = total[self.cfg.diag_landscape_grid // 2, self.cfg.diag_landscape_grid // 2]
            self.writer.add_scalar('Diag/Landscape_CenterLoss', center, step)
            self.writer.add_scalar('Diag/Landscape_Std', float(total.std()), step)
            self.writer.add_scalar('Diag/Landscape_Range', float(total.max() - total.min()), step)

            # 3. Gradient slice (moderate cost)
            slice_data = self.landscape_viz.compute_gradient_slice(
                start_states, real_actions, num_points=21,
                range_scale=self.cfg.diag_landscape_range
            )
            self.landscape_viz.plot_gradient_slice(slice_data, save_dir=save_dir)
            if slice_data:
                self.writer.add_scalar('Diag/GradNorm_Actor', slice_data['grad_norm'], step)

            # 4. Per-layer gradient norms
            self.landscape_viz._run_actor_forward_with_grad(start_states, real_actions)
            norms = self.grad_analyzer.per_layer_grad_norms()
            self.grad_analyzer.plot_grad_norms(norms, save_dir=save_dir)

            vanishing = sum(1 for v in norms.values() if v < 1e-7)
            self.writer.add_scalar('Diag/VanishingLayers', vanishing, step)

            # 5. Finite-difference check (expensive, optional)
            if self.cfg.diag_fd_check:
                fd_data = self.grad_analyzer.finite_difference_check(
                    start_states, real_actions, num_params=self.cfg.diag_fd_params
                )
                self.grad_analyzer.plot_fd_comparison(fd_data, save_dir=save_dir)
                self.writer.add_scalar('Diag/FD_CosineSim', fd_data['cosine_similarity'], step)
                self.writer.add_scalar('Diag/FD_RelError', fd_data['mean_relative_error'], step)

            print(f"[Diagnostics] Done. Plots saved to {save_dir}")

        except Exception as e:
            print(f"[Diagnostics] Error: {e}")
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
                    self.train_world(batch)
                print("World model bootstrap complete.")
            # ────────────────────────────────────────────────────────────────

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
                    
                    raw_states, real_actions, wm_loss = self.train_world(batch)
                    avg_wm_loss.append(wm_loss)

                    if step % world_update_freq == 0:
                        for _ in range(agent_updates_per_world):
                            # raw_states:   (B, T,   D) — real encoded states
                            # real_actions: (B, T-1, A) — real actions from the replay buffer
                            v_loss, a_loss = self.train_agent(raw_states, real_actions)
                            avg_v_loss.append(v_loss)
                            avg_a_loss.append(a_loss)
                    
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