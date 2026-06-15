"""
Flow Matching Diagnostics.

Visualizes flow matching trajectories and velocity fields in 2D using PCA.
Plots both the full flow tree and detailed velocity steps for selected sequences.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class FlowDiagnostics:
    """Diagnostic tool to visualize conditional flow matching trajectories and vector fields."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device

    def fit_pca(self, states, q=2):
        """Fit a linear PCA projection onto 2D using PyTorch SVD.
        
        Args:
            states: (N, D) tensor of latent states
            q: target dimensions (default 2)
            
        Returns:
            mean: (1, D) tensor
            V: (D, q) projection matrix
        """
        states = states.float()
        mean = states.mean(dim=0, keepdim=True)
        states_centered = states - mean
        U, S, V = torch.pca_lowrank(states_centered, q=q)
        return mean, V

    @torch.no_grad()
    def collect_flow_data(self, start_states, real_actions):
        """Run step-by-step flow matching integration and collect trajectory/velocity data.
        
        Args:
            start_states: (B, T, D) real encoded states
            real_actions: (B, T-1, A) real actions
            
        Returns:
            dict containing projected trajectory coordinates and batch-wide metrics.
        """
        trainer = self.trainer
        cfg = trainer.cfg
        ctx = cfg.imagination_ctx_frames

        B_wm, T, D = start_states.shape
        A = real_actions.size(-1)
        max_start = T - ctx

        if max_start < 0:
            pad_s = start_states[:, :1].expand(B_wm, ctx - T, D)
            start_states = torch.cat([pad_s, start_states], dim=1)
            pad_a = real_actions[:, :1].expand(B_wm, ctx - T, A) * 0
            real_actions = torch.cat([pad_a, real_actions], dim=1)
            max_start = 0

        # We visualize the first few sequences in detail
        num_viz = min(3, B_wm)
        ctx_windows = start_states[:num_viz, :ctx]
        future_actions = real_actions[:num_viz, ctx-1:]
        future_targets = start_states[:num_viz, ctx:]

        H = future_targets.shape[1]
        AS = future_actions.shape[-1]
        
        with torch.amp.autocast(device_type=self.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype):
            action_embedding = trainer.world.action(future_actions.reshape(-1, AS)).reshape(num_viz, H, -1)

            K = trainer.world.dynamics_flow.num_euler_steps
            dt = 1.0 / K

            # Normalize context/targets if needed
            std_l = cfg.flow_standardize_latents
            if std_l:
                ctx_windows_norm = trainer.world.latent_rms.normalize(ctx_windows)
                future_targets_norm = trainer.world.latent_rms.normalize(future_targets)
            else:
                ctx_windows_norm = ctx_windows
                future_targets_norm = future_targets

            # Start integration at t=0 — in normalized or unnormalized space depending on std_l
            z = ctx_windows_norm[:, -1:].expand(-1, H, -1).clone()
            sigma = trainer.world.dynamics_flow.source_noise_sigma
            if sigma > 0:
                z = z + sigma * torch.randn_like(z)

            trajectory = [z.clone()]
            velocities = []
            cfg_scale = cfg.flow_cfg_scale

            for k in range(K):
                t_k = torch.full((z.size(0),), k * dt, device=z.device, dtype=z.dtype)
                if cfg_scale != 1.0:
                    v_cond = trainer.world.dynamics_flow.compute_velocity(z, t_k, ctx_windows_norm, action_embedding, drop_ctx=False)
                    v_uncond = trainer.world.dynamics_flow.compute_velocity(z, t_k, ctx_windows_norm, action_embedding, drop_ctx=True)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = trainer.world.dynamics_flow.compute_velocity(z, t_k, ctx_windows_norm, action_embedding)
                velocities.append(v)
                z = z + dt * v
                trajectory.append(z.clone())

        # Optimal Transport (OT) linear interpolation for comparison
        # Uses the same noisy source as the integration for fair comparison
        z_0 = trajectory[0]  # already has noise applied
        z_1 = future_targets_norm
        ot_trajectory = []
        for k in range(K + 1):
            t_val = k * dt
            z_ot_k = (1.0 - t_val) * z_0 + t_val * z_1
            ot_trajectory.append(z_ot_k)

        # Fit PCA and project each sequence separately to avoid cross-sequence scale squeezing
        sequences = []
        for i in range(num_viz):
            pts = [
                ctx_windows_norm[i],
                future_targets_norm[i]
            ]
            for traj in trajectory:
                pts.append(traj[i])
            for ot_traj in ot_trajectory:
                pts.append(ot_traj[i])
            
            pts_tensor = torch.cat(pts, dim=0)
            mean, V = self.fit_pca(pts_tensor, q=2)

            # Project coordinates and velocities
            ctx_proj = (ctx_windows_norm[i].float() - mean) @ V
            target_proj = (future_targets_norm[i].float() - mean) @ V
            
            traj_proj = torch.stack([(t[i].float() - mean) @ V for t in trajectory], dim=0)
            ot_proj = torch.stack([(t[i].float() - mean) @ V for t in ot_trajectory], dim=0)
            vel_proj = torch.stack([v[i].float() @ V for v in velocities], dim=0)

            sequences.append({
                'ctx_proj': ctx_proj.cpu().numpy(),
                'target_proj': target_proj.cpu().numpy(),
                'traj_proj': traj_proj.cpu().numpy(),
                'ot_proj': ot_proj.cpu().numpy(),
                'vel_proj': vel_proj.cpu().numpy(),
            })

        # --- Calculate batch-wide metrics ---
        with torch.amp.autocast(device_type=self.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype):
            z_0_all = start_states[:, ctx-1:ctx].expand(-1, T-ctx, -1)
            z_1_all = start_states[:, ctx:]
            
            if std_l:
                z_all_norm = trainer.world.latent_rms.normalize(z_0_all.clone())
                start_states_norm = trainer.world.latent_rms.normalize(start_states)
                z_1_all_norm = trainer.world.latent_rms.normalize(z_1_all)
            else:
                z_all_norm = z_0_all.clone()
                start_states_norm = start_states
                z_1_all_norm = z_1_all

            # Add source noise to match inference
            sigma = trainer.world.dynamics_flow.source_noise_sigma
            if sigma > 0:
                z_all_norm = z_all_norm + sigma * torch.randn_like(z_all_norm)
            
            act_emb_all = trainer.world.action(real_actions[:, ctx-1:].reshape(-1, A)).reshape(B_wm, T-ctx, -1)
            
            v_all_list = []
            for k in range(K):
                t_k = torch.full((B_wm,), k * dt, device=z_all_norm.device, dtype=z_all_norm.dtype)
                if cfg_scale != 1.0:
                    v_cond = trainer.world.dynamics_flow.compute_velocity(z_all_norm, t_k, start_states_norm[:, :ctx], act_emb_all, drop_ctx=False)
                    v_uncond = trainer.world.dynamics_flow.compute_velocity(z_all_norm, t_k, start_states_norm[:, :ctx], act_emb_all, drop_ctx=True)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = trainer.world.dynamics_flow.compute_velocity(z_all_norm, t_k, start_states_norm[:, :ctx], act_emb_all)
                v_all_list.append(v)
                z_all_norm = z_all_norm + dt * v

            dist_to_target = (z_all_norm - z_1_all_norm).pow(2).sum(dim=-1).sqrt().mean().item()
            v_target_norm = z_1_all_norm - (trainer.world.latent_rms.normalize(z_0_all) if std_l else z_0_all)
            v_pred_t0 = v_all_list[0]
            cos_sim = torch.nn.functional.cosine_similarity(v_pred_t0, v_target_norm, dim=-1).mean().item()

        return {
            'sequences': sequences,
            'dist_to_target': dist_to_target,
            'cos_sim': cos_sim,
        }

    def plot_flow_trajectories(self, flow_data, save_dir='./diagnostics_output'):
        """Generate PCA plots showing predicted integration vs OT straight lines."""
        os.makedirs(save_dir, exist_ok=True)
        
        num_seqs = len(flow_data['sequences'])
        if num_seqs == 0:
            return None

        fig, axes = plt.subplots(num_seqs, 2, figsize=(14, 5 * num_seqs))
        if num_seqs == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, seq_data in enumerate(flow_data['sequences']):
            # Panel 1: Flow tree (all future steps)
            ax = axes[i, 0]
            ctx = seq_data['ctx_proj']
            ax.plot(ctx[:, 0], ctx[:, 1], 'g-o', markersize=5, label='Context', alpha=0.8)
            ax.plot(ctx[-1, 0], ctx[-1, 1], 'bo', markersize=7, label='t=0 Start')
            
            # Context transition arrows (chronological sequence)
            for j in range(len(ctx) - 1):
                ax.annotate('', xy=(ctx[j+1, 0], ctx[j+1, 1]), xytext=(ctx[j, 0], ctx[j, 1]),
                            arrowprops=dict(arrowstyle="-|>", color='green', lw=1.2, alpha=0.5, mutation_scale=8))
            
            targets = seq_data['target_proj']
            ax.scatter(targets[:, 0], targets[:, 1], color='red', marker='*', s=80, label='Targets', zorder=5)
            
            # Chronological target transitions (ctx[-1] -> targets[0] -> targets[1] -> ...)
            if len(targets) > 0:
                ax.annotate('', xy=(targets[0, 0], targets[0, 1]), xytext=(ctx[-1, 0], ctx[-1, 1]),
                            arrowprops=dict(arrowstyle="-|>", color='red', linestyle=':', lw=1.2, alpha=0.4, mutation_scale=8))
                for h in range(len(targets) - 1):
                    ax.annotate('', xy=(targets[h+1, 0], targets[h+1, 1]), xytext=(targets[h, 0], targets[h, 1]),
                                arrowprops=dict(arrowstyle="-|>", color='red', linestyle=':', lw=1.2, alpha=0.4, mutation_scale=8))
            
            traj = seq_data['traj_proj']
            ot = seq_data['ot_proj']
            
            H = targets.shape[0]
            for h in range(H):
                # OT flow matching transition path (ground truth straight line)
                ax.plot(ot[:, h, 0], ot[:, h, 1], 'k--', alpha=0.15)
                if ot.shape[0] > 1:
                    ax.annotate('', xy=(ot[-1, h, 0], ot[-1, h, 1]), xytext=(ot[-2, h, 0], ot[-2, h, 1]),
                                arrowprops=dict(arrowstyle="-|>", color='black', alpha=0.25, lw=1.0, mutation_scale=8))
                
                # Model predicted flow matching trajectory
                ax.plot(traj[:, h, 0], traj[:, h, 1], 'b-', alpha=0.5)
                if traj.shape[0] > 1:
                    ax.annotate('', xy=(traj[-1, h, 0], traj[-1, h, 1]), xytext=(traj[-2, h, 0], traj[-2, h, 1]),
                                arrowprops=dict(arrowstyle="-|>", color='blue', alpha=0.4, lw=1.2, mutation_scale=8))
                
            ax.set_xlabel('PCA Dim 1')
            ax.set_ylabel('PCA Dim 2')
            ax.set_title(f'Seq {i} - Future Flow Tree (H={H})')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
                
            # Panel 2: Detail zoom of the final step (h = H-1) + velocity arrows
            ax_detail = axes[i, 1]
            ax_detail.plot(ctx[:, 0], ctx[:, 1], 'g-o', markersize=5, alpha=0.5)
            ax_detail.plot(ctx[-1, 0], ctx[-1, 1], 'bo', markersize=7, label='t=0 Start')
            
            # Context transition arrows in detail view
            for j in range(len(ctx) - 1):
                ax_detail.annotate('', xy=(ctx[j+1, 0], ctx[j+1, 1]), xytext=(ctx[j, 0], ctx[j, 1]),
                                   arrowprops=dict(arrowstyle="-|>", color='green', lw=1.0, alpha=0.3, mutation_scale=8))
            
            h_det = H - 1
            ax_detail.scatter(targets[h_det, 0], targets[h_det, 1], color='red', marker='*', s=120, label='Target', zorder=5)
            ax_detail.plot(ot[:, h_det, 0], ot[:, h_det, 1], 'k--', alpha=0.3, label='OT Path')
            
            # OT arrow at the end pointing to target state
            if ot.shape[0] > 1:
                ax_detail.annotate('', xy=(ot[-1, h_det, 0], ot[-1, h_det, 1]), xytext=(ot[-2, h_det, 0], ot[-2, h_det, 1]),
                                   arrowprops=dict(arrowstyle="-|>", color='black', alpha=0.4, lw=1.2, mutation_scale=10))
            
            ax_detail.plot(traj[:, h_det, 0], traj[:, h_det, 1], 'b-', linewidth=2, label='Pred Traj')
            ax_detail.scatter(traj[:, h_det, 0], traj[:, h_det, 1], color='blue', s=20, alpha=0.6)
            
            vel = seq_data['vel_proj']
            K = vel.shape[0]
            dt = 1.0 / K
            
            for k in range(K):
                ax_detail.quiver(
                    traj[k, h_det, 0], traj[k, h_det, 1],
                    vel[k, h_det, 0] * dt, vel[k, h_det, 1] * dt,
                    angles='xy', scale_units='xy', scale=1.0,
                    color='royalblue', alpha=0.8, width=0.007
                )
                
            ax_detail.set_xlabel('PCA Dim 1')
            ax_detail.set_ylabel('PCA Dim 2')
            ax_detail.set_title(f'Seq {i} - Zoom Final Step (h={h_det}) & Velocity Field')
            ax_detail.grid(True, alpha=0.3)
            if i == 0:
                ax_detail.legend()
                
        fig.suptitle('Flow Matching Latent Space Trajectories', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        fig.savefig(os.path.join(save_dir, 'flow_trajectories.png'), dpi=150)
        return fig
