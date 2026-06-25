import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ARDiagnostics:
    """Diagnostic tool to analyze action sensitivity / steering impact on the Autoregressive dynamics model."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device

    def fit_pca(self, states, q=2):
        """Fit a linear PCA projection onto 2D using PyTorch SVD."""
        states = states.float()
        mean = states.mean(dim=0, keepdim=True)
        states_centered = states - mean
        U, S, V = torch.pca_lowrank(states_centered, q=q)
        return mean, V

    @torch.no_grad()
    def collect_ar_data(self, start_states, real_actions):
        """
        Run open-loop rollouts for the Autoregressive Dynamics model under different actions.
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
            pad_a = real_actions[:, :1].expand(B_wm, ctx - T, A) * 0.0
            real_actions = torch.cat([pad_a, real_actions], dim=1)
            max_start = 0

        # We visualize the first few sequences in detail
        num_viz = min(3, B_wm)
        ctx_windows = start_states[:num_viz, :ctx]
        future_actions = real_actions[:num_viz, ctx-1:]
        future_targets = start_states[:num_viz, ctx:]

        H = future_targets.shape[1]
        AS = future_actions.shape[-1]

        # Define alternative actions
        actions_true = future_actions
        actions_zero = torch.zeros_like(future_actions)
        actions_rand = torch.rand_like(future_actions) * 2.0 - 1.0  # Assumes normalized action space [-1, 1]
        actions_opposite = -future_actions

        with torch.amp.autocast(device_type=self.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype):
            emb_true = trainer.world.action(actions_true.reshape(-1, AS)).reshape(num_viz, H, -1)
            emb_zero = trainer.world.action(actions_zero.reshape(-1, AS)).reshape(num_viz, H, -1)
            emb_rand = trainer.world.action(actions_rand.reshape(-1, AS)).reshape(num_viz, H, -1)
            emb_opposite = trainer.world.action(actions_opposite.reshape(-1, AS)).reshape(num_viz, H, -1)

            ctx_actions = real_actions[:num_viz, :ctx-1]
            ctx_emb = trainer.world.action(ctx_actions.reshape(-1, AS)).reshape(num_viz, ctx-1, -1)

            def ar_rollout(action_embeddings):
                curr_seq = ctx_windows.clone()
                rollout_states = []
                for i in range(H):
                    act_seq = torch.cat([ctx_emb, action_embeddings[:, :i+1]], dim=1)
                    pred = trainer.world.dynamics_ar(curr_seq, act_seq)
                    next_state = pred[:, -1:]
                    rollout_states.append(next_state)
                    curr_seq = torch.cat([curr_seq, next_state], dim=1)
                return torch.cat(rollout_states, dim=1)

            traj_true = ar_rollout(emb_true)
            traj_zero = ar_rollout(emb_zero)
            traj_rand = ar_rollout(emb_rand)
            traj_opposite = ar_rollout(emb_opposite)

        # Project via PCA for each visualized sequence
        sequences = []
        for i in range(num_viz):
            pts = [
                ctx_windows[i],
                future_targets[i],
                traj_true[i],
                traj_zero[i],
                traj_rand[i],
                traj_opposite[i]
            ]
            pts_tensor = torch.cat(pts, dim=0)
            mean, V = self.fit_pca(pts_tensor, q=2)

            sequences.append({
                'ctx_proj': (ctx_windows[i].float() - mean) @ V,
                'target_proj': (future_targets[i].float() - mean) @ V,
                'traj_true_proj': (traj_true[i].float() - mean) @ V,
                'traj_zero_proj': (traj_zero[i].float() - mean) @ V,
                'traj_rand_proj': (traj_rand[i].float() - mean) @ V,
                'traj_opposite_proj': (traj_opposite[i].float() - mean) @ V,
            })

        # Calculate batch-wide metrics
        with torch.amp.autocast(device_type=self.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype):
            full_ctx = start_states[:, :ctx]
            full_actions = real_actions[:, ctx-1:]
            full_targets = start_states[:, ctx:]

            full_emb_true = trainer.world.action(full_actions.reshape(-1, AS)).reshape(B_wm, H, -1)
            full_emb_zero = trainer.world.action(torch.zeros_like(full_actions).reshape(-1, AS)).reshape(B_wm, H, -1)
            full_emb_rand = trainer.world.action((torch.rand_like(full_actions) * 2.0 - 1.0).reshape(-1, AS)).reshape(B_wm, H, -1)
            full_emb_opp = trainer.world.action((-full_actions).reshape(-1, AS)).reshape(B_wm, H, -1)

            full_ctx_actions = real_actions[:, :ctx-1]
            full_ctx_emb = trainer.world.action(full_ctx_actions.reshape(-1, AS)).reshape(B_wm, ctx-1, -1)

            def full_ar_rollout(action_embeddings):
                curr_seq = full_ctx.clone()
                rollout_states = []
                for i in range(H):
                    act_seq = torch.cat([full_ctx_emb, action_embeddings[:, :i+1]], dim=1)
                    pred = trainer.world.dynamics_ar(curr_seq, act_seq)
                    next_state = pred[:, -1:]
                    rollout_states.append(next_state)
                    curr_seq = torch.cat([curr_seq, next_state], dim=1)
                return torch.cat(rollout_states, dim=1)

            full_traj_true = full_ar_rollout(full_emb_true)
            full_traj_zero = full_ar_rollout(full_emb_zero)
            full_traj_rand = full_ar_rollout(full_emb_rand)
            full_traj_opp = full_ar_rollout(full_emb_opp)

            # Step-by-step MSE compared to True actions trajectory
            mse_zero = (full_traj_true - full_traj_zero).pow(2).mean(dim=-1).mean(dim=0).cpu().numpy()
            mse_rand = (full_traj_true - full_traj_rand).pow(2).mean(dim=-1).mean(dim=0).cpu().numpy()
            mse_opp = (full_traj_true - full_traj_opp).pow(2).mean(dim=-1).mean(dim=0).cpu().numpy()

            dist_to_target = (full_traj_true - full_targets).pow(2).mean().sqrt().item()

        return {
            'sequences': sequences,
            'mse_zero': mse_zero,
            'mse_rand': mse_rand,
            'mse_opp': mse_opp,
            'dist_to_target': dist_to_target,
        }

    def plot_ar_trajectories(self, ar_data, save_dir='./diagnostics_output'):
        """Plot projected trajectories under different action sequences and step-wise MSE impact."""
        os.makedirs(save_dir, exist_ok=True)
        num_seqs = len(ar_data['sequences'])
        if num_seqs == 0:
            return None

        fig, axes = plt.subplots(num_seqs, 2, figsize=(14, 5 * num_seqs))
        if num_seqs == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, seq_data in enumerate(ar_data['sequences']):
            # Panel 1: PCA Trajectories
            ax = axes[i, 0]
            ctx = seq_data['ctx_proj'].cpu().numpy()
            targets = seq_data['target_proj'].cpu().numpy()
            traj_true = seq_data['traj_true_proj'].cpu().numpy()
            traj_zero = seq_data['traj_zero_proj'].cpu().numpy()
            traj_rand = seq_data['traj_rand_proj'].cpu().numpy()
            traj_opp = seq_data['traj_opposite_proj'].cpu().numpy()

            # Context
            ax.plot(ctx[:, 0], ctx[:, 1], 'g-o', markersize=5, label='Context', alpha=0.8)
            ax.plot(ctx[-1, 0], ctx[-1, 1], 'bo', markersize=7, label='t=0 Start')

            H = targets.shape[0]

            # True, Zero, Rand, Opp
            ax.plot(traj_true[:, 0], traj_true[:, 1], 'b-o', markersize=4, label='True Actions', alpha=0.9)
            ax.plot(traj_zero[:, 0], traj_zero[:, 1], 'gray', linestyle='--', marker='x', markersize=4, label='Zero Actions', alpha=0.7)
            ax.plot(traj_rand[:, 0], traj_rand[:, 1], 'orange', linestyle=':', marker='^', markersize=4, label='Random Actions', alpha=0.7)
            ax.plot(traj_opp[:, 0], traj_opp[:, 1], 'purple', linestyle='-.', marker='v', markersize=4, label='Opposite Actions', alpha=0.7)

            # Targets
            ax.scatter(targets[:, 0], targets[:, 1], color='red', marker='*', s=100, label='Targets', zorder=5)

            ax.set_xlabel('PCA Dim 1')
            ax.set_ylabel('PCA Dim 2')
            ax.set_title(f'Seq {i} - Autoregressive Action Steering (H={H})')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

            # Panel 2: Step-by-step Action Impact (MSE)
            ax_impact = axes[i, 1]
            steps = np.arange(1, H + 1)
            ax_impact.plot(steps, ar_data['mse_zero'], 'gray', linestyle='--', marker='x', label='Zero vs True', linewidth=2)
            ax_impact.plot(steps, ar_data['mse_rand'], 'orange', linestyle=':', marker='^', label='Random vs True', linewidth=2)
            ax_impact.plot(steps, ar_data['mse_opp'], 'purple', linestyle='-.', marker='v', label='Opposite vs True', linewidth=2)

            ax_impact.set_xlabel('Imagination Horizon Steps')
            ax_impact.set_ylabel('Latent MSE relative to True')
            ax_impact.set_title(f'Action Impact Sensitivity (MSE)')
            ax_impact.set_xticks(steps)
            ax_impact.grid(True, alpha=0.3)
            if i == 0:
                ax_impact.legend()

        fig.suptitle('Autoregressive Action Steering Diagnostics', fontsize=16, fontweight='bold')
        fig.tight_layout()

        fig.savefig(os.path.join(save_dir, 'ar_action_impact.png'), dpi=150)
        return fig
