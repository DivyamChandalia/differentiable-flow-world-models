"""
Action Distribution Monitor for the Actor network.

Tracks action distribution health to detect:
- Bang-bang collapse (tanh saturation)
- Entropy collapse
- Mean/std drift across the imagination horizon
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class ActionDistributionMonitor:
    """Monitor actor's action distributions for pathological behavior."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device

    @torch.no_grad()
    def collect_action_stats(self, start_states, real_actions):
        """
        Run imagination and collect detailed action statistics at each step.
        
        Returns dict with arrays of shape (horizon,) for:
        - pre_tanh_mean_abs: mean |x| before tanh (>2 = saturation risk)
        - pre_tanh_std: std of pre-tanh activations
        - post_tanh_mean_abs: mean |tanh(x)| (near 1.0 = bang-bang)
        - action_std: mean std from the actor distribution
        - analytical_entropy: base distribution entropy
        - effective_entropy: entropy estimated from samples
        - action_means: mean action values per horizon step
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

        batch_size = min(cfg.batch_size, B_wm)
        batch_indices = torch.randint(0, B_wm, (batch_size,))
        time_indices = torch.randint(0, max_start + 1, (batch_size,))

        ctx_windows = torch.stack([
            start_states[b, t:t + ctx]
            for b, t in zip(batch_indices.tolist(), time_indices.tolist())
        ], dim=0)

        trainer.world.reset_cache()

        with torch.amp.autocast(device_type=self.device.type, enabled=trainer.use_amp, dtype=trainer.amp_dtype):
            history_list = list(ctx_windows.unbind(dim=1))
            while len(history_list) < cfg.imagination_ctx_frames:
                history_list.insert(0, history_list[0])
            history_list = history_list[-cfg.imagination_ctx_frames:]

            actor_input = torch.stack(history_list, dim=1).reshape(batch_size, -1)

            # Get the raw distribution for detailed stats
            dist = trainer.actor.get_distribution(actor_input)
            
            # dist.loc and dist.scale have shape (batch_size, chunk_size, action_dim)
            raw_mean = dist.loc    # pre-tanh mean
            raw_std = dist.scale   # pre-tanh std

            # Sample pre-tanh activations
            x = dist.sample()  # (batch_size, chunk_size, action_dim)
            y = torch.tanh(x)  # post-tanh

            # Analytical entropy of base Normal per step
            analytical_ent = dist.entropy()  # (batch_size, chunk_size, action_dim)

        H = raw_mean.shape[1]  # chunk_size
        
        stats = {
            'pre_tanh_mean_abs': raw_mean.float().abs().mean(dim=(0, 2)).cpu().numpy(),
            'pre_tanh_std': raw_std.float().mean(dim=(0, 2)).cpu().numpy(),
            'post_tanh_mean_abs': y.float().abs().mean(dim=(0, 2)).cpu().numpy(),
            'action_std': raw_std.float().mean(dim=(0, 2)).cpu().numpy(),
            'analytical_entropy': analytical_ent.float().sum(dim=-1).mean(dim=0).cpu().numpy(),
            'action_means_per_dim': raw_mean.float().mean(dim=0).cpu().numpy(),  # (H, action_dim)
            'action_stds_per_dim': raw_std.float().mean(dim=0).cpu().numpy(),    # (H, action_dim)
            'pre_tanh_samples': x.float().cpu().numpy(),  # for histogram
        }
        return stats

    def plot_action_stats(self, stats, save_dir='./diagnostics_output'):
        """Generate all action distribution diagnostic plots."""
        os.makedirs(save_dir, exist_ok=True)

        H = len(stats['pre_tanh_mean_abs'])
        steps = np.arange(H)

        # --- Plot 1: Saturation indicators ---
        fig_health, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(steps, stats['pre_tanh_mean_abs'], 'b-o', markersize=4)
        ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Saturation threshold')
        ax.set_xlabel('Imagination Step')
        ax.set_ylabel('Mean |pre-tanh activation|')
        ax.set_title('Pre-Tanh Activation Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(steps, stats['post_tanh_mean_abs'], 'r-o', markersize=4)
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Bang-bang threshold')
        ax.set_xlabel('Imagination Step')
        ax.set_ylabel('Mean |post-tanh activation|')
        ax.set_title('Post-Tanh Activation Magnitude (closer to 1 = bang-bang)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(steps, stats['pre_tanh_std'], 'g-o', markersize=4)
        ax.set_xlabel('Imagination Step')
        ax.set_ylabel('Mean σ (pre-tanh)')
        ax.set_title('Action Distribution Std Over Horizon')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(steps, stats['analytical_entropy'], 'm-o', markersize=4)
        ax.set_xlabel('Imagination Step')
        ax.set_ylabel('Entropy (summed over action dims)')
        ax.set_title('Analytical Entropy Over Horizon')
        ax.grid(True, alpha=0.3)

        fig_health.suptitle('Action Distribution Health', fontsize=14, fontweight='bold')
        fig_health.tight_layout()
        fig_health.savefig(os.path.join(save_dir, 'action_distribution_health.png'), dpi=150)

        # --- Plot 2: Pre-tanh activation histogram ---
        fig_hist, ax = plt.subplots(1, 1, figsize=(10, 5))
        samples_flat = stats['pre_tanh_samples'].flatten()
        ax.hist(samples_flat, bins=100, density=True, alpha=0.7, color='steelblue')
        ax.axvline(-2.0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='±2.0 saturation zone')
        ax.set_xlabel('Pre-tanh activation value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Pre-Tanh Activations (all steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(save_dir, 'pre_tanh_histogram.png'), dpi=150)

        # --- Plot 3: Per-dimension action means across horizon ---
        action_means = stats['action_means_per_dim']  # (H, action_dim)
        action_stds = stats['action_stds_per_dim']
        n_actions = action_means.shape[1]

        fig_profile, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 4), squeeze=False)
        for d in range(n_actions):
            ax = axes[0, d]
            ax.plot(steps, action_means[:, d], 'b-o', markersize=4, label='Mean')
            ax.fill_between(steps,
                            action_means[:, d] - action_stds[:, d],
                            action_means[:, d] + action_stds[:, d],
                            alpha=0.2, color='blue')
            ax.set_xlabel('Imagination Step')
            ax.set_ylabel(f'Action dim {d}')
            ax.set_title(f'Action dim {d} across horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig_profile.suptitle('Per-Dimension Action Profile', fontsize=14, fontweight='bold')
        fig_profile.tight_layout()
        fig_profile.savefig(os.path.join(save_dir, 'action_profile_per_dim.png'), dpi=150)

        print(f"Action distribution plots saved to {save_dir}")
        return fig_health, fig_hist, fig_profile

