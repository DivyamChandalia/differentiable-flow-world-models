"""
Loss Landscape Visualizer for the Actor network.

Implements filter-normalized random direction perturbation (Li et al., 2018)
to produce 2D contour plots of the actor loss surface and 1D slices along
the gradient direction.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import os


class LossLandscapeVisualizer:
    """
    Visualize the actor loss landscape by perturbing actor parameters
    along two random orthogonal directions in parameter space.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device

    # ------------------------------------------------------------------
    # Direction utilities (filter-normalized, Li et al. 2018)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_params_as_vector(model):
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.flatten() for p in model.parameters()])

    @staticmethod
    def _set_params_from_vector(model, vec):
        """Load a flat parameter vector back into the model."""
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(vec[offset:offset + numel].view_as(p))
            offset += numel

    @staticmethod
    def _random_direction_filter_normalized(model):
        """
        Create a random direction with filter-wise normalization.
        For each parameter tensor, generate a random tensor of the same shape,
        then normalize it so that its norm matches the parameter's norm.
        This prevents large parameters from dominating the landscape.
        """
        direction = []
        for p in model.parameters():
            d = torch.randn_like(p)
            # Filter-wise normalization: scale random direction to match param norm
            if p.dim() >= 2:
                # For weight matrices: normalize per filter (output unit)
                for i in range(p.shape[0]):
                    p_norm = p[i].norm()
                    d_norm = d[i].norm()
                    if d_norm > 1e-10:
                        d[i] = d[i] * (p_norm / d_norm)
            else:
                # For biases: simple norm matching
                p_norm = p.norm()
                d_norm = d.norm()
                if d_norm > 1e-10:
                    d = d * (p_norm / d_norm)
            direction.append(d.flatten())
        return torch.cat(direction)

    @staticmethod
    def _orthogonalize(d1, d2):
        """Make d2 orthogonal to d1 via Gram-Schmidt."""
        proj = (d2 @ d1) / (d1 @ d1 + 1e-10)
        d2_orth = d2 - proj * d1
        return d2_orth

    # ------------------------------------------------------------------
    # Loss evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_actor_loss(self, start_states, real_actions):
        """
        Run the imagination pipeline and compute actor loss components.
        Returns (total_loss, target_loss, entropy_loss) as floats.
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
        ctx_windows = start_states[:batch_size, 0:ctx]
        ctx_action_windows = real_actions[:batch_size, 0:ctx-1]



        trainer.world.reset_cache()
        
        # Disable AMP during diagnostics to prevent bfloat16 truncation in finite differences
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            history_list = list(ctx_windows.unbind(dim=1))
            while len(history_list) < cfg.imagination_ctx_frames:
                history_list.insert(0, history_list[0])
            history_list = history_list[-cfg.imagination_ctx_frames:]

            actor_input = torch.stack(history_list, dim=1).reshape(batch_size, -1)

            # Enable grad temporarily for the actor forward (needed for entropy)
            with torch.enable_grad():
                torch.manual_seed(42) # FIX: deterministic rsample for finite difference
                action_chunk_scaled, squashed_log_prob, analytical_entropy = trainer.actor(actor_input)

            ctx_actions_real = ctx_action_windows.to(device=self.device, dtype=action_chunk_scaled.dtype)
            actions_full = torch.cat([ctx_actions_real, action_chunk_scaled], dim=1)

            imagined_states, imagined_rewards, imagined_terminals = trainer.world.generate_chunk(
                actions=actions_full,
                start_states=ctx_windows
            )

            history_windows = trainer._get_history_windows(imagined_states, cfg.imagination_ctx_frames, context=ctx_windows)
            ema_imagined_values = trainer.target_value_model(history_windows)
            imagined_values = trainer.value_model(history_windows)

            pred_term_probs = torch.sigmoid(imagined_terminals)
            pcont = (1.0 - pred_term_probs)

            targets_a = trainer.value_loss_fn._compute_lambda_returns(
                ema_imagined_values, imagined_rewards, pcont
            )

            if cfg.use_advantage or cfg.reinforce:
                actor_targets = targets_a - imagined_values[:-1]
            else:
                actor_targets = targets_a

            if trainer.return_ema is not None:
                scale = (trainer.return_ema.high - trainer.return_ema.low).clamp(min=1e-2)
                normalized_targets = actor_targets / scale
            else:
                normalized_targets = actor_targets

            entropy_to_use = -squashed_log_prob.transpose(0, 1)
            log_prob = squashed_log_prob.transpose(0, 1)

            a_loss, a_target_loss, a_entropy_loss, _ = trainer.actor_loss_fn(
                normalized_targets, entropy_to_use, log_prob=log_prob, pcont=pcont
            )

        return a_loss.item(), a_target_loss.item(), a_entropy_loss.item()

    # ------------------------------------------------------------------
    # 2D landscape
    # ------------------------------------------------------------------

    def compute_landscape(self, start_states, real_actions, grid_size=21, range_scale=1.0):
        """
        Compute actor loss on a 2D grid of parameter perturbations.

        Returns:
            alphas, betas: 1D arrays of perturbation magnitudes
            total_loss, target_loss, entropy_loss: 2D grids (grid_size x grid_size)
            d1, d2: the two random directions used
        """
        actor = self.trainer.actor
        original_params = self._get_params_as_vector(actor).clone()

        # Generate filter-normalized orthogonal directions
        d1 = self._random_direction_filter_normalized(actor)
        d2 = self._random_direction_filter_normalized(actor)
        d2 = self._orthogonalize(d1, d2)

        # Normalize directions
        d1 = d1 / d1.norm()
        d2 = d2 / d2.norm()

        alphas = np.linspace(-range_scale, range_scale, grid_size)
        betas = np.linspace(-range_scale, range_scale, grid_size)

        total_grid = np.zeros((grid_size, grid_size))
        target_grid = np.zeros((grid_size, grid_size))
        entropy_grid = np.zeros((grid_size, grid_size))

        print(f"Computing {grid_size}x{grid_size} loss landscape...")
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                perturbed = original_params + alpha * d1 + beta * d2
                self._set_params_from_vector(actor, perturbed)

                total, target, ent = self._evaluate_actor_loss(start_states, real_actions)
                total_grid[i, j] = total
                target_grid[i, j] = target
                entropy_grid[i, j] = ent

            print(f"  Row {i + 1}/{grid_size} done")

        # Restore original parameters
        self._set_params_from_vector(actor, original_params)

        return {
            'alphas': alphas,
            'betas': betas,
            'total_loss': total_grid,
            'target_loss': target_grid,
            'entropy_loss': entropy_grid,
        }

    # ------------------------------------------------------------------
    # 1D gradient slice
    # ------------------------------------------------------------------

    def compute_gradient_slice(self, start_states, real_actions, num_points=41, range_scale=1.0):
        """
        Compute actor loss along the negative gradient direction (the direction 
        the optimizer would step). This shows whether the loss actually decreases
        when you follow the gradient.
        """
        actor = self.trainer.actor
        original_params = self._get_params_as_vector(actor).clone()

        # Compute the actual gradient at the current point
        actor.zero_grad()
        start_states_grad = start_states.detach()
        real_actions_grad = real_actions.detach()

        # We need to do a forward pass WITH gradients
        self._run_actor_forward_with_grad(start_states_grad, real_actions_grad)

        grad_vec = torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=self.device)
            for p in actor.parameters()
        ])

        grad_norm = grad_vec.norm()
        if grad_norm < 1e-10:
            print("WARNING: Gradient is near-zero, cannot compute gradient slice")
            return None

        grad_direction = grad_vec / grad_norm  # Unit vector in gradient direction

        alphas = np.linspace(-range_scale, range_scale, num_points)
        losses = np.zeros(num_points)

        print(f"Computing 1D gradient slice ({num_points} points)...")
        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                # Move along negative gradient (descent direction)
                perturbed = original_params - alpha * grad_direction
                self._set_params_from_vector(actor, perturbed)
                total, _, _ = self._evaluate_actor_loss(start_states, real_actions)
                losses[i] = total

        self._set_params_from_vector(actor, original_params)

        return {
            'alphas': alphas,
            'losses': losses,
            'grad_norm': grad_norm.item(),
        }

    def _run_actor_forward_with_grad(self, start_states, real_actions):
        """Run full imagination pipeline with gradients to get actor gradient."""
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
        ctx_windows = start_states[:batch_size, 0:ctx]
        ctx_action_windows = real_actions[:batch_size, 0:ctx-1]



        trainer.world.reset_cache()
        trainer.world.freeze()

        # Disable AMP during diagnostics to prevent bfloat16 truncation in finite differences
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            history_list = list(ctx_windows.unbind(dim=1))
            while len(history_list) < cfg.imagination_ctx_frames:
                history_list.insert(0, history_list[0])
            history_list = history_list[-cfg.imagination_ctx_frames:]

            actor_input = torch.stack(history_list, dim=1).reshape(batch_size, -1)
            
            torch.manual_seed(42) # FIX: deterministic rsample to match FD
            action_chunk_scaled, squashed_log_prob, analytical_entropy = trainer.actor(actor_input)

            ctx_actions_real = ctx_action_windows.to(device=self.device, dtype=action_chunk_scaled.dtype)
            actions_full = torch.cat([ctx_actions_real, action_chunk_scaled], dim=1)

            imagined_states, imagined_rewards, imagined_terminals = trainer.world.generate_chunk(
                actions=actions_full, start_states=ctx_windows
            )

            history_windows = trainer._get_history_windows(imagined_states, cfg.imagination_ctx_frames, context=ctx_windows)
            ema_imagined_values = trainer.target_value_model(history_windows)
            imagined_values = trainer.value_model(history_windows)

            pred_term_probs = torch.sigmoid(imagined_terminals)
            pcont = (1.0 - pred_term_probs).detach()

            targets_a = trainer.value_loss_fn._compute_lambda_returns(
                ema_imagined_values, imagined_rewards, pcont
            )

            if cfg.use_advantage or cfg.reinforce:
                actor_targets = targets_a - imagined_values[:-1]
            else:
                actor_targets = targets_a

            if trainer.return_ema is not None:
                scale = (trainer.return_ema.high - trainer.return_ema.low).clamp(min=1e-2)
                normalized_targets = actor_targets / scale
            else:
                normalized_targets = actor_targets

            entropy_to_use = -squashed_log_prob.transpose(0, 1)
            log_prob = squashed_log_prob.transpose(0, 1)

            a_loss, _, _, _ = trainer.actor_loss_fn(
                normalized_targets, entropy_to_use, log_prob=log_prob, pcont=pcont
            )

        # Backward to populate gradients
        trainer.actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_landscape(self, landscape_data, save_dir='./diagnostics_output'):
        """Generate and save 2D contour plots and 3D surface plots."""
        os.makedirs(save_dir, exist_ok=True)

        alphas = landscape_data['alphas']
        betas = landscape_data['betas']
        A, B = np.meshgrid(alphas, betas, indexing='ij')

        for name, grid in [('total_loss', landscape_data['total_loss']),
                           ('target_loss', landscape_data['target_loss']),
                           ('entropy_loss', landscape_data['entropy_loss'])]:
            # 2D contour
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            cf = ax.contourf(A, B, grid, levels=30, cmap='RdYlBu_r')
            ax.contour(A, B, grid, levels=30, colors='k', linewidths=0.3, alpha=0.5)
            ax.plot(0, 0, 'r*', markersize=15, label='Current params')
            plt.colorbar(cf, ax=ax, label='Loss')
            ax.set_xlabel('Direction 1 (α)')
            ax.set_ylabel('Direction 2 (β)')
            ax.set_title(f'Actor {name.replace("_", " ").title()} Landscape')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'landscape_{name}.png'), dpi=150)
            plt.close(fig)

            # 3D surface
            fig = plt.figure(figsize=(10, 8))
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.plot_surface(A, B, grid, cmap='RdYlBu_r', alpha=0.8, edgecolor='k', linewidth=0.1)
            ax3d.scatter([0], [0], [grid[len(alphas)//2, len(betas)//2]],
                         color='red', s=100, zorder=5, label='Current params')
            ax3d.set_xlabel('Direction 1 (α)')
            ax3d.set_ylabel('Direction 2 (β)')
            ax3d.set_zlabel('Loss')
            ax3d.set_title(f'Actor {name.replace("_", " ").title()} Surface')
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'surface_{name}.png'), dpi=150)
            plt.close(fig)

        print(f"Landscape plots saved to {save_dir}")

    def plot_gradient_slice(self, slice_data, save_dir='./diagnostics_output'):
        """Plot the 1D loss curve along the gradient direction."""
        if slice_data is None:
            print("No gradient slice data to plot.")
            return

        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(slice_data['alphas'], slice_data['losses'], 'b-', linewidth=2)
        ax.axvline(0, color='r', linestyle='--', label='Current params')
        ax.set_xlabel('Step size along negative gradient')
        ax.set_ylabel('Actor Loss')
        ax.set_title(f'Loss Along Gradient Direction (‖∇L‖ = {slice_data["grad_norm"]:.4e})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'gradient_slice.png'), dpi=150)
        plt.close(fig)
        print(f"Gradient slice plot saved to {save_dir}")
