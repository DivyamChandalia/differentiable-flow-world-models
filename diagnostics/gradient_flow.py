"""
Gradient Flow Analyzer for the Actor network.

Analyzes gradient health through the imagination chain:
- Per-layer gradient norms
- Gradient cosine similarity between consecutive steps
- Jacobian spectral norm estimation through the WM imagination chain
- Finite-difference vs analytical gradient comparison
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class GradientFlowAnalyzer:
    """Analyze gradient flow through the actor and imagination pipeline."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.device = trainer.device
        self._prev_grad = None

    # ------------------------------------------------------------------
    # Per-layer gradient norms
    # ------------------------------------------------------------------

    def per_layer_grad_norms(self):
        """
        Return a dict mapping parameter name → gradient L2 norm.
        Call AFTER a backward pass.
        """
        norms = {}
        for name, p in self.trainer.actor.named_parameters():
            if p.grad is not None:
                norms[name] = p.grad.norm().item()
            else:
                norms[name] = 0.0
        return norms

    def plot_grad_norms(self, norms, save_dir='./diagnostics_output'):
        """Bar chart of per-layer gradient norms."""
        os.makedirs(save_dir, exist_ok=True)

        names = list(norms.keys())
        values = list(norms.values())

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        colors = ['#e74c3c' if v < 1e-6 else '#2ecc71' if v < 1.0 else '#f39c12' for v in values]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Gradient L2 Norm')
        ax.set_title('Actor Per-Layer Gradient Norms')
        ax.set_xscale('log')
        ax.axvline(1e-6, color='red', linestyle=':', alpha=0.5, label='Vanishing threshold')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'actor_grad_norms.png'), dpi=150)
        plt.close(fig)
        print(f"Gradient norms plot saved to {save_dir}")

    # ------------------------------------------------------------------
    # Gradient cosine similarity
    # ------------------------------------------------------------------

    def gradient_cosine_similarity(self):
        """
        Compute cosine similarity between the current gradient and the
        previous one. Returns None on first call. High values = stable
        gradient direction; low/negative = thrashing.
        """
        curr_grad = torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=self.device)
            for p in self.trainer.actor.parameters()
        ])

        if self._prev_grad is None:
            self._prev_grad = curr_grad.clone()
            return None

        cos_sim = torch.nn.functional.cosine_similarity(
            self._prev_grad.unsqueeze(0), curr_grad.unsqueeze(0)
        ).item()

        self._prev_grad = curr_grad.clone()
        return cos_sim

    # ------------------------------------------------------------------
    # Finite-difference vs analytical gradient comparison
    # ------------------------------------------------------------------

    def finite_difference_check(self, start_states, real_actions, epsilon=1e-4, num_params=20):
        """
        Compare analytical gradients to finite-difference approximations
        for a random subset of actor parameters. Large discrepancies
        indicate the WM is distorting the gradient signal.
        
        Returns:
            dict with 'analytical', 'numerical', 'cosine_sim', 'relative_error'
        """
        from .landscape import LossLandscapeVisualizer
        viz = LossLandscapeVisualizer(self.trainer)
        
        actor = self.trainer.actor
        world = self.trainer.world
        value_model = self.trainer.value_model
        target_value_model = self.trainer.target_value_model
        actor_loss_fn = self.trainer.actor_loss_fn
        value_loss_fn = self.trainer.value_loss_fn
        return_ema = self.trainer.return_ema

        orig_device = self.device
        cpu_device = torch.device('cpu')

        with torch.no_grad():
            original_params = viz._get_params_as_vector(actor).clone()

        # Temporarily set devices to CPU
        self.device = cpu_device
        self.trainer.device = cpu_device
        viz.device = cpu_device

        # Move models to CPU and cast to double precision (float64)
        actor.to(cpu_device).double()
        world.to(cpu_device).double()
        value_model.to(cpu_device).double()
        target_value_model.to(cpu_device).double()
        actor_loss_fn.to(cpu_device).double()
        value_loss_fn.to(cpu_device).double()
        if return_ema is not None:
            return_ema.to(cpu_device).double()

        # Convert input tensors to double precision on CPU
        states_db = start_states.detach().cpu().double()
        actions_db = real_actions.detach().cpu().double()

        # Compute analytical gradient in double precision
        viz._run_actor_forward_with_grad(states_db, actions_db)
        analytical_grad = torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=cpu_device, dtype=torch.float64)
            for p in actor.parameters()
        ]).clone()

        # Sample random parameter indices
        total_params = original_params.numel()
        indices = torch.randperm(total_params)[:num_params]

        analytical_vals = analytical_grad[indices].cpu().numpy()
        numerical_vals = np.zeros(num_params)

        # FD perturbation loop — no grad needed, in double precision
        with torch.no_grad():
            # Flat param vector in double
            params_db = viz._get_params_as_vector(actor).clone()
            for k, idx in enumerate(indices):
                # f(θ + ε*e_i)
                perturbed_plus = params_db.clone()
                perturbed_plus[idx] += epsilon
                viz._set_params_from_vector(actor, perturbed_plus)
                loss_plus, _, _ = viz._evaluate_actor_loss(states_db, actions_db)

                # f(θ - ε*e_i)
                perturbed_minus = params_db.clone()
                perturbed_minus[idx] -= epsilon
                viz._set_params_from_vector(actor, perturbed_minus)
                loss_minus, _, _ = viz._evaluate_actor_loss(states_db, actions_db)

                numerical_vals[k] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original parameters to actor
            viz._set_params_from_vector(actor, original_params)

        # Move models back to original device and restore float32
        actor.to(orig_device).float()
        world.to(orig_device).float()
        value_model.to(orig_device).float()
        target_value_model.to(orig_device).float()
        actor_loss_fn.to(orig_device).float()
        value_loss_fn.to(orig_device).float()
        if return_ema is not None:
            return_ema.to(orig_device).float()

        # Restore devices
        self.device = orig_device
        self.trainer.device = orig_device
        viz.device = orig_device

        # Compute agreement metrics
        a_tensor = torch.tensor(analytical_vals)
        n_tensor = torch.tensor(numerical_vals)

        cos_sim = torch.nn.functional.cosine_similarity(
            a_tensor.unsqueeze(0), n_tensor.unsqueeze(0)
        ).item()

        denom = np.maximum(np.abs(analytical_vals) + np.abs(numerical_vals), 1e-15)
        relative_error = np.mean(np.abs(analytical_vals - numerical_vals) / denom)

        return {
            'analytical': analytical_vals,
            'numerical': numerical_vals,
            'cosine_similarity': cos_sim,
            'mean_relative_error': relative_error,
        }

    def plot_fd_comparison(self, fd_data, save_dir='./diagnostics_output'):
        """Scatter plot comparing analytical vs finite-difference gradients."""
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(fd_data['analytical'], fd_data['numerical'], alpha=0.7, s=60)

        # Perfect agreement line
        lims = [
            min(fd_data['analytical'].min(), fd_data['numerical'].min()),
            max(fd_data['analytical'].max(), fd_data['numerical'].max()),
        ]
        margin = (lims[1] - lims[0]) * 0.1
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect agreement')

        ax.set_xlabel('Analytical Gradient')
        ax.set_ylabel('Finite Difference Gradient')
        ax.set_title(
            f'Gradient Check: cos_sim={fd_data["cosine_similarity"]:.4f}, '
            f'rel_err={fd_data["mean_relative_error"]:.4e}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'finite_diff_check.png'), dpi=150)
        plt.close(fig)
        print(f"Finite difference comparison saved to {save_dir}")
