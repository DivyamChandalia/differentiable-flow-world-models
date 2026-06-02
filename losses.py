import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import symlog, symexp

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean() # average over projections and time

class WeakSIGReg(nn.Module):
    """
    Weak Sketch Isotropic Gaussian Regularizer.
    Forces Covariance(z) ~ Identity by minimising ||Cov(z) - I||_F.
    Matches only the 2nd moment (spherical cloud) — cheaper than the full
    Epps-Pulley characteristic-function test used by SIGReg.

    Input convention: (T, B, D)  — same as SIGReg — so it is a drop-in
    replacement in the training loop.
    """

    def __init__(self, sketch_dim: int = 64):
        super().__init__()
        self.sketch_dim = sketch_dim

    def forward(self, proj):
        """
        proj: (T, B, D)
        Returns a scalar loss.
        """
        T, B, D = proj.shape
        x = proj.reshape(T * B, D).float()   # (N, D)
        N, C = x.shape

        # Optional random sketch for large C
        sketch_dim = self.sketch_dim
        if C > sketch_dim:
            S = torch.randn(sketch_dim, C, device=x.device) / (C ** 0.5)
            x = x @ S.T          # (N, sketch_dim)
        else:
            sketch_dim = C

        # Centre
        x = x - x.mean(dim=0, keepdim=True)

        # Sample covariance
        cov = (x.T @ x) / (N - 1 + 1e-6)   # (sketch_dim, sketch_dim)

        # Target identity
        target = torch.eye(sketch_dim, device=x.device)

        return torch.norm(cov - target, p='fro')


class ValueLoss(nn.Module):

    def __init__(self, discount=0.99, lambda_=0.99, batch_first=False, use_symlog=True):
        super().__init__()
        self.discount = discount
        self.lambda_ = lambda_
        self.batch_first = batch_first
        self.use_symlog = use_symlog

    def forward(self, values, rewards, pcont=None, target_values=None):
        """
        Calculates the Value Loss using lambda-returns.
        
        Args:
            values (torch.Tensor): Predicted values. 
                Shape: [H+1, B, 1] (if batch_first=False) or [B, H+1, 1] (if batch_first=True)
            rewards (torch.Tensor): Predicted rewards. 
                Shape: [H, B, 1] or [B, H, 1]
            pcont (torch.Tensor, optional): Continuation flags. Defaults to 1.0.
                Shape: [H, B, 1] or [B, H, 1]
            target_values (torch.Tensor, optional): Target predicted values for bootstrapping.
                                  
        Returns:
            loss (torch.Tensor): The scalar MSE loss.
            targets (torch.Tensor): The lambda-returns. Matches input format (B first or H first).
        """
        if self.batch_first:
            values = values.transpose(0, 1)
            rewards = rewards.transpose(0, 1)
            if pcont is not None:
                pcont = pcont.transpose(0, 1)
            if target_values is not None:
                target_values = target_values.transpose(0, 1)

        if pcont is None:
            pcont = torch.ones_like(rewards)

        if target_values is None:
            target_values = values

        targets = self._compute_lambda_returns(target_values, rewards, pcont)

        value_preds = values[:-1]
        
        loss = F.smooth_l1_loss(value_preds, targets.detach(), reduction='mean')

        if self.batch_first:
            targets = targets.transpose(0, 1)
        
        return loss, targets

    def _compute_lambda_returns(self, values, rewards, pcont):
        """
        Calculates lambda-returns by iterating backward through the trajectory.
        
        The formula:
        G_t = r_t + gamma * pcont_t * ((1 - lambda) * V_{t+1} + lambda * G_{t+1})
        """
        if self.use_symlog:
            raw_values = symexp(values)
            raw_rewards = symexp(rewards)
        else:
            raw_values = values
            raw_rewards = rewards
        
        H, B, _ = rewards.shape
        targets_list = []
        
        next_target = raw_values[-1] 
        
        for t in reversed(range(H)):
            target_t = raw_rewards[t] + self.discount * pcont[t] * (
                (1.0 - self.lambda_) * raw_values[t + 1] + self.lambda_ * next_target
            )
            targets_list.append(target_t)
            next_target = target_t
            
        targets_list.reverse()
        targets = torch.stack(targets_list, dim=0)
        
        if self.use_symlog:
            return symlog(targets)
        return targets
    
class ActorLoss(nn.Module):
    def __init__(self, entropy_scale=1e-4, discount=0.99, batch_first=False, reinforce=False):
        """
        Actor Loss supporting both analytical gradients (reparameterization trick)
        and REINFORCE policy gradients.
        
        Args:
            entropy_scale (float): Weight of the entropy bonus.
            discount (float): Discount factor gamma used for per-step weighting.
            batch_first (bool): If True, expects [B, H, 1], else [H, B, 1].
            reinforce (bool): If True, uses the REINFORCE gradient estimator.
        """
        super().__init__()
        self.entropy_scale = entropy_scale
        self.discount = discount
        self.batch_first = batch_first
        self.reinforce = reinforce

    def forward(self, targets, entropy, log_prob=None, pcont=None):
        """
        Args:
            targets (torch.Tensor): Lambda-returns from ValueLoss.
                                   Shape: [H, B, 1] or [B, H, 1]
            entropy (torch.Tensor): Pre-computed entropy of the policy distribution.
                                   Shape: [H, B, 1] or [B, H, 1]
            log_prob (torch.Tensor, optional): Log probabilities of the actions taken.
                                   Shape: [H, B, 1] or [B, H, 1]
            pcont (torch.Tensor, optional): Continuation probabilities.
                                   Shape: [H, B, 1] or [B, H, 1]
                                  
        Returns:
            loss (torch.Tensor): Scalar combined loss to minimize.
            target_loss (torch.Tensor): Scalar loss from negative targets only.
            entropy_loss (torch.Tensor): Scalar loss from negative entropy only.
            entropy (torch.Tensor): Mean entropy for logging.
        """
        
        if len(entropy.shape) < len(targets.shape):
            entropy = entropy.unsqueeze(-1)

        if self.batch_first:
            targets = targets.transpose(0, 1)
            entropy = entropy.transpose(0, 1)
            if log_prob is not None:
                log_prob = log_prob.transpose(0, 1)
            if pcont is not None:
                pcont = pcont.transpose(0, 1)

        if pcont is None:
            # Compute per-step discount weights gamma^t, shape [H, 1, 1]
            H = targets.shape[0]
            discount_weights = torch.tensor(
                [self.discount ** t for t in range(H)],
                dtype=targets.dtype, device=targets.device
            ).reshape(H, 1, 1)
        else:
            H, B, _ = targets.shape
            # discount_weights[t] = gamma^t * \prod_{i=0}^{t-1} pcont[i]
            ones = torch.ones(1, B, 1, dtype=pcont.dtype, device=pcont.device)
            discounts = torch.cat([ones, self.discount * pcont[:-1]], dim=0)
            discount_weights = torch.cumprod(discounts, dim=0)

        if self.reinforce:
            assert log_prob is not None, "log_prob must be provided if reinforce is True"
            # REINFORCE loss: log_prob * detached_targets
            target_loss = (-discount_weights * targets.detach() * log_prob).mean()
        else:
            # Pathwise/analytical gradients loss
            target_loss = (-discount_weights * targets).mean()

        entropy_loss = (-discount_weights * self.entropy_scale * entropy).mean()
        combined_loss = target_loss + entropy_loss

        return combined_loss, target_loss, entropy_loss, entropy.mean()


class ReturnEMA(nn.Module):
    """
    DreamerV3-style return normalization using exponential moving average
    of percentiles. Tracks the 5th and 95th percentile of lambda-returns
    and normalizes them to roughly [0, 1].
    
    This prevents the actor from seeing raw return magnitudes that can
    cause gradient explosion (large returns) or vanishing (tiny returns),
    which is a key contributor to bang-bang policy collapse.
    """

    def __init__(self, decay=0.99, low_percentile=5, high_percentile=95):
        super().__init__()
        self.decay = decay
        self.low_pct = low_percentile / 100.0
        self.high_pct = high_percentile / 100.0
        self.register_buffer('low', torch.tensor(0.0))
        self.register_buffer('high', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def update(self, returns):
        """Update running percentiles from a batch of returns."""
        flat = returns.detach().float().flatten()
        low = torch.quantile(flat, self.low_pct)
        high = torch.quantile(flat, self.high_pct)

        if not self.initialized:
            self.low.copy_(low)
            self.high.copy_(high)
            self.initialized.fill_(True)
        else:
            self.low.copy_(self.decay * self.low + (1 - self.decay) * low)
            self.high.copy_(self.decay * self.high + (1 - self.decay) * high)

    def normalize(self, returns):
        """
        Normalize returns using tracked percentiles.
        Gradients flow through `returns` — only the scale/offset are fixed (stop-gradient).
        """
        scale = (self.high - self.low).clamp(min=1e-2)
        return (returns - self.low) / scale