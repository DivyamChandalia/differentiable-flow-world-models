"""
models/augmentations.py
Modular, sequence-level consistent data augmentations for world model training.

All operations are designed to be *sequence-consistent*: the same random
parameters are drawn once per sequence so that temporal coherence is preserved
(e.g. all frames in a trajectory get the same crop shift or the same colour
jitter, simulating realistic sensor variation rather than independent per-frame
noise).
"""

import torch
import torch.nn.functional as F


def random_shift(imgs: torch.Tensor, pad: int = 3) -> torch.Tensor:
    """Apply a *single* random spatial shift consistently across an entire sequence.

    The shift is sampled once per batch element so all T frames in that
    sequence share the same (dx, dy) translation.  This prevents the
    augmentation from artificially breaking temporal coherence.

    Args:
        imgs: Float tensor of shape ``(B, T, C, H, W)`` in [0, 1].
        pad:  Number of pixels to pad on each side before cropping. Larger
              values give larger possible shifts (pad=3 → up to 3px shift).

    Returns:
        Shifted tensor of the same shape ``(B, T, C, H, W)``.
    """
    B, T, C, H, W = imgs.shape

    # Pad each spatial dimension symmetrically
    padded = F.pad(
        imgs.reshape(B * T, C, H, W),
        (pad, pad, pad, pad),
        mode='replicate',
    )  # (B*T, C, H+2p, W+2p)
    H_p, W_p = H + 2 * pad, W + 2 * pad

    # Sample ONE crop offset per batch element (shape: (B, 1) broadcast over T)
    top  = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)  # [0, 2*pad]
    left = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)

    # Repeat the same shift for every frame in the sequence
    top_T  = top.unsqueeze(1).expand(B, T).reshape(B * T)   # (B*T,)
    left_T = left.unsqueeze(1).expand(B, T).reshape(B * T)

    # Build a grid manually so we can apply per-sample shifts inside the batch
    # theta for affine_grid: [[1, 0, tx], [0, 1, ty]]  (normalised coordinates)
    #   tx = (left - pad) / (W/2),  ty = (top - pad) / (H/2)
    tx = (left_T.float() - pad) / (W / 2.0)
    ty = (top_T.float()  - pad) / (H / 2.0)

    theta = torch.zeros(B * T, 2, 3, device=imgs.device, dtype=imgs.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    # We want to sample a (H, W) window from the padded image.
    # Scale factor to map output coords in [-1,1] to padded-image coords:
    #   scale_x = W / W_p,  scale_y = H / H_p
    scale_x = W / W_p
    scale_y = H / H_p
    theta[:, 0, 0] = scale_x
    theta[:, 1, 1] = scale_y
    # Translation in normalised padded coords:
    #   centre of the desired crop (in padded pixels) = left + W/2, top + H/2
    #   normalised centre = (2 * centre / W_p) - 1
    cx_norm = (2.0 * (left_T.float() + W / 2.0) / W_p) - 1.0
    cy_norm = (2.0 * (top_T.float()  + H / 2.0) / H_p) - 1.0
    theta[:, 0, 2] = cx_norm
    theta[:, 1, 2] = cy_norm

    grid = F.affine_grid(theta, (B * T, C, H, W), align_corners=False)
    shifted = F.grid_sample(
        padded.to(dtype=imgs.dtype),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return shifted.reshape(B, T, C, H, W)


def augment_obs(
    imgs: torch.Tensor,
    use_shift: bool = False,
    use_color: bool = False,
    noise_std: float = 0.0,
    shift_pad: int = 3,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> torch.Tensor:
    """Combine spatial shifts, sequence-consistent colour jitter, and sensor noise.

    Each augmentation is independently toggled so the caller can mix and match.

    Args:
        imgs:        Float tensor ``(B, T, C, H, W)`` in [0, 1].
        use_shift:   If True, apply a sequence-consistent random spatial shift.
        use_color:   If True, apply sequence-consistent brightness & contrast
                     jitter (same parameters for all frames in a sequence).
        noise_std:   Standard deviation of *independent* per-pixel Gaussian
                     sensor noise added on top.  0.0 = disabled.
        shift_pad:   Padding size (pixels) for :func:`random_shift`.
        brightness:  Max absolute brightness delta in [0, 1].
        contrast:    Max relative contrast scale factor delta.

    Returns:
        Augmented tensor of the same shape, clamped to [0, 1].
    """
    out = imgs

    if use_shift:
        out = random_shift(out, pad=shift_pad)

    if use_color:
        B, T, C, H, W = out.shape
        # Draw one brightness / contrast factor per batch element
        delta_b = (torch.rand(B, device=out.device, dtype=out.dtype) * 2 - 1) * brightness  # (B,)
        delta_c = 1.0 + (torch.rand(B, device=out.device, dtype=out.dtype) * 2 - 1) * contrast  # (B,)

        # Reshape for broadcast: (B, 1, 1, 1, 1)
        delta_b = delta_b.view(B, 1, 1, 1, 1)
        delta_c = delta_c.view(B, 1, 1, 1, 1)

        # Contrast: scale around per-sequence mean to avoid overall brightness shift
        seq_mean = out.mean(dim=(1, 2, 3, 4), keepdim=True)
        out = delta_c * (out - seq_mean) + seq_mean + delta_b

    if noise_std > 0.0:
        out = out + noise_std * torch.randn_like(out)

    return out.clamp(0.0, 1.0)
