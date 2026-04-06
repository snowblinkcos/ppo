"""
optical_sim.py  —  Angular Spectrum Method (ASM) propagation
                   + detector-strip scoring for the focusing task.

Paper parameters (energy focusing):
  λ = 520 nm,  z = 9.6 cm,  pixel_size = 16 μm,  layer = 128×128
"""

import math
import torch


# ---------------------------------------------------------------------------
# ASM propagation
# ---------------------------------------------------------------------------

def asm_propagate(field: torch.Tensor, z: float, wavelength: float,
                  pixel_size: float) -> torch.Tensor:
    """
    Angular Spectrum Method propagation.

    H(fx, fy) = exp(2πj·z·√(1/λ² − fx² − fy²))  for propagating modes
              = 0                                    for evanescent modes

    Args:
        field      : complex tensor (..., H, W)
        z          : propagation distance [m]
        wavelength : wavelength [m]
        pixel_size : pixel pitch [m]

    Returns:
        output_field : complex tensor (..., H, W)
    """
    H, W   = field.shape[-2], field.shape[-1]
    device = field.device

    fx = torch.fft.fftfreq(W, d=pixel_size, device=device)   # (W,)
    fy = torch.fft.fftfreq(H, d=pixel_size, device=device)   # (H,)
    Fy, Fx = torch.meshgrid(fy, fx, indexing='ij')            # (H, W)

    k       = 1.0 / wavelength                                # 1/λ  [cycles/m]
    f_sq    = Fx ** 2 + Fy ** 2
    prop    = f_sq <= k ** 2                                  # propagating mask

    kz = torch.zeros(H, W, device=device, dtype=torch.float32)
    kz[prop] = torch.sqrt((k ** 2 - f_sq[prop]).clamp(min=0.0))

    tf = torch.zeros(H, W, dtype=torch.complex64, device=device)
    tf[prop] = torch.exp(1j * (2.0 * math.pi * z * kz[prop]))

    spectrum = torch.fft.fft2(field)
    return torch.fft.ifft2(spectrum * tf)


# ---------------------------------------------------------------------------
# Detector ROIs  —  3-4-3 grid layout (10 square regions, as in paper Fig. 1a)
# ---------------------------------------------------------------------------

def make_grid_rois(H: int, W: int, roi_size: int = 20):
    """
    Create 10 square ROIs arranged in a 3-4-3 grid pattern.

    Layout (paper Fig. 1a):
        Row 0:  ○ ○ ○          (3 ROIs)
        Row 1:  ○ ○ ○ ○        (4 ROIs)
        Row 2:  ○ ○ ○          (3 ROIs)

    ROI index order: row-major, left-to-right, top-to-bottom.
      0,1,2  → row 0
      3,4,5,6 → row 1
      7,8,9  → row 2

    Args:
        H, W     : detector plane size (pixels)
        roi_size : side length of each square ROI (pixels)

    Returns:
        rois : list of 10 (y0, y1, x0, x1) tuples
    """
    layout = [3, 4, 3]
    n_rows = len(layout)

    gap_y = (H - n_rows * roi_size) // (n_rows + 1)
    rois  = []

    for row, n_cols in enumerate(layout):
        y0  = gap_y + row * (roi_size + gap_y)
        y1  = y0 + roi_size
        gap_x = (W - n_cols * roi_size) // (n_cols + 1)
        for col in range(n_cols):
            x0 = gap_x + col * (roi_size + gap_x)
            x1 = x0 + roi_size
            rois.append((y0, y1, x0, x1))

    return rois


def roi_scores(intensity: torch.Tensor, rois: list) -> torch.Tensor:
    """
    Sum intensity inside each rectangular ROI.

    Args:
        intensity : float tensor (N, H, W)
        rois      : list of (y0, y1, x0, x1) tuples

    Returns:
        scores : float tensor (N, n_rois)
    """
    parts = [
        intensity[:, y0:y1, x0:x1].sum(dim=(-2, -1))
        for y0, y1, x0, x1 in rois
    ]
    return torch.stack(parts, dim=1)   # (N, n_rois)
