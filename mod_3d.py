"""
mod_3d.py
Plane + parallax moving-object detection for monocular video (PyTorch, CUDA when available).
Implements M. Irani & P. Anandan, "A Unified Approach to Moving Object Detection in 2D and 3D
Scenes", IEEE TPAMI 20(6), 1998.

Sections:
    1. Utilities
    2. 2D Parametric Motion (Eq. 2, 4) & Dominant-Motion Initialisation
    3. Local Misalignment
    4. Dense Planar-Parallax Flow (Eq. 5-6)
    5. Parallax Rigidity Constraint (Eq. 7-8)
    6. Moving Object Detector (Steps 1-9 + Stratification)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ===============================================
# UTILITIES
# ===============================================
"""Low-level tensor utilities (all GPU-friendly, pure PyTorch).

Conventions
-----------
* Images are float tensors of shape (B, 1, H, W) with values in [0, 1].
* "Normalized" coordinates follow ``grid_sample(..., align_corners=True)``:
  x_n = 2 x / (W - 1) - 1, y_n = 2 y / (H - 1) - 1, so pixel centres of the
  corner pixels are at -1 / +1.  Parametric motion models are expressed in
  normalized coordinates so their parameters are resolution independent
  (this makes coarse-to-fine propagation trivial).
* Flows / displacements returned to the user are in **pixels**.
"""

# -------------------------- Device --------------------------
def get_device(prefer: str | None = None) -> torch.device:
  if prefer is not None:
    return torch.device(prefer)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
      return torch.device("mps")
  return torch.device("cpu")

def frame_to_tensor(frame, device: torch.device) -> torch.Tensor:
  """uint8 HxW or HxWx3 (RGB or BGR - luminance weights are near-symmetric) -> (1,1,H,W) float"""
  t = torch.as_tensor(frame, device=device)
  if t.ndim == 3:
      t = t.float()
      t = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
  else:
      t = t.float()
  return (t / 255.0)[None, None]

# -------------------------- Filtering --------------------------
def _gauss1d(sigma: float, device, dtype) -> torch.Tensor:
  """1D Gaussian blur along the last dimension of a 4D tensor"""
  r = max(1, math.ceil(3.0 * sigma))
  x = torch.arange(-r, r + 1, device=device, dtype=dtype)
  k = torch.exp(-0.5 * (x / sigma) ** 2)
  return k / k.sum()

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
  """Separable Gaussian blur with replicate padding. img: (B,C,H,W)"""
  if sigma <= 0:
    return img
  k = _gauss1d(sigma, img.device, img.dtype)
  r = (k.numel() - 1) // 2
  C = img.shape[1]
  kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
  ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
  out = F.pad(img, (r, r, 0, 0), mode="replicate")
  out = F.conv2d(out, kx, groups=C)
  out = F.pad(out, (0, 0, r, r), mode="replicate")
  out = F.conv2d(out, ky, groups=C)
  return out

def box_filter(img: torch.Tensor, radius: int) -> torch.Tensor:
  """Mean over a (2r+1)^2 window; borders use only valid pixels"""
  if radius <= 0:
    return img
  k = 2 * radius + 1
  return F.avg_pool2d(img, k, stride=1, padding=radius, count_include_pad=False)

def gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Central-difference image gradients in pixel units (replicate border)."""
  p = F.pad(img, (1, 1, 1, 1), mode="replicate")
  gx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
  gy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
  return gx, gy

def median_filter3(x: torch.Tensor) -> torch.Tensor:
  """3x3 median filter (per channel)"""
  B, C, H, W = x.shape
  p = F.pad(x, (1, 1, 1, 1), mode="replicate")
  patches = F.unfold(p, 3) # (B, C*9, H*W)
  patches = patches.view(B, C, 9, H * W)
  return patches.median(dim=2).values.view(B, C, H, W)

def dialate(mask: torch.Tensor, radius: int) -> torch.Tensor:
  """Dilate a binary mask by a given radius (in pixels)"""
  if radius <= 0:
    return mask
  m = mask.float()
  return F.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius) > 0.5

def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
  if radius <= 0:
    return mask
  return ~dialate(~mask.bool(), radius)

# -------------------------- Pyramids --------------------------
def build_pyramid(img: torch.Tensor, levels: int, kind: str = "gaussian") -> list[torch.Tensor]:
  """Returns list [finest, ..., coarsest].
  kind = 'gaussian'  : low-pass images.
  kind = 'laplacian' : band-pass images  (I - G*I), as in Irani & Anandan
  Sec. 2.1; less sensitive to illumination changes.
  """
  pyr = [img]
  for _ in range(levels - 1):
    b = gaussian_blur(pyr[-1], 1.0)
    pyr.append(b[..., ::2, ::2].contiguous())
  if kind == "laplacian":
    pyr = [p - gaussian_blur(p, 2.0) for p in pyr]
  elif kind != "gaussian":
    raise ValueError(kind)
  return pyr

def auto_levels(H: int, W: int, min_size: int = 24, max_levels: int = 6) -> int:
  n = 1
  while min(H, W) / (2 ** n) >= min_size and n < max_levels:
    n += 1
  return n

def downsample_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
  return F.interpolate(mask.float(), size=shape, mode="area")

# -------------------------- Coordinates --------------------------
def normalized_grid(H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
  ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
  xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
  Y, X = torch.meshgrid(ys, xs, indexing="ij")
  return X, Y


def pixel_grid(H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
  ys = torch.arange(H, device=device, dtype=dtype)
  xs = torch.arange(W, device=device, dtype=dtype)
  Y, X = torch.meshgrid(ys, xs, indexing="ij")
  return X, Y


def warp_by_flow(img: torch.Tensor, flow: torch.Tensor, padding: str = "border") -> torch.Tensor:
  """Sample img at x + flow(x). flow: (B,2,H,W) in pixels."""
  _, _, H, W = img.shape
  X, Y = pixel_grid(H, W, img.device, img.dtype)
  gx = (X + flow[:, 0]) * (2.0 / max(W - 1, 1)) - 1.0
  gy = (Y + flow[:, 1]) * (2.0 / max(H - 1, 1)) - 1.0
  grid = torch.stack([gx, gy], dim=-1)
  return F.grid_sample(img, grid, mode="bilinear", padding_mode=padding, align_corners=True)


def flow_valid(flow: torch.Tensor) -> torch.Tensor:
  _, _, H, W = flow.shape
  X, Y = pixel_grid(H, W, flow.device, flow.dtype)
  x = X + flow[:, 0]
  y = Y + flow[:, 1]
  return ((x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1))[:, None]


def border_mask(H: int, W: int, margin: int, device) -> torch.Tensor:
  m = torch.zeros(1, 1, H, W, dtype=torch.bool, device=device)
  m[..., margin:H - margin, margin:W - margin] = True
  return m
