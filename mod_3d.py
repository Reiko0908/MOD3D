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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# 1. UTILITIES
# ============================================================
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
  """1D normalized Gaussian kernel"""
  r = max(1, math.ceil(3.0 * sigma))
  x = torch.arange(-r, r + 1, device=device, dtype=dtype)
  k = torch.exp(-0.5 * (x / sigma) ** 2)
  return k / k.sum()

def gauss_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
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

def grad(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
  """Dilate a binary mask by a given radius (in pixels)"""
  if radius <= 0:
    return mask
  m = mask.float()
  return F.max_pool2d(m, 2 * radius + 1, stride=1, padding=radius) > 0.5

def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
  if radius <= 0:
    return mask
  return ~dilate(~mask.bool(), radius)

# -------------------------- Pyramids --------------------------
def build_pyramid(img: torch.Tensor, levels: int, kind: str = "gaussian") -> list[torch.Tensor]:
  """Returns list [finest, ..., coarsest].
  kind = 'gaussian'  : low-pass images.
  kind = 'laplacian' : band-pass images  (I - G*I), as in Irani & Anandan
  Sec. 2.1; less sensitive to illumination changes.
  """
  pyr = [img]
  for _ in range(levels - 1):
    b = gauss_blur(pyr[-1], 1.0)
    pyr.append(b[..., ::2, ::2].contiguous())
  if kind == "laplacian":
    pyr = [p - gauss_blur(p, 2.0) for p in pyr]
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
def norm_grid(H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
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

# ============================================================
# 2. 2D PARAMETRIC MOTION + DOMINANT-MOTION INITIALIZATION
# ===========================================================
MODELS = {"translation", "affine", "quadratic", "homography"}
# indices into the 8-vector for each model family
#   quadratic : (a, b, c, d, e, f, g, h)      as in Eq. (2)
#   homography: H = I + [[p0 p1 p2],[p3 p4 p5],[p6 p7 0]]
_ACTIVE = {
  "quadratic": {"translation": [0, 3], "affine": [0, 1, 2, 3, 4, 5], "full": list(range(8))},
  "homography": {"translation": [2, 5], "affine": [0, 1, 2, 3, 4, 5], "full": list(range(8))},
}

def _family(model: str) -> str:
  if model in ("translation", "affine", "quadratic"):
    return "quadratic"
  if model == "homography":
    return "homography"
  raise ValueError(f"unknown model {model!r}; choose from {MODELS}")

def _max_stage(model: str) -> str:
  return {"translation": "translation", "affine": "affine"}.get(model, "full")

# -------------------------- Model Evaluation --------------------------
def apply_model(p: torch.Tensor, model: str, X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Warp normalized coords (H,W) by params p(B,8) -> (B,H,W) x', y'"""
  fam = _family(model)
  P = [p[:, i].view(-1, 1, 1) for i in range(8)]
  if fam == "quadratic":
    a, b, c, d, e, f, g, h = P
    u = a + b * X + c * Y + g * X * X + h * X * Y
    v = d + e * X + f * Y + g * X * Y + h * Y * Y
    return X + u, Y + v
  # homography
  D = P[6] * X + P[7] * Y + 1.0
  Nx = (1.0 + P[0]) * X + P[1] * Y + P[2]
  Ny = P[3] * X + (1.0 + P[4]) * Y + P[5]
  return Nx / D, Ny / D

def model_jacobian(p: torch.Tensor, model: str, X: torch.Tensor, Y:  torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  B = p.shape[0]
  H, W = X.shape
  fam = _family(model)
  one = torch.ones_like(X)
  zero = torch.zeros_like(X)
  if fam == "quadratic":
    JX = torch.stack([one, X, Y, zero, zero, zero, X * X, X * Y])
    JY = torch.stack([zero, zero, zero, one, X, Y, X * Y, Y * Y])
    return JX.unsqueeze(0).expand(B, -1, -1, -1), JY.unsqueeze(0).expand(B, -1, -1, -1)
  xw, yw = apply_model(p, model, X, Y)
  D = p[:, 6].view(-1, 1, 1) * X + p[:, 7].view(-1, 1, 1) * Y + 1.0
  iD = 1.0 / D
  Xb, Yb = X.expand(B, H, W), Y.expand(B, H, W)
  z = torch.zeros_like(xw)
  JX = torch.stack([Xb * iD, Yb * iD, iD, z, z, z, -Xb * xw * iD, -Yb * xw * iD], 1)
  JY = torch.stack([z, z, z, Xb * iD, Yb * iD, iD, -Xb * yw * iD, -Yb * yw * iD], 1)
  return JX, JY

def displacement_px(p: torch.Tensor, model: str, H: int, W: int) -> torch.Tensor:
  """Displacement field u(x) = W(x;p) - x in pixels, (B,2,H,W)"""
  X, Y = norm_grid(H, W, p.device, p.dtype)
  xw, yw = apply_model(p, model, X, Y)
  u = (xw - X) * (W - 1) / 2.0
  v = (yw - Y) * (H - 1) / 2.0
  return torch.stack([u, v], 1)

def warp_img(img: torch.Tensor, p: torch.Tensor, model: str) -> tuple[torch.Tensor, torch.Tensor]:
  """Resample inspection image(s) img (B,1,H,W) at W(x;p)
  (warped, valid) where valid marks pixel whose source lies inside"""
  _, _, H, W = img.shape
  X, Y = norm_grid(H, W, img.device, img.dtype)
  xw, yw = apply_model(p, model, X, Y)
  grid = torch.stack([xw, yw], -1)
  out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
  valid = ((xw.abs() <= 1.0) & (yw.abs() <= 1.0))[:, None]
  return out, valid

def identity_params(B: int, device, dtype=torch.float32) -> torch.Tensor:
  return torch.zeros(B, 8, device=device, dtype=dtype)

# -------------------------- Estimation --------------------------
@dataclass(frozen=True)
class ParamConfig:
  model: str = "homography"          # 'translation' | 'affine' | 'quadratic' | 'homography'
  levels: int | None = None      # pyramid levels (None -> auto)
  pyramid: str = "gaussian"          # 'gaussian' | 'laplacian'
  iters: int = 12                    # Gauss-Newton iterations per level
  robust: bool = True                # Geman-McClure IRLS outlier rejection
  robust_c: float = 2.0              # GM scale in units of the MAD noise estimate
  sigma_min: float = 0.004           # floor on the residual scale (intensity units)
  damping: float = 1e-4              # Levenberg-Marquardt damping (relative)
  tol: float = 1e-6                  # stop when max |delta p| < tol

def _stage_sched(n_levels: int, model: str) -> list[str]:
  """Stage for each level, coarsest first (paper: translation -> affine -> full)."""
  maxs = _max_stage(model)
  order = ["translation", "affine", "full"]
  cap = order.index(maxs)
  if n_levels == 1:
    st = ["full"]
  elif n_levels == 2:
    st = ["affine", "full"]
  else:
    n_t = max(1, n_levels // 3)
    n_a = max(1, (n_levels - n_t) // 2)
    st = ["translation"] * n_t + ["affine"] * n_a + ["full"] * (n_levels - n_t - n_a)
  return [order[min(order.index(s), cap)] for s in st]

def _weighted_median_abs(r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
  """Per-batch median |r} over pixels w/ w > 0 -> (B,)"""
  B = r.shape[0]
  out = []
  for b in range(B):
    sel = r[b][w[b] > 0]
    out.append(sel.abs().median() if sel.numel() > 0 else r.new_tensor(1.0))
  return torch.stack(out)

@torch.no_grad()
def est_param(ref: torch.Tensor, insp: torch.Tensor, cfg: ParamConfig = ParamConfig(),
              roi: torch.Tensor | None = None, init: torch.Tensor | None = None) -> torch.Tensor:
  """Estimate the dominant 2D parametric motion aligning ``insp`` to ``ref``."""
  B = insp.shape[0]
  if ref.shape[0] == 1 and B > 1:
    ref = ref.expand(B, -1, -1, -1)
  H, W = ref.shape[-2:]
  L = cfg.levels or auto_levels(H, W)
  pyr_r = build_pyramid(ref, L, cfg.pyramid)
  pyr_i = build_pyramid(insp, L, cfg.pyramid)
  stages = _stage_sched(L, cfg.model)
  fam = _family(cfg.model)
  p = identity_params(B, ref.device, ref.dtype) if init is None else init.clone()

  for li, lvl in enumerate(range(L - 1, -1, -1)): # coarest -> finest
    R, I = pyr_r[lvl], pyr_i[lvl]
    h, w = R.shape[-2:]
    X, Y = norm_grid(h, w, R.device, R.dtype)
    gx, gy = grad(I)
    gx = gx * (w - 1) / 2.0 # dI/dx_norm
    gy = gy * (h - 1) / 2.0
    G = torch.cat([I, gx, gy], 1)
    roi_l = None if roi is None else downsample_mask(roi, (h, w))
    active = _ACTIVE[fam][stages[li]]

    for it in range(cfg.iters):
      xw, yw = apply_model(p, cfg.model, X, Y)
      grid = torch.stack([xw, yw], -1)
      S = F.grid_sample(G, grid, mode="bilinear", padding_mode="border", align_corners=True)
      Iw, Gx, Gy = S[:, 0], S[:, 1], S[:, 2]
      valid = ((xw.abs() <= 1.0) & (yw.abs() <= 1.0)).to(R.dtype)
      r = Iw - R[:, 0]
      wgt = valid if roi_l is None else valid * roi_l[:, 0]
      if cfg.robust:
        sig = 1.4826 * _weighted_median_abs(r, wgt)
        sig = torch.clamp(sig, min=cfg.sigma_min).view(-1, 1, 1) * cfg.robust_c
        wgt = wgt * (sig ** 2 / (sig ** 2 + r ** 2)) ** 2
      JX, JY = model_jacobian(p, cfg.model, X, Y)
      J = Gx[:, None] * JX[:, active] + Gy[:, None] * JY[:, active]  # (B,k,h,w)
      Jw = J * wgt[:, None]
      A = torch.einsum("bihw,bjhw->bij", Jw, J)
      g = torch.einsum("bihw,bhw->bi", Jw, r)
      diag = torch.diagonal(A, dim1=1, dim2=2)
      A = A + torch.diag_embed(cfg.damping * diag + 1e-9)
      try:
        delta = torch.linalg.solve(A, g)
      except RuntimeError:
        delta = torch.linalg.lstsq(A, g[..., None]).solution[..., 0]
      p[:, active] = p[:, active] - delta
      if delta.abs().max().item() < cfg.tol:
        break
  return p

# -------------------------- Initialization --------------------------
@torch.no_grad()
def phase_corr_peaks(ref: torch.Tensor, insp: torch.Tensor, k: int = 3, nms: int = 6) -> torch.Tensor:
  """Top-k global translation hypotheses d (pixels) with insp(x + d) ~= ref(x),
  from the phase-correlation surface (GPU FFT).  Returns (B, k, 2)."""
  B, _, H, W = insp.shape
  if ref.shape[0] == 1 and B > 1:
    ref = ref.expand(B, -1, -1, -1)
  win = torch.outer(torch.hann_window(H, periodic=False, device=ref.device, dtype=ref.dtype),
                    torch.hann_window(W, periodic=False, device=ref.device, dtype=ref.dtype))
  a = (ref[:, 0] - ref[:, 0].mean(dim=(1, 2), keepdim=True)) * win
  b = (insp[:, 0] - insp[:, 0].mean(dim=(1, 2), keepdim=True)) * win
  R = torch.fft.rfft2(b) * torch.fft.rfft2(a).conj()
  c = torch.fft.irfft2(R / R.abs().clamp_min(1e-9), s=(H, W))
  c = torch.fft.fftshift(c, dim=(-2, -1))[:, None]          # zero shift at the centre
  peaks = F.max_pool2d(c, 2 * nms + 1, stride=1, padding=nms)
  c = torch.where(c >= peaks, c, torch.full_like(c, -1e9)).flatten(1)
  idx = c.topk(k, dim=1).indices
  dy = (idx // W - H // 2).to(ref.dtype)
  dx = (idx % W - W // 2).to(ref.dtype)
  return torch.stack([dx, dy], -1)

def _trans_to_params(d: torch.Tensor, model: str, H: int, W: int) -> torch.Tensor:
  p = torch.zeros(d.shape[:-1] + (8,), device=d.device, dtype=d.dtype)
  ix, iy = (0, 3) if _family(model) == "quadratic" else (2, 5)
  p[..., ix] = d[..., 0] * 2.0 / max(W - 1, 1)
  p[..., iy] = d[..., 1] * 2.0 / max(H - 1, 1)
  return p

@torch.no_grad()
def est_dominant(ref: torch.Tensor, insp: torch.Tensor, cfg: ParamConfig = ParamConfig(),
                      warm: torch.Tensor | None = None, n_peaks: int = 3, align_thr: float = 0.6) -> torch.Tensor:
  """Dominant 2D motion = the parametric motion that aligns the LARGEST image area."""
  B, _, H, W = insp.shape
  if ref.shape[0] == 1 and B > 1:
    ref = ref.expand(B, -1, -1, -1)
  hyps = [torch.zeros(B, 8, device=ref.device, dtype=ref.dtype)]
  if n_peaks > 0:
    pk = phase_corr_peaks(ref, insp, n_peaks)          # (B, k, 2)
    hyps += [_trans_to_params(pk[:, i], cfg.model, H, W) for i in range(pk.shape[1])]
  if warm is not None:
    hyps.append(warm.to(ref))
  n = len(hyps)
  init = torch.cat(hyps, 0)                                     # (n*B, 8), hypothesis-major
  R = ref.repeat(n, 1, 1, 1)
  I = insp.repeat(n, 1, 1, 1)
  p = est_param(R, I, cfg, init=init)
  Wp, valid = warp_img(I, p, cfg.model)
  M, tex = norm_flow_misalign(R, Wp, 2)
  good = ((M < align_thr) & (tex > 2e-4) & valid).float().mean(dim=(1, 2, 3)).view(n, B)
  best = good.argmax(0) # (B,)
  return p.view(n, B, 8)[best, torch.arange(B, device=p.device)]

# ============================================================
# 3. LOCAL MISALIGNMENT
# ============================================================
@torch.no_grad()
def norm_flow_misalign(ref: torch.Tensor, warped: torch.Tensor, radius: int = 2,
                       eps: float = 1e-5, presmooth: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (M, texture) both (B,1,H,W). M in pixels"""
  if presmooth > 0:
    ref = gauss_blur(ref, presmooth)
    warped = gauss_blur(warped, presmooth)
  gx, gy = grad(0.5 * (ref + warped))
  g2 = gx * gx + gy * gy
  diff = (ref - warped).abs()
  num = box_filter(diff * g2.sqrt(), radius)
  tex = box_filter(g2, radius)
  return num / (tex + eps), tex

def misalignment_mask(M: torch.Tensor, texture: torch.Tensor, valid: torch.Tensor,
                      thr: float = 0.5, tex_thr: float = 2e-4) -> torch.Tensor:
    """Boolean mask of misaligned (moving or parallax) pixels."""
    return (M > thr) & (texture > tex_thr) & valid.bool()
