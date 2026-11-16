"""
mod_3d.py

Plane + parallax moving-object detection for monocular image triplets/sequences
(PyTorch, CUDA when available). Implements M. Irani & P. Anandan, "A Unified
Approach to Moving Object Detection in 2D and 3D Scenes", IEEE TPAMI 20(6), 1998,
with binary MRF graph-cut energy minimisation (max-flow / min-cut) labelling.

Sections:
    1. Utilities
    2. 2D Parametric Motion (Eq. 2, 4) & Dominant-Motion Initialization
    3. Local Misalignment
    4. Dense Planar-Parallax Flow (Eq. 5-6)
    5. Parallax Rigidity Constraint (Eq. 7-8)
    6. Graph-Cut Labelling (Binary MRF, Max-Flow / Min-Cut)
    7. Moving Object Detector (Steps 1-9 + Stratification + Labelling)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace, asdict

import cv2
import maxflow
import numpy as np
import torch
import torch.nn.functional as F

# =============================================================
# 1. UTILITIES
# =============================================================


# -------------------------- Device --------------------------
def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def frame_to_tensor(
    frame: np.ndarray, device: torch.device, bgr: bool = True
) -> torch.Tensor:
    """uint8 HxW or HxWx3 -> (1, 1, H, W) float luminance in [0, 1]."""
    t = torch.as_tensor(frame, device=device, dtype=torch.float32)
    if t.ndim == 3:
        w0, w2 = (0.114, 0.299) if bgr else (0.299, 0.114)
        t = w0 * t[..., 0] + 0.587 * t[..., 1] + w2 * t[..., 2]
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
    return F.conv2d(out, ky, groups=C)


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
    """3x3 median filter (per channel)."""
    B, C, H, W = x.shape
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    patches = F.unfold(p, 3).view(B, C, 9, H * W)
    return patches.median(dim=2).values.view(B, C, H, W)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a binary mask by a given radius (in pixels)"""
    if radius <= 0:
        return mask
    return F.max_pool2d(mask.float(), 2 * radius + 1, stride=1, padding=radius) > 0.5


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return ~dilate(~mask.bool(), radius)


# -------------------------- Pyramids --------------------------
def build_pyramid(
    img: torch.Tensor, levels: int, kind: str = "gaussian"
) -> list[torch.Tensor]:
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
        raise ValueError(f"Unknown pyramid kind: {kind!r}")
    return pyr


def auto_levels(H: int, W: int, min_size: int = 24, max_levels: int = 6) -> int:
    n = 1
    while min(H, W) / (2**n) >= min_size and n < max_levels:
        n += 1
    return n


def downsample_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=shape, mode="area")


# -------------------------- Coordinates --------------------------
def norm_grid(
    H: int, W: int, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y


def pixel_grid(
    H: int, W: int, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y


def warp_by_flow(
    img: torch.Tensor, flow: torch.Tensor, padding: str = "border"
) -> torch.Tensor:
    """Sample img at x + flow(x). flow: (B,2,H,W) in pixels."""
    _, _, H, W = img.shape
    X, Y = pixel_grid(H, W, img.device, img.dtype)
    gx = (X + flow[:, 0]) * (2.0 / max(W - 1, 1)) - 1.0
    gy = (Y + flow[:, 1]) * (2.0 / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode=padding, align_corners=True
    )


def flow_valid(flow: torch.Tensor) -> torch.Tensor:
    _, _, H, W = flow.shape
    X, Y = pixel_grid(H, W, flow.device, flow.dtype)
    x = X + flow[:, 0]
    y = Y + flow[:, 1]
    return ((x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1))[:, None]


def border_mask(H: int, W: int, margin: int, device) -> torch.Tensor:
    m = torch.zeros(1, 1, H, W, dtype=torch.bool, device=device)
    if margin > 0 and H > 2 * margin and W > 2 * margin:
        m[..., margin : H - margin, margin : W - margin] = True
    else:
        m.fill_(True)
    return m


# =============================================================
# 2. 2D PARAMETRIC MOTION + DOMINANT-MOTION INITIALIZATION
# ============================================================
MODELS = {"translation", "affine", "quadratic", "homography"}

# indices into the 8-vector for each model family
#   quadratic : (a, b, c, d, e, f, g, h)      as in Eq. (2)
#   homography: H = I + [[p0 p1 p2],[p3 p4 p5],[p6 p7 0]]
_ACTIVE = {
    "quadratic": {
        "translation": [0, 3],
        "affine": [0, 1, 2, 3, 4, 5],
        "full": list(range(8)),
    },
    "homography": {
        "translation": [2, 5],
        "affine": [0, 1, 2, 3, 4, 5],
        "full": list(range(8)),
    },
}


def _family(model: str) -> str:
    if model in ("translation", "affine", "quadratic"):
        return "quadratic"
    if model == "homography":
        return "homography"
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def _max_stage(model: str) -> str:
    return {"translation": "translation", "affine": "affine"}.get(model, "full")


def _safe_denom(D: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    sign = torch.where(D >= 0, torch.ones_like(D), -torch.ones_like(D))
    return torch.where(D.abs() < eps, sign * eps, D)


# -------------------------- Model Evaluation --------------------------
def apply_model(
    p: torch.Tensor, model: str, X: torch.Tensor, Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp normalized coords (H,W) by params p(B,8) -> (B,H,W) x', y'"""
    fam = _family(model)
    P = [p[:, i].view(-1, 1, 1) for i in range(8)]
    if fam == "quadratic":
        a, b, c, d, e, f, g, h = P
        u = a + b * X + c * Y + g * X * X + h * X * Y
        v = d + e * X + f * Y + g * X * Y + h * Y * Y
        return X + u, Y + v
    # homography
    D = _safe_denom(P[6] * X + P[7] * Y + 1.0)
    Nx = (1.0 + P[0]) * X + P[1] * Y + P[2]
    Ny = P[3] * X + (1.0 + P[4]) * Y + P[5]
    return Nx / D, Ny / D


def model_jacobian(
    p: torch.Tensor, model: str, X: torch.Tensor, Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """dx'/dp and dy'/dp, each (B,8,H,W)"""
    B = p.shape[0]
    H, W = X.shape
    fam = _family(model)
    one = torch.ones_like(X)
    zero = torch.zeros_like(X)
    if fam == "quadratic":
        JX = torch.stack([one, X, Y, zero, zero, zero, X * X, X * Y])
        JY = torch.stack([zero, zero, zero, one, X, Y, X * Y, Y * Y])
        return JX.unsqueeze(0).expand(B, -1, -1, -1), JY.unsqueeze(0).expand(
            B, -1, -1, -1
        )
    xw, yw = apply_model(p, model, X, Y)
    D = _safe_denom(p[:, 6].view(-1, 1, 1) * X + p[:, 7].view(-1, 1, 1) * Y + 1.0)
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


def warp_img(
    img: torch.Tensor, p: torch.Tensor, model: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample inspection image(s) img (B,1,H,W) at W(x;p)
    (warped, valid) where valid marks pixel whose source lies inside"""
    _, _, H, W = img.shape
    X, Y = norm_grid(H, W, img.device, img.dtype)
    xw, yw = apply_model(p, model, X, Y)
    grid = torch.stack([xw, yw], -1)
    out = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    valid = ((xw.abs() <= 1.0) & (yw.abs() <= 1.0))[:, None]
    return out, valid


def identity_params(B: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(B, 8, device=device, dtype=dtype)


# -------------------------- Estimation --------------------------
@dataclass(frozen=True)
class ParamConfig:
    model: str = "homography"  # 'translation' | 'affine' | 'quadratic' | 'homography'
    levels: int | None = None  # pyramid levels (None -> auto)
    pyramid: str = "gaussian"  # 'gaussian' | 'laplacian'
    iters: int = 12  # Gauss-Newton iterations per level
    robust: bool = True  # Geman-McClure IRLS outlier rejection
    robust_c: float = 2.0  # GM scale in units of the MAD noise estimate
    sigma_min: float = 0.004  # floor on the residual scale (intensity units)
    damping: float = 1e-4  # Levenberg-Marquardt damping (relative)
    tol: float = 1e-6  # stop when max |delta p| < tol


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
        st = (
            ["translation"] * n_t + ["affine"] * n_a + ["full"] * (n_levels - n_t - n_a)
        )
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
def est_param(
    ref: torch.Tensor,
    insp: torch.Tensor,
    cfg: ParamConfig = ParamConfig(),
    roi: torch.Tensor | None = None,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
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

    for li, lvl in enumerate(range(L - 1, -1, -1)):  # coarest -> finest
        R, I = pyr_r[lvl], pyr_i[lvl]
        h, w = R.shape[-2:]
        X, Y = norm_grid(h, w, R.device, R.dtype)
        gx, gy = grad(I)
        gx = gx * (w - 1) / 2.0  # dI/dx_norm
        gy = gy * (h - 1) / 2.0
        G = torch.cat([I, gx, gy], 1)
        roi_l = None if roi is None else downsample_mask(roi, (h, w))
        active = _ACTIVE[fam][stages[li]]

        for it in range(cfg.iters):
            xw, yw = apply_model(p, cfg.model, X, Y)
            grid = torch.stack([xw, yw], -1)
            S = F.grid_sample(
                G, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            Iw, Gx, Gy = S[:, 0], S[:, 1], S[:, 2]
            valid = ((xw.abs() <= 1.0) & (yw.abs() <= 1.0)).to(R.dtype)
            r = Iw - R[:, 0]
            wgt = valid if roi_l is None else valid * roi_l[:, 0]
            if cfg.robust:
                sig = 1.4826 * _weighted_median_abs(r, wgt)
                sig = torch.clamp(sig, min=cfg.sigma_min).view(-1, 1, 1) * cfg.robust_c
                wgt = wgt * (sig**2 / (sig**2 + r**2)) ** 2
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
def phase_corr_peaks(
    ref: torch.Tensor, insp: torch.Tensor, k: int = 3, nms: int = 6
) -> torch.Tensor:
    """Top-k global translation hypotheses d (pixels) via GPU FFT phase correlation."""
    B, _, H, W = insp.shape
    if ref.shape[0] == 1 and B > 1:
        ref = ref.expand(B, -1, -1, -1)
    win = torch.outer(
        torch.hann_window(H, periodic=False, device=ref.device, dtype=ref.dtype),
        torch.hann_window(W, periodic=False, device=ref.device, dtype=ref.dtype),
    )
    a = (ref[:, 0] - ref[:, 0].mean(dim=(1, 2), keepdim=True)) * win
    b = (insp[:, 0] - insp[:, 0].mean(dim=(1, 2), keepdim=True)) * win
    R = torch.fft.rfft2(b) * torch.fft.rfft2(a).conj()
    c = torch.fft.irfft2(R / R.abs().clamp_min(1e-9), s=(H, W))
    c = torch.fft.fftshift(c, dim=(-2, -1))[:, None]  # zero shift at the centre
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
def est_dominant(
    ref: torch.Tensor,
    insp: torch.Tensor,
    cfg: ParamConfig = ParamConfig(),
    warm: torch.Tensor | None = None,
    n_peaks: int = 3,
    align_thr: float = 0.6,
    tex_thr: float = 2e-4,
) -> torch.Tensor:
    """Dominant 2D motion = the parametric motion that aligns the LARGEST image area."""
    B, _, H, W = insp.shape
    if ref.shape[0] == 1 and B > 1:
        ref = ref.expand(B, -1, -1, -1)
    hyps = [torch.zeros(B, 8, device=ref.device, dtype=ref.dtype)]
    if n_peaks > 0:
        pk = phase_corr_peaks(ref, insp, n_peaks)  # (B, k, 2)
        hyps += [
            _trans_to_params(pk[:, i], cfg.model, H, W) for i in range(pk.shape[1])
        ]
    if warm is not None:
        hyps.append(warm.to(ref))
    n = len(hyps)
    init = torch.cat(hyps, 0)  # (n*B, 8), hypothesis-major
    R = ref.repeat(n, 1, 1, 1)
    I = insp.repeat(n, 1, 1, 1)
    p = est_param(R, I, cfg, init=init)
    Wp, valid = warp_img(I, p, cfg.model)
    M, tex = norm_flow_misalign(R, Wp, 2)
    good = (
        ((M < align_thr) & (tex > tex_thr) & valid)
        .float()
        .mean(dim=(1, 2, 3))
        .view(n, B)
    )
    best = good.argmax(0)  # (B,)
    return p.view(n, B, 8)[best, torch.arange(B, device=p.device)]


# =============================================================
# 3. LOCAL MISALIGNMENT
# =============================================================
@torch.no_grad()
def norm_flow_misalign(
    ref: torch.Tensor,
    warped: torch.Tensor,
    radius: int = 2,
    eps: float = 1e-5,
    presmooth: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def misalign_mask(
    M: torch.Tensor,
    texture: torch.Tensor,
    valid: torch.Tensor,
    thr: float = 0.5,
    tex_thr: float = 2e-4,
) -> torch.Tensor:
    """Boolean mask of misaligned (moving or parallax) pixels."""
    return (M > thr) & (texture > tex_thr) & valid.bool()


# =============================================================
# 4. DENSE PLANAR-PARALLAX FLOW
# =============================================================
def _shift(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """out(x, y) = img(x + dx, y + dy) with replicate border."""
    if dx == 0 and dy == 0:
        return img
    r = max(abs(dx), abs(dy))
    p = F.pad(img, (r, r, r, r), mode="replicate")
    H, W = img.shape[-2:]
    return p[..., r + dy : r + dy + H, r + dx : r + dx + W]


def _win_offset(radius: int, device, dtype):
    offs, wts = [], []
    sig = max(radius / 1.5, 0.5)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offs.append((dx, dy))
            wts.append(math.exp(-0.5 * (dx * dx + dy * dy) / sig**2))
    w = torch.tensor(wts, device=device, dtype=dtype)
    return offs, w / w.sum()


@torch.no_grad()
def _lk_lvl(
    R: torch.Tensor,
    I: torch.Tensor,
    flow: torch.Tensor,
    radius: int,
    iters: int,
    lam: float,
    max_step: float,
    median: bool,
    chunk: int = 7,
):
    B, _, H, W = R.shape
    offs, wts = _win_offset(radius, R.device, R.dtype)
    gx, gy = grad(R)
    Rn = torch.cat([_shift(R, dx, dy) for dx, dy in offs], dim=1)
    Gxn = torch.cat([_shift(gx, dx, dy) for dx, dy in offs], dim=1)
    Gyn = torch.cat([_shift(gy, dx, dy) for dx, dy in offs], dim=1)
    w_view = wts.view(1, -1, 1, 1)
    Sxx = (w_view * Gxn * Gxn).sum(dim=1, keepdim=True)
    Syy = (w_view * Gyn * Gyn).sum(dim=1, keepdim=True)
    Sxy = (w_view * Gxn * Gyn).sum(dim=1, keepdim=True)
    lamb = lam * 0.5 * (Sxx + Syy).mean(dim=(1, 2, 3), keepdim=True) + 1e-12
    Axx, Ayy = Sxx + lamb, Syy + lamb
    det = Axx * Ayy - Sxy * Sxy
    X, Y = pixel_grid(H, W, R.device, R.dtype)
    sx, sy = 2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)
    n_off = len(offs)

    for _ in range(iters):
        bx = torch.zeros_like(R)
        by = torch.zeros_like(R)
        base_x = X + flow[:, 0]
        base_y = Y + flow[:, 1]
        for s in range(0, n_off, chunk):
            e = min(s + chunk, n_off)
            idx = range(s, e)
            gxs = torch.cat(
                [(base_x + offs[i][0]) * sx - 1.0 for i in idx], 1
            )  # (B, c*H, W)
            gys = torch.cat([(base_y + offs[i][1]) * sy - 1.0 for i in idx], 1)
            grid = torch.stack([gxs, gys], dim=-1)
            samp = F.grid_sample(
                I, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            samp = samp.view(B, len(idx), H, W)
            It = samp - Rn[:, s:e]
            wc = w_view[:, s:e]
            bx = bx + (wc * Gxn[:, s:e] * It).sum(dim=1, keepdim=True)
            by = by + (wc * Gyn[:, s:e] * It).sum(dim=1, keepdim=True)
        du = -(Ayy * bx - Sxy * by) / det
        dv = -(Axx * by - Sxy * bx) / det
        flow = flow + torch.cat([du, dv], 1).clamp(-max_step, max_step)
        if median:
            flow = median_filter3(flow)
    conf = 0.5 * (
        Sxx + Syy - torch.sqrt(torch.clamp((Sxx - Syy) ** 2 + 4 * Sxy**2, min=0))
    )
    return flow, conf


@torch.no_grad()
def _block_search(
    R: torch.Tensor, I: torch.Tensor, flow: torch.Tensor, s: int, radius: int
) -> torch.Tensor:
    """Interger-displacement search around ``flow`` minimizing windowed SAD"""
    best = None
    best_d = torch.zeros_like(flow)
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            d = torch.tensor([dx, dy], device=R.device, dtype=R.dtype).view(1, 2, 1, 1)
            Iw = warp_by_flow(I, flow + d)
            c = box_filter((Iw - R).abs(), radius) + 1e-4 * (dx * dx + dy * dy)
            if best is None:
                best = c
                best_d = d.expand_as(flow).clone()
            else:
                better = c < best
                best = torch.where(better, c, best)
                best_d = torch.where(better.expand_as(flow), d.expand_as(flow), best_d)
    return flow + best_d


@torch.no_grad()
def dense_lk(
    ref: torch.Tensor,
    insp: torch.Tensor,
    init: torch.Tensor | None = None,
    levels: int | None = None,
    iters: int = 6,
    radius: int = 3,
    max_step: float = 2.0,
    median: bool = True,
    min_size: int = 20,
    search: int = 3,
    lam: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow mu such that insp(x + mu(x)) ~= ref(x)"""
    B = insp.shape[0]
    if ref.shape[0] == 1 and B > 1:
        ref = ref.expand(B, -1, -1, -1)
    H, W = ref.shape[-2:]
    L = auto_levels(H, W, min_size=min_size, max_levels=5) if levels is None else levels
    if L < 1:
        raise ValueError(f"levels must be >= 1, got {L}")
    pr = build_pyramid(gauss_blur(ref, 0.7), L)
    pi = build_pyramid(gauss_blur(insp, 0.7), L)

    # Coarsest level initialization
    R, I = pr[L - 1], pi[L - 1]
    h, w = R.shape[-2:]
    if init is not None:
        flow = F.interpolate(init, size=(h, w), mode="bilinear", align_corners=True) / (
            2 ** (L - 1)
        )
    else:
        flow = torch.zeros(B, 2, h, w, device=R.device, dtype=R.dtype)
    if search > 0:
        flow = _block_search(R, I, flow, search, radius)
    flow, conf = _lk_lvl(R, I, flow, radius, iters, lam, max_step, median)

    # Remaining finer levels
    for lvl in range(L - 2, -1, -1):
        R, I = pr[lvl], pi[lvl]
        h, w = R.shape[-2:]
        flow = 2.0 * F.interpolate(
            flow, size=(h, w), mode="bilinear", align_corners=True
        )
        flow, conf = _lk_lvl(R, I, flow, radius, iters, lam, max_step, median)
    return flow, conf


# =============================================================
# 5. PARALLAX RIGIDITY CONSTRAINT
# =============================================================
def _cross(ax, ay, bx, by):
    """Cross product of 2D vectors (ax, ay) x (bx, by)"""
    return ax * by - ay * bx


@torch.no_grad()
def pairwise_inconsistency(
    P: torch.Tensor,
    Mj: torch.Tensor,
    Mk: torch.Tensor,
    R: torch.Tensor,
    Rj: torch.Tensor,
    Rk: torch.Tensor,
    eps: float = 0.25,
    min_dist: float = 6.0,
    min_b: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigidity inconsistency of N points w.r.t K ref points"""
    Dxj = (P[None, :, 0] + Mj[None, :, 0]) - (R[:, None, 0] + Rj[:, None, 0])
    Dyj = (P[None, :, 1] + Mj[None, :, 1]) - (R[:, None, 1] + Rj[:, None, 1])
    Dxk = (P[None, :, 0] + Mk[None, :, 0]) - (R[:, None, 0] + Rk[:, None, 0])
    Dyk = (P[None, :, 1] + Mk[None, :, 1]) - (R[:, None, 1] + Rk[:, None, 1])
    nj = torch.sqrt(Dxj**2 + Dyj**2).clamp_min(1e-6)
    nk = torch.sqrt(Dxk**2 + Dyk**2).clamp_min(1e-6)

    aj = _cross(Dxj, Dyj, Mj[None, :, 0], Mj[None, :, 1]) / nj
    ak = _cross(Dxk, Dyk, Mk[None, :, 0], Mk[None, :, 1]) / nk
    bj = _cross(Dxj, Dyj, Rj[:, None, 0], Rj[:, None, 1]) / nj
    bk = _cross(Dxk, Dyk, Rk[:, None, 0], Rk[:, None, 1]) / nk
    num = (aj * bk - ak * bj).abs()
    den = torch.sqrt(aj**2 + ak**2 + bj**2 + bk**2 + eps**2)
    m = num / den
    informative = (
        (torch.sqrt(bj**2 + bk**2) >= min_b) & (nj >= min_dist) & (nk >= min_dist)
    )
    return m, informative


def _masked_median(m: torch.Tensor, valid: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Median of m over valid pixels (B,)"""
    x = m.masked_fill(~valid, float("nan"))
    med = torch.nanmedian(x, dim=dim).values
    return torch.nan_to_num(med, nan=0.0)


@torch.no_grad()
def select_ref_points(
    mu_j: torch.Tensor,
    mu_k,
    cand: torch.Tensor,
    n_cand: int = 300,
    n_ref: int = 32,
    seed: int = 0,
    user_points: torch.Tensor | None = None,
    **kw,
) -> dict[str, torch.Tensor]:
    """Pick static ref points"""
    H, W = cand.shape
    dev = mu_j.device
    if user_points is not None:
        pts = user_points.to(dev).float()
        xi = pts[:, 0].round().long().clamp(0, W - 1)
        yi = pts[:, 1].round().long().clamp(0, H - 1)
        return {
            "pts": pts,
            "mj": mu_j[:, yi, xi].T,
            "mk": mu_k[:, yi, xi].T,
            "score": torch.zeros(len(pts), device=dev),
        }
    idx = torch.nonzero(cand.flatten(), as_tuple=False)[:, 0]
    if idx.numel() < 8:
        return {
            "pts": torch.zeros(0, 2, device=dev),
            "mj": torch.zeros(0, 2, device=dev),
            "mk": torch.zeros(0, 2, device=dev),
            "score": torch.zeros(0, device=dev),
        }
    g = torch.Generator(device="cpu").manual_seed(seed)
    if idx.numel() > n_cand:
        sel = torch.randperm(idx.numel(), generator=g)[:n_cand].to(dev)
        idx = idx[sel]
    ys, xs = idx // W, idx % W
    pts = torch.stack([xs, ys], 1).float()
    mj, mk = mu_j[:, ys, xs].T, mu_k[:, ys, xs].T
    m, inf = pairwise_inconsistency(pts, mj, mk, pts, mj, mk, **kw)
    score = _masked_median(m, inf, dim=1)
    n_inf = inf.sum(1)
    score = torch.where(n_inf >= 5, score, torch.full_like(score, float("inf")))
    order = torch.argsort(score)[: min(n_ref, len(score))]
    order = order[torch.isfinite(score[order])]
    return {"pts": pts[order], "mj": mj[order], "mk": mk[order], "score": score[order]}


@torch.no_grad()
def inconsistency_map(
    mu_j: torch.Tensor,
    mu_k: torch.Tensor,
    refs: dict[str, torch.Tensor],
    region: torch.Tensor | None = None,
    min_refs: int = 3,
    chunk: int = 65536,
    **kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median rigidity inconsistency of each pixel w.r.t. the reference set."""
    _, H, W = mu_j.shape
    dev = mu_j.device
    out = torch.zeros(H * W, device=dev)
    cnt = torch.zeros(H * W, device=dev)
    if refs["pts"].shape[0] == 0:
        return out.view(H, W), cnt.view(H, W)
    X, Y = pixel_grid(H, W, dev)
    P = torch.stack([X.flatten(), Y.flatten()], 1)
    Mj = mu_j.reshape(2, -1).T
    Mk = mu_k.reshape(2, -1).T
    idx = (
        torch.arange(H * W, device=dev)
        if region is None
        else torch.nonzero(region.flatten())[:, 0]
    )
    for s in range(0, idx.numel(), chunk):
        ii = idx[s : s + chunk]
        m, inf = pairwise_inconsistency(
            P[ii], Mj[ii], Mk[ii], refs["pts"], refs["mj"], refs["mk"], **kw
        )
        n = inf.sum(0)
        med = _masked_median(m, inf, dim=0)
        out[ii] = torch.where(n >= min_refs, med, torch.zeros_like(med))
        cnt[ii] = n.float()
    return out.view(H, W), cnt.view(H, W)


# =============================================================
# 6. GRAPH-CUT LABELLING (Binary MRF, Max-Flow / Min-Cut)
# =============================================================
@dataclass
class GraphCutConfig:
    solver: str = "bk"  # 'dinic' (SciPy) | 'bk' (PyMaxflow)
    thr: float = 0.6  # evidence threshold [px]
    k: float = 2.0  # log-odds slope
    lmax: float = 3.0  # clip of the log-odds
    lam: float = 8.0  # smoothness weight relative to the data term
    w0: float = 0.1  # contrast-independent pairwise weight
    sigma: float = 0.15  # contrast sensitivity (intensity in [0, 1])
    unknown_bias: float = 0.05  # cost of labelling an unmeasurable pixel "moving"
    neighbours: int = 8  # 4 or 8
    scale: int = 1000  # integer capacity scale for Dinic
    min_area: int = 60
    # Second pass for additional smaller moving objects
    others: bool = True
    others_lam: float = 4.0
    others_thr: float = 0.5
    others_min_area: int = 40
    others_seed_thr: float = 1.2
    others_seed_n: int = 12
    others_dil: int = 8
    others_photo: bool = True
    others_photo_scale: float = 0.6
    others_photo_seed: float = 3.0


def layered_evidence(
    r: dict[str, np.ndarray], tex_thr: float = 2e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous layered motion evidence: misaligne w/ both plan and 2nd layer"""
    s = np.minimum(r["ev_plane"], r["ev_layer2"])
    measurable = (
        (r["tex_plane"] > tex_thr)
        & (r["tex_layer2"] > tex_thr)
        & r["valid_both"]
        & r["valid_layer2"]
    )
    return s, measurable


def rigidity_evidence(
    r: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """3D level: rigidity inconsistency (px), eval where reliable parallax exists."""
    m = r["inconsistency"]
    return m, m > 0


def log_odds(
    s: np.ndarray,
    measurable: np.ndarray,
    cfg: GraphCutConfig,
    thr: float | None = None,
) -> np.ndarray:
    t = cfg.thr if thr is None else thr
    l = cfg.k * np.log(np.maximum(s, 1e-3) / max(t, 1e-6))
    l = np.clip(l, -cfg.lmax, cfg.lmax)
    return np.where(measurable, l, 0.0)


class Grid:
    """Neighbour idx arr for an HxW grid (cached per resolution"""

    def __init__(self, H: int, W: int, neighbours: int = 8):
        self.H, self.W = H, W
        idx = np.arange(H * W).reshape(H, W)
        pairs, dists = [], []
        shifts = [(0, 1, 1.0), (1, 0, 1.0)] + (
            [(1, 1, 2**0.5), (1, -1, 2**0.5)] if neighbours == 8 else []
        )
        for dy, dx, dist in shifts:
            ys, yd = slice(0, H - dy), slice(dy, H)
            if dx >= 0:
                xs, xd = slice(0, W - dx), slice(dx, W)
            else:
                xs, xd = slice(-dx, W), slice(0, W + dx)
            pairs.append((idx[ys, xs].ravel(), idx[yd, xd].ravel()))
            dists.append(dist)
        self.p = np.concatenate([a for a, _ in pairs])
        self.q = np.concatenate([b for _, b in pairs])
        self.dist = np.concatenate(
            [np.full(len(a), d) for (a, _), d in zip(pairs, dists)]
        )


_GRIDS: dict[tuple[int, int, int], Grid] = {}


def _grid(H: int, W: int, n: int) -> Grid:
    if (H, W, n) not in _GRIDS:
        _GRIDS[(H, W, n)] = Grid(H, W, n)
    return _GRIDS[(H, W, n)]


def pairwise_weights(gray: np.ndarray, cfg: GraphCutConfig) -> tuple[Grid, np.ndarray]:
    H, W = gray.shape
    g = _grid(H, W, cfg.neighbours)
    I = gray.astype(np.float32).ravel() / 255.0
    d2 = (I[g.p] - I[g.q]) ** 2
    contrast = np.exp(-d2 / (2 * cfg.sigma**2)) if cfg.sigma > 0 else np.ones_like(d2)
    return g, cfg.lam * (cfg.w0 + contrast) / g.dist


def solve_binary(
    l: np.ndarray,
    measurable: np.ndarray,
    gray: np.ndarray,
    cfg: GraphCutConfig,
    hard_static: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Minimise binary MRF energy over {static, moving} via Boykov-Kolmogorov s-t min-cut"""
    H, W = l.shape
    d_move = np.maximum(0.0, -l)
    d_stat = np.maximum(0.0, l)
    d_move = np.where(measurable, d_move, cfg.unknown_bias)
    if hard_static is not None:
        d_move = np.where(hard_static, 1e3, d_move)
        d_stat = np.where(hard_static, 0.0, d_stat)
    net = d_stat - d_move
    src_cap = np.maximum(net, 0).ravel()
    snk_cap = np.maximum(-net, 0).ravel()
    g, w = pairwise_weights(gray, cfg)
    t0 = time.perf_counter()

    N = H * W
    gr = maxflow.Graph[float](N, len(g.p))
    gr.add_nodes(N)
    gr.add_edges(
        g.p.astype(np.int32),
        g.q.astype(np.int32),
        w.astype(np.float64),
        w.astype(np.float64),
    )
    gr.add_grid_tedges(
        np.arange(N), src_cap.astype(np.float64), snk_cap.astype(np.float64)
    )
    flow = gr.maxflow()
    seg = gr.get_grid_segments(np.arange(N))
    lab = (~seg).reshape(H, W)
    info = {
        "solve_s": time.perf_counter() - t0,
        "flow": float(flow),
        "nodes": N,
        "edges": len(g.p),
        "solver": "bk",
    }
    return lab, info


def secondary_objects(
    r: dict[str, np.ndarray],
    main: np.ndarray,
    cfg: GraphCutConfig,
    border: int = 10,
    tex_thr: float = 2e-4,
) -> np.ndarray:
    """Second graph-cut pass with weaker smoothing to recover smaller secondary moving objects."""
    sec_cfg = replace(cfg, lam=cfg.others_lam, thr=cfg.others_thr)
    s, meas = layered_evidence(r, tex_thr=tex_thr)
    l = log_odds(s, meas, sec_cfg)
    seed = (s > cfg.others_seed_thr) & meas
    meas2 = meas
    if cfg.others_photo and "res_plane" in r:
        thr_ph = r["res_thr"] * cfg.others_photo_scale
        use = (~meas) & r["res_valid"]
        l_ph = np.clip(
            sec_cfg.k * np.log(np.maximum(r["res_plane"], 1e-4) / thr_ph),
            -sec_cfg.lmax,
            sec_cfg.lmax,
        )
        l = np.where(use, l_ph, l)
        meas2 = meas | use
        seed = seed | (use & (r["res_plane"] > cfg.others_photo_seed * thr_ph))
    H, W = l.shape
    hard = np.ones((H, W), bool)
    if border > 0 and H > 2 * border and W > 2 * border:
        hard[border:-border, border:-border] = False
    else:
        hard.fill(False)
    lab, _ = solve_binary(l, meas2, r["gray"], sec_cfg, hard_static=hard)
    d = 2 * cfg.others_dil + 1
    zone = cv2.dilate(main.astype(np.uint8), np.ones((d, d), np.uint8)) > 0
    cand = (lab & ~zone).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    if n > 1:
        mask_idx = np.asarray(seed & (cand > 0), dtype=bool)
        cnt = np.bincount(lb[mask_idx], minlength=n)
        keep[1:] = (st[1:, 4] >= cfg.others_min_area) & (cnt[1:] >= cfg.others_seed_n)
    out = keep[lb].astype(np.uint8)
    if out.any():
        cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def label_frame(
    r: dict[str, np.ndarray],
    cfg: GraphCutConfig,
    level: str | None = None,
    border: int = 10,
    tex_thr: float = 2e-4,
    rig_thr: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Graph-cut labelling of one frame from the detector's continuous evidence."""
    level = level or r["level"]
    if level == "layered":
        s, meas = layered_evidence(r, tex_thr=tex_thr)
        l = log_odds(s, meas, cfg)
    else:
        s, meas = rigidity_evidence(r)
        l = log_odds(s, meas, cfg, thr=rig_thr or cfg.thr)
    H, W = l.shape
    hard = np.ones((H, W), bool)
    if border > 0 and H > 2 * border and W > 2 * border:
        hard[border:-border, border:-border] = False
    else:
        hard.fill(False)
    return solve_binary(l, meas, r["gray"], cfg, hard_static=hard)


# =============================================================
# 7. MOVING OBJECT DETECTOR
# ============================================================
@dataclass
class MODConfig:
    # --- 2D parametric registration (Sec. 2) ---
    model: str = "homography"
    pyramid: str = "gaussian"
    gn_iters: int = 12
    init_peaks: int = 3
    # --- misalignment detection (Sec. 2 / [14]) ---
    misalign_thr: float = 0.6
    misalign_radius: int = 2
    texture_thr: float = 2e-4
    # --- layers (Sec. 3) ---
    use_layer2: bool = True
    min_layer_frac: float = 0.01
    # --- parallax (Sec. 4.1) ---
    skip_3d: bool = (
        False  # True skips Steps 7-9 (dense LK + rigidity) for pure 2D/layered speed
    )
    parallax: str = "dense"  # 'dense' | 'layers'
    layer_init: bool = True
    lk_radius: int = 3
    lk_iters: int = 5
    lk_search: int = 3
    min_conf_rel: float = 0.1
    flow_residual_thr: float = 0.0
    # --- rigidity constraint (Sec. 4.2) ---
    gap: int = 3
    min_parallax: float = 0.7
    n_candidates: int = 400
    n_refs: int = 40
    rigidity_thr: float = 1.2
    rigidity_low: float = 0.3
    rigidity_smooth: int = 2
    rig_eps: float = 0.25
    rig_min_dist: float = 8.0
    rig_min_b: float = 0.5
    ref_points: list[tuple[float, float]] | None = None
    # --- stratification ---
    stratify: bool = True
    dominant_frac: float = 0.75
    # --- labelling ---
    labeling: str = (
        "auto"  # 'auto' (graph-cut MRF) | 'threshold' (hysteresis + morphology)
    )
    gc: GraphCutConfig = field(default_factory=GraphCutConfig)
    # --- post-processing ---
    border: int = 10
    open_radius: int = 1
    close_radius: int = 2
    min_area: int = 60
    fill_holes: bool = True
    final_close_frac: float = 0.012
    hole_close_frac: float = 0.03
    hole_thick_frac: float = 0.015
    seed: int = 0

    def param_cfg(self) -> ParamConfig:
        return ParamConfig(model=self.model, pyramid=self.pyramid, iters=self.gn_iters)


def _components(
    m: np.ndarray, cfg: MODConfig, seeds: np.ndarray | None = None
) -> np.ndarray:
    m = m.astype(np.uint8)
    if not m.any():
        return m.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_area
    if seeds is not None:
        has_seed = np.zeros(n, bool)
        has_seed[np.unique(lab[seeds & (m > 0)])] = True
        keep &= has_seed
    out = keep[lab].astype(np.uint8)
    if cfg.fill_holes and out.any():
        cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def _close_fill(m: np.ndarray, radius: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mc = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mc)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out > 0


def _fill_thick_holes(
    pre: np.ndarray, base: np.ndarray, big_radius: int, thick_radius: int
) -> np.ndarray:
    closed = _close_fill(pre, big_radius)
    add = (closed & ~base).astype(np.uint8)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thick_radius + 1, 2 * thick_radius + 1)
    )
    add = cv2.morphologyEx(add, cv2.MORPH_OPEN, k) > 0
    return _close_fill(base | add, 1)


def _largest_share(m: np.ndarray) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0.0
    a = stats[1:, cv2.CC_STAT_AREA]
    return float(a.max() / a.sum())


def _cleanup(
    mask: torch.Tensor, cfg: MODConfig, seeds: torch.Tensor | None = None
) -> np.ndarray:
    m = mask.bool()
    if cfg.open_radius > 0:
        m = dilate(erode(m, cfg.open_radius), cfg.open_radius)
    if cfg.close_radius > 0:
        m = erode(dilate(m, cfg.close_radius), cfg.close_radius)
    sd = None if seeds is None else seeds[0, 0].cpu().numpy().astype(bool)
    return _components(m[0, 0].cpu().numpy(), cfg, sd)


class MovingObjectDetector:
    """Plane + parallax moving-object detector for monocular image sequences."""

    def __init__(self, cfg: MODConfig | None = None, device: str | None = None):
        self.cfg = MODConfig() if cfg is None else cfg
        self.device = get_device(device)
        self._init_p1: dict[tuple[int, int], torch.Tensor] = {}

    @torch.no_grad()
    def detect_tensors(
        self,
        ref: torch.Tensor,
        Ij: torch.Tensor,
        Ik: torch.Tensor,
        init_key: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        cfg = self.cfg
        dev = self.device
        H, W = ref.shape[-2:]
        insp = torch.cat([Ij, Ik], 0)
        bmask = border_mask(H, W, cfg.border, dev)
        pc = cfg.param_cfg()

        # ---- steps 1-2: dominant 2D parametric motion (batched over j, k)
        init = self._init_p1.get(init_key) if init_key is not None else None
        if cfg.init_peaks > 0:
            p1 = est_dominant(
                ref,
                insp,
                pc,
                warm=init,
                n_peaks=cfg.init_peaks,
                align_thr=cfg.misalign_thr,
                tex_thr=cfg.texture_thr,
            )
        else:
            p1 = est_param(ref, insp, pc, init=init)
        if init_key is not None:
            self._init_p1[init_key] = p1.clone()

        # ---- steps 3-4: align and detect misaligned regions
        W1, valid1 = warp_img(insp, p1, cfg.model)
        M1, tex = norm_flow_misalign(ref.expand_as(W1), W1, cfg.misalign_radius)
        mis1 = misalign_mask(M1, tex, valid1, cfg.misalign_thr, cfg.texture_thr) & bmask
        valid_both = valid1.all(0, keepdim=True) & bmask

        # Photometric residual w.r.t. the dominant plane; also responds on
        # untextured, smooth surfaces.
        _d = (ref.expand_as(W1) - W1).abs()
        _d = torch.where(valid1, _d, torch.full_like(_d, 1e9))
        res_ph = _d.min(0, keepdim=True).values
        res_ok = valid1.any(0, keepdim=True) & bmask
        _v = res_ph[res_ok]
        res_thr = (
            float(
                max(
                    0.03,
                    3.5 * 1.4826 * (_v - _v.median()).abs().median().item(),
                )
            )
            if _v.numel()
            else 0.03
        )

        mis1_both = mis1.all(0, keepdim=True)
        mask_2d = mis1_both

        # ---- steps 5-6: second 2D layer fitted on the misaligned regions only
        p2 = None
        mis2 = mis1.clone()
        M2min, tex2min, valid2all = (
            M1.min(0).values,
            tex.min(0).values,
            valid1.all(0),
        )
        frac = mis1.float().mean().item()
        if cfg.use_layer2 and frac >= cfg.min_layer_frac:
            roi = (dilate(mis1, 2) & valid1).float()
            p2 = est_param(ref, insp, pc, roi=roi, init=p1)
            W2, valid2 = warp_img(insp, p2, cfg.model)
            M2, tex2 = norm_flow_misalign(ref.expand_as(W2), W2, cfg.misalign_radius)
            mis2 = (
                misalign_mask(M2, tex2, valid2, cfg.misalign_thr, cfg.texture_thr)
                | ~valid2
            )
            M2min, tex2min, valid2all = (
                M2.min(0).values,
                tex2.min(0).values,
                valid2.all(0),
            )
        mask_layered = (mis1 & mis2).all(0, keepdim=True)

        # ---- steps 7-9: planar-parallax flow & 3D rigidity constraint
        if cfg.skip_3d:
            mu_j = torch.zeros(2, H, W, device=dev, dtype=ref.dtype)
            mu_k = torch.zeros(2, H, W, device=dev, dtype=ref.dtype)
            refs = {"pts": torch.zeros(0, 2, device=dev)}
            m = torch.zeros_like(M1[:1])
            reliable = valid_both
            mask_3d = mask_layered
            seeds_3d = None
            used_3d = False
        else:
            u1 = displacement_px(p1, cfg.model, H, W)
            if p2 is not None:
                layer_par = (displacement_px(p2, cfg.model, H, W) - u1) * dilate(
                    mis1, 2
                ).float()
            else:
                layer_par = torch.zeros_like(u1)

            if cfg.parallax == "layers":
                mu = layer_par
                conf = torch.ones_like(M1)
            elif cfg.parallax == "dense":
                mu, conf = dense_lk(
                    ref,
                    W1,
                    init=layer_par if cfg.layer_init else None,
                    iters=cfg.lk_iters,
                    radius=cfg.lk_radius,
                    search=cfg.lk_search,
                )
            else:
                raise ValueError(cfg.parallax)

            mu_j, mu_k = mu[0], mu[1]

            if cfg.flow_residual_thr > 0:
                Wmu = warp_by_flow(W1, mu)
                Mres, _ = norm_flow_misalign(
                    ref.expand_as(Wmu), Wmu, cfg.misalign_radius
                )
                flow_ok = (Mres < cfg.flow_residual_thr).all(
                    0, keepdim=True
                ) & flow_valid(mu).all(0, keepdim=True)
            else:
                flow_ok = torch.ones_like(M1[:1], dtype=torch.bool)

            # ---- step 8: reference point(s) known / assumed to be static
            cthr = cfg.min_conf_rel * conf.flatten(1).median(dim=1).values.view(
                -1, 1, 1, 1
            )
            reliable = (conf >= cthr).all(0, keepdim=True) & flow_ok
            mag = mu.norm(dim=1, keepdim=True)
            cand = (
                mis1_both
                & reliable
                & (mag >= cfg.min_parallax).all(0, keepdim=True)
                & valid_both
            )
            user = (
                torch.tensor(cfg.ref_points, dtype=torch.float32)
                if cfg.ref_points
                else None
            )
            kw = dict(eps=cfg.rig_eps, min_dist=cfg.rig_min_dist, min_b=cfg.rig_min_b)
            refs = select_ref_points(
                mu_j,
                mu_k,
                cand[0, 0],
                cfg.n_candidates,
                cfg.n_refs,
                cfg.seed,
                user_points=user,
                **kw,
            )

            # ---- step 9: parallax-based rigidity constraint, Eq. (8)
            region = mis1_both & valid_both & reliable
            if refs["pts"].shape[0] > 0:
                m, _ = inconsistency_map(mu_j, mu_k, refs, region=region[0, 0], **kw)
                m = m[None, None]
                if cfg.rigidity_smooth > 0:
                    rf = region.float()
                    m = (
                        box_filter(m * rf, cfg.rigidity_smooth)
                        / box_filter(rf, cfg.rigidity_smooth).clamp_min(1e-6)
                        * rf
                    )
                mask_3d = (m > cfg.rigidity_low) & region
                seeds_3d = (m > cfg.rigidity_thr) & region
                used_3d = True
            else:
                m = torch.zeros_like(M1[:1])
                mask_3d = mask_layered
                seeds_3d = None
                used_3d = False

        m3d = _cleanup(mask_3d, cfg, seeds_3d)
        m2d = _cleanup(mask_2d, cfg)
        mlay = _cleanup(mask_layered, cfg)
        level = "3d" if used_3d else "layered"
        final = m3d
        if cfg.stratify and used_3d and _largest_share(mlay) >= cfg.dominant_frac:
            final, level = mlay, "layered"

        pre_final = final
        if cfg.final_close_frac > 0 and final.any():
            final = _close_fill(final, max(1, int(round(cfg.final_close_frac * W))))
            if cfg.hole_close_frac > 0:
                final = _fill_thick_holes(
                    pre_final,
                    final,
                    max(1, int(round(cfg.hole_close_frac * W))),
                    max(1, int(round(cfg.hole_thick_frac * W))),
                )

        out: dict[str, object] = {
            "pre_final": pre_final,
            "textured": (tex > cfg.texture_thr).all(0)[0].cpu().numpy(),
            "gray": (ref[0, 0].clamp(0, 1) * 255).byte().cpu().numpy(),
            "mask": final,
            "level": level,
            "mask_3d": m3d,
            "mask_2d": m2d,
            "mask_layered": mlay,
            "inconsistency": m[0, 0].cpu().numpy(),
            "misalignment": M1.min(0).values[0].cpu().numpy(),
            "parallax_j": mu_j.cpu().numpy(),
            "parallax_k": mu_k.cpu().numpy(),
            "ref_points": refs["pts"].cpu().numpy(),
            "p1": p1.cpu().numpy(),
            "p2": None if p2 is None else p2.cpu().numpy(),
            "used_rigidity": used_3d,
            "reliable": reliable[0, 0].cpu().numpy(),
            "mis_both": mis1_both[0, 0].cpu().numpy(),
            "ev_plane": M1.min(0).values[0].cpu().numpy(),
            "tex_plane": tex.min(0).values[0].cpu().numpy(),
            "ev_layer2": M2min[0].cpu().numpy(),
            "tex_layer2": tex2min[0].cpu().numpy(),
            "valid_both": valid_both[0, 0].cpu().numpy(),
            "valid_layer2": valid2all[0].cpu().numpy(),
            "res_plane": torch.where(res_ok, res_ph, torch.zeros_like(res_ph))[0, 0]
            .cpu()
            .numpy(),
            "res_valid": res_ok[0, 0].cpu().numpy(),
            "res_thr": res_thr,
            "mask_threshold": final,
            "mask_layered_gc": mlay,
            "mask_3d_gc": m3d,
        }

        if cfg.labeling != "threshold":
            lab_lay, info = label_frame(
                out,
                cfg.gc,
                level="layered",
                border=cfg.border,
                tex_thr=cfg.texture_thr,
            )
            main_gc = _components(lab_lay, cfg)
            out["mask_main_gc"] = main_gc
            out["mask_layered_gc"] = main_gc
            if cfg.gc.others:
                out["mask_layered_gc"] = main_gc | secondary_objects(
                    out,
                    main_gc,
                    cfg.gc,
                    border=cfg.border,
                    tex_thr=cfg.texture_thr,
                )
            out["graphcut_info"] = info

            if used_3d:
                lab_3d, _ = label_frame(
                    out,
                    cfg.gc,
                    level="3d",
                    border=cfg.border,
                    tex_thr=cfg.texture_thr,
                    rig_thr=cfg.rigidity_low,
                )
                out["mask_3d_gc"] = _components(
                    lab_3d, cfg, seeds=seeds_3d[0, 0].cpu().numpy()
                )

            if level == "layered":
                out["mask"] = out["mask_layered_gc"]
            elif used_3d:
                out["mask"] = out["mask_3d_gc"]

        return out

    def detect(
        self,
        ref: np.ndarray,
        frame_j: np.ndarray,
        frame_k: np.ndarray,
        init_key: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        return self.detect_tensors(
            frame_to_tensor(ref, self.device),
            frame_to_tensor(frame_j, self.device),
            frame_to_tensor(frame_k, self.device),
            init_key,
        )

    def triplet_indices(self, t: int, T: int) -> tuple[int, int]:
        if T < 3:
            raise ValueError(f"Need at least 3 frames to form a triplet, got T={T}")
        g = min(self.cfg.gap, (T - 1) // 2)
        if t - g >= 0 and t + g < T:
            return t - g, t + g
        if t + 2 * g < T:
            return t + g, t + 2 * g
        return max(t - 2 * g, 0), max(t - g, 0)

    def process_sequence(
        self, frames: Sequence[np.ndarray], callback=None, verbose: bool = True
    ) -> list[dict[str, object]]:
        """Run detection across an in-memory sequence of uint8 frames."""
        T = len(frames)
        tens: dict[int, torch.Tensor] = {}

        def get(i: int) -> torch.Tensor:
            if i not in tens:
                tens[i] = frame_to_tensor(frames[i], self.device)
            return tens[i]

        results = []
        t0 = time.time()
        for t in range(T):
            j, k = self.triplet_indices(t, T)
            res = self.detect_tensors(get(t), get(j), get(k), init_key=(j - t, k - t))
            res["frame"], res["j"], res["k"] = t, j, k
            results.append(res)
            for old in [i for i in tens if i < t - 2 * self.cfg.gap]:
                del tens[old]
            if callback is not None:
                callback(t, res)
            if verbose and (t % 10 == 0 or t == T - 1):
                print(
                    f"[MOD] frame {t + 1}/{T}  "
                    f"({(time.time() - t0) / (t + 1):.2f} s/frame, "
                    f"device={self.device})",
                    flush=True,
                )
        return results

    process_video = process_sequence


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "video",
        help="input monocular video file, or a folder of frames (jpg/png, sorted by name)",
    )
    ap.add_argument(
        "--gt",
        default=None,
        help="optional folder of ground-truth binary mask images (sorted by name) to compute IoU",
    )
    ap.add_argument("--out", default="mod_out", help="output directory")
    ap.add_argument(
        "--device",
        default=None,
        help="cuda | cuda:1 | cpu | mps (default: cuda if available)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=640,
        help="resize frames to this width (0 = keep)",
    )
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="process at most N frames (0 = all)",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=1,
        help="temporal subsampling of the input video",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=3,
        help="offset of frames j/k from the reference frame",
    )
    ap.add_argument(
        "--model",
        default="homography",
        choices=["homography", "quadratic", "affine", "translation"],
        help="2D parametric model of the dominant plane",
    )
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "stratified", "3d", "layered", "2d"],
        help="auto: layered level + graph-cut labelling + further moving objects (default); "
        "layered: same, main object only (no further-object pass); "
        "stratified: per-frame choice between layered level and 3D rigidity; "
        "3d: always parallax rigidity; 2d: single plane",
    )
    ap.add_argument(
        "--parallax",
        default="dense",
        choices=["dense", "layers"],
        help="dense: per-pixel residual flow; layers: u2 - u1",
    )
    ap.add_argument(
        "--rigidity-thr",
        type=float,
        default=None,
        help="seed threshold on inconsistency [px]",
    )
    ap.add_argument(
        "--rigidity-low",
        type=float,
        default=None,
        help="hysteresis low threshold [px]",
    )
    ap.add_argument(
        "--misalign-thr",
        type=float,
        default=None,
        help="normal-flow misalignment threshold [px]",
    )
    ap.add_argument(
        "--labeling",
        default="auto",
        choices=["auto", "threshold"],
        help="auto: graph-cut energy minimisation; threshold: hysteresis + morphology",
    )
    ap.add_argument(
        "--gc-lam",
        type=float,
        default=None,
        help="graph-cut smoothness weight (default 8)",
    )
    ap.add_argument(
        "--no-others",
        action="store_true",
        help="do not search for further moving objects",
    )
    ap.add_argument(
        "--others-lam",
        type=float,
        default=None,
        help="smoothness weight of second pass",
    )
    ap.add_argument(
        "--gc-solver",
        default="bk",
        choices=["bk"],
        help="max-flow solver: bk (Boykov-Kolmogorov via PyMaxflow)",
    )
    ap.add_argument(
        "--ref",
        action="append",
        default=None,
        metavar="X,Y",
        help="static reference point (x, y)",
    )
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    ap.add_argument("--fps", type=float, default=0, help="output fps")
    return ap.parse_args(argv)


class VideoSource:
    """Video file or folder of frames. Yields (original_bgr, processing_bgr)."""

    def __init__(self, path, width=0, start=0, step=1, max_frames=0, fps=25.0):
        self.width, self.start, self.step, self.max = (
            width,
            start,
            step,
            max_frames,
        )
        self.files = None
        if os.path.isdir(path):
            ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            self.files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(ext)
            )
            if not self.files:
                raise IOError(f"no image frames in folder {path!r}")
            self.fps = fps
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f"cannot open video {path!r}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            if start > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def _read(self):
        if self.files is not None:
            for f in self.files[self.start :]:
                yield cv2.imread(f, cv2.IMREAD_COLOR)
        else:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield frame
            self.cap.release()

    def __iter__(self):
        n = 0
        for i, frame in enumerate(self._read()):
            if i % self.step:
                continue
            small = frame
            if self.width and frame.shape[1] != self.width:
                h = int(round(frame.shape[0] * self.width / frame.shape[1] / 2) * 2)
                small = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)
            yield frame, small
            n += 1
            if self.max and n >= self.max:
                break


class VideoSink:
    """H.264 via imageio-ffmpeg when available, otherwise OpenCV mp4v."""

    def __init__(self, path, fps):
        self.path, self.fps, self.w = path, fps, None
        try:
            import imageio.v2 as iio

            self.w = iio.get_writer(
                path,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=2,
                pixelformat="yuv420p",
            )
            self.kind = "imageio"
        except Exception:
            self.kind = "cv2"

    def write(self, bgr):
        if self.kind == "imageio":
            self.w.append_data(bgr[..., ::-1])
        else:
            if self.w is None:
                h, w = bgr.shape[:2]
                self.w = cv2.VideoWriter(
                    self.path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (w, h),
                )
            self.w.write(bgr)

    def close(self):
        if self.w is not None:
            self.w.close() if self.kind == "imageio" else self.w.release()


def load_gt_masks(
    path: str, start: int = 0, step: int = 1, max_frames: int = 0
) -> list[np.ndarray]:
    """Load sorted ground-truth mask images as boolean arrays."""
    ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(ext)
    )
    if not files:
        raise IOError(f"no ground-truth mask images in folder {path!r}")
    masks = []
    for i, f in enumerate(files[start:]):
        if i % step:
            continue
        gt = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        masks.append(gt > 0)
        if max_frames and len(masks) >= max_frames:
            break
    return masks


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute binary Intersection over Union (returns 1.0 if both masks are empty)."""
    if pred.shape != gt.shape:
        pred = (
            cv2.resize(
                pred.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 1.0


def overlay(frame, mask, color=(0, 0, 255), alpha=0.5):
    """Colour fill blended with alpha + 2 px contour."""
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    col = np.zeros_like(out)
    col[:] = color
    out[mask] = (alpha * col[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def render_overlay(bgr, mask, incons=None, refs=None, vmax=2.0, panel=False, alpha=0.5):
    """bgr at output resolution; mask / incons / refs at processing resolution."""
    H, W = bgr.shape[:2]
    sx, sy = W / mask.shape[1], H / mask.shape[0]
    if mask.shape != (H, W):
        mask = (
            cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            > 0
        )
    out = overlay(bgr, mask, alpha=alpha)
    if not panel or incons is None:
        return out
    if refs is not None:
        for x, y in refs[:60]:
            cv2.circle(out, (int(x * sx), int(y * sy)), 2, (255, 255, 0), -1)
    incons = cv2.resize(
        incons.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
    )
    hm = np.clip(incons / vmax, 0, 1)
    hm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.putText(
        hm,
        "rigidity inconsistency",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return np.concatenate([out, hm], 1)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    skip_3d = a.mode in ("auto", "layered", "2d")
    cfg = MODConfig(
        model=a.model,
        gap=a.gap,
        parallax=a.parallax,
        labeling=a.labeling,
        skip_3d=skip_3d,
    )
    if a.gc_lam is not None:
        cfg.gc.lam = a.gc_lam
    if a.no_others or a.mode == "layered":
        cfg.gc.others = False
    if a.others_lam is not None:
        cfg.gc.others_lam = a.others_lam
    if a.rigidity_thr is not None:
        cfg.rigidity_thr = a.rigidity_thr
    if a.rigidity_low is not None:
        cfg.rigidity_low = a.rigidity_low
    if a.misalign_thr is not None:
        cfg.misalign_thr = a.misalign_thr
    if a.ref:
        cfg.ref_points = [tuple(float(v) for v in r.split(",")) for r in a.ref]
    det = MovingObjectDetector(cfg, device=a.device)
    src = VideoSource(
        a.video,
        a.width,
        a.start,
        a.step,
        a.max_frames,
        fps=a.fps or 25.0,
    )
    gt_masks = load_gt_masks(a.gt, a.start, a.step, a.max_frames) if a.gt else None
    fps = a.fps or src.fps / a.step
    ov_sink = VideoSink(os.path.join(a.out, "overlay.mp4"), fps)
    mk_sink = VideoSink(os.path.join(a.out, "mask.mp4"), fps)
    if a.save_masks:
        os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    key = {
        "auto": "mask_layered_gc" if a.labeling == "auto" else "mask",
        "stratified": "mask",
        "3d": "mask_3d_gc" if a.labeling == "auto" else "mask_3d",
        "2d": "mask_2d",
        "layered": ("mask_layered" if a.labeling == "threshold" else "mask_layered_gc"),
    }[a.mode]
    if a.mode != "stratified" and not (a.mode == "auto" and a.labeling == "threshold"):
        cfg.stratify = False
    print(
        f"[MOD] device={det.device}  model={cfg.model}  mode={a.mode}  gap={cfg.gap}",
        flush=True,
    )

    g = cfg.gap
    buf: OrderedDict[int, tuple] = OrderedDict()
    stats, t_next, t0 = [], 0, time.time()

    def process(t, last_idx):
        T = last_idx + 1
        j, k = det.triplet_indices(t, T)
        if j not in buf or k not in buf:
            return False
        res = det.detect_tensors(
            buf[t][1], buf[j][1], buf[k][1], init_key=(j - t, k - t)
        )
        m = res[key]
        orig = buf[t][2]
        mo = (
            cv2.resize(
                m.astype(np.uint8),
                (orig.shape[1], orig.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        ov_sink.write(
            render_overlay(
                orig,
                m,
                res["inconsistency"],
                res["ref_points"],
                panel=a.panel,
                alpha=a.alpha,
            )
        )
        mk_sink.write(np.repeat((mo * 255).astype(np.uint8)[..., None], 3, 2))
        if a.save_masks:
            cv2.imwrite(
                os.path.join(a.out, "masks", f"{t:05d}.png"),
                (mo * 255).astype(np.uint8),
            )
        frame_stat = {
            "frame": t,
            "j": j,
            "k": k,
            "moving_px": int(mo.sum()),
            "level": res.get("level", a.mode),
            "n_refs": int(len(res["ref_points"])),
            "used_rigidity": bool(res["used_rigidity"]),
        }
        if gt_masks is not None and t < len(gt_masks):
            frame_stat["iou"] = round(compute_iou(mo, gt_masks[t]), 4)
        stats.append(frame_stat)
        if t % 10 == 0:
            el = time.time() - t0
            iou_str = f"  IoU={frame_stat['iou']:.3f}" if "iou" in frame_stat else ""
            print(
                f"[MOD] frame {t}  moving_px={int(m.sum())}  refs={len(res['ref_points'])}{iou_str}  "
                f"{el / (t + 1):.2f} s/frame",
                flush=True,
            )
        return True

    last = -1
    for idx, (orig, bgr) in enumerate(src):
        buf[idx] = (bgr, frame_to_tensor(bgr, det.device), orig)
        last = idx
        while (t_next + g <= idx and t_next - g >= 0) or (
            t_next < g and t_next + 2 * g <= idx
        ):
            process(t_next, 10**9)
            t_next += 1
        for old in [i for i in buf if i < t_next - 2 * g]:
            del buf[old]
    while t_next <= last:
        process(t_next, last)
        t_next += 1
    ov_sink.close()
    mk_sink.close()
    el = time.time() - t0
    ious = [s["iou"] for s in stats if "iou" in s]
    mean_iou = float(np.mean(ious)) if ious else None
    summary = {
        "input": os.path.abspath(a.video),
        "frames_processed": len(stats),
        "seconds": el,
        "seconds_per_frame": el / max(len(stats), 1),
        "device": str(det.device),
        "mode": a.mode,
        "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
        "config": asdict(cfg),
        "per_frame": stats,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    miou_msg = f"  mIoU={mean_iou:.4f}" if mean_iou is not None else ""
    print(
        f"[MOD] done: {len(stats)} frames in {el:.1f}s{miou_msg} -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
