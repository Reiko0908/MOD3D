from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# -----------------------------------------------
# Image Preprocessing
# -----------------------------------------------
def to_gray(img):
  """float32 grayscale in [0, 1]"""
  if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if img.dtype == np.uint8:
    img = img.astype(np.float32) / 255.0
  return img.astype(np.float32)

def clean_mask(mask, open_px=3, close_px=7, min_area=40):
  """Remove specks (opening), close small gaps, drop blobs smaller than min_area."""
  m = mask.astype(np.uint8)
  if open_px > 1:
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px,) * 2))
  if close_px > 1:
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px,) * 2))
  n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  keep = np.zeros(n, bool)
  keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
  return keep[lab]

def fill_holes(mask):
  """Fill each blob's interior (an untextured object is only misaligned at its edges)."""
  m = mask.astype(np.uint8)
  cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  out = np.zeros_like(m)
  cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
  return out > 0

# -----------------------------------------------
# 8-parameter quadratic motion model
# -----------------------------------------------

# theta = [a, b, c, d, e, f, g, h]
MODEL_INDICIES = {"translation": [0, 3],
                  "affine": [0, 1, 2, 3, 4, 5],
                  "quadratic": [0, 1, 2, 3, 4, 5, 6, 7]}

@dataclass(frozen=True)
class CoordNorm:
  """Coordinate normalization x_n = (c-cx) / s (conditioning of x^2, xy, y^2)"""
  cx: float
  cy: float
  s: float

  @staticmethod
  def from_shape(shape):
    h,w = shape[:2]
    return CoordNorm((w - 1) / 2.0, (h - 1) / 2.0, max(w, h) / 2.0)

def _flow_jacobian(xn, yn):
  """du/dθ, dv/dθ for the quadratic model, shape (..., 8), used by Gauss-Newton"""
  o, z = np.ones_like(xn), np.zeros_like(xn)
  bu = np.stack([o, xn, yn, z, z, z, xn * xn, xn * yn], -1)
  bv = np.stack([z, z, z, o, xn, yn, xn * yn, yn * yn], -1)
  return bu, bv

def eval_flow(theta, x, y, norm):
  """Flow in pixels at pixel coordinates x, y"""
  a, b, c, d, e, f, g, h = theta
  xn = (np.asarray(x, np.float64) - norm.cx) / norm.s
  yn = (np.asarray(y, np.float64) - norm.cy) / norm.s
  u = a + b * xn + c * yn + g * xn * xn + h * xn * yn
  v = d + e * xn + f * yn + g * xn * yn + h * yn * yn
  return u, v

def model_flow_field(theta, shape, norm, origin_xy=(0, 0)):
  """Dense flow (u, v) on a grid of `shape` whose top-left pixel is `origin`"""
  h, w = shape[:2]
  ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
  u, v = eval_flow(theta, xs + origin_xy[0], ys + origin_xy[1], norm)
  return u.astype(np.float32), v.astype(np.float32)

def backward_warp(img, u, v):
  """out(x) = img(x + flow(x)), and a mask of pixels that landed inside img"""
  h, w = img.shape[:2]
  ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
  mx, my = xs + u, ys + v
  out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
  inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
  return out, inside


def align_frame(other, theta, norm, shape=None, margin=8):
  """'Align' of the description: bring `other` into the reference frame"""
  shape = shape or other.shape[:2]
  u, v = model_flow_field(theta, shape, norm)
  warped, inside = backward_warp(to_gray(other), u, v)
  if margin:
    inside = (
      cv2.erode(inside.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    )
  return warped, inside

# -----------------------------------------------
# Direct Gauss-Newton estimation
# -----------------------------------------------
def _n_levels(shape, min_size=24, max_levels=5):
  """Number of pyramid levels for Gauss-Newton"""
  n, m = 0, min(shape[:2])
  while m / 2 ** (n + 1) >= min_size and n < max_levels:
    n += 1
  return n

def gauss_newton(ref, other, theta0=None, model="quadratic", mask=None, norm=None,
                 origin_xy=(0,0), levels=None, iters=15, init_translation=True):
  """θ such that other(x + flow(x; θ)) ~= ref(x) on mask. origin_xy = (x0, y0) = (column, row) of the crop s top-left pixel."""
  ref, other = to_gray(ref), to_gray(other)
  norm = norm or CoordNorm.from_shape(ref.shape)
  theta = np.zeros(8) if theta0 is None else np.array(theta0, np.float64).copy()
  if init_translation and model != "translation":
    theta = gauss_newton(ref, other, theta, "translation", mask, norm, origin_xy, levels, iters, False)
  idx = MODEL_INDICIES[model]
  if levels is None:
    levels = _n_levels(ref.shape)
  # image pyramid (mask pyramid)
  pr, po = [ref], [other]
  for _ in range(levels):
    pr.append(cv2.pyrDown(pr[-1]))
    po.append(cv2.pyrDown(po[-1]))
  pm = None
  if mask is not None:
    pm = [mask.astype(np.float32)]
    for l in range(1, levels + 1):
      pm.append(cv2.resize(pm[-1], (pr[l].shape[1], pr[l].shape[0]),
        interpolation=cv2.INTER_LINEAR
      ))
  for lvl in range(levels, -1, -1): # coarse -> fine
    sc = float(2 ** lvl)
    R, O = pr[lvl], po[lvl]
    h, w = R.shape
    gx = cv2.Sobel(O, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(O, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xs * sc + origin_xy[0] - norm.cx) / norm.s  # full-image coords
    yn = (ys * sc + origin_xy[1] - norm.cy) / norm.s
    bu, bv = _flow_jacobian(xn.astype(np.float64), yn.astype(np.float64))
    sel = np.ones((h, w), bool) if pm is None else pm[lvl] > 0.5
    sel[:1, :] = sel[-1:, :] = False
    sel[:, :1] = sel[:, -1:] = False
    if sel.sum() < 4 * len(idx): continue
    for it in range(iters):
      u = (bu @ theta) / sc # flow at this level
      v = (bv @ theta) / sc
      mx, my = (xs + u).astype(np.float32), (ys + v).astype(np.float32)
      Iw  = cv2.remap(O, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      gxw = cv2.remap(gx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      gyw = cv2.remap(gy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      act = sel & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
      if act.sum() < 4 * len(idx): break
      e = (Iw - R)[act].astype(np.float64)
      J = (gxw[act][:, None] * bu[act][:, idx] + gyw[act][:, None] * bv[act][:, idx]) / sc
      sigma = max(1.4826 * np.median(np.abs(e)), 2e-3)
      if it < 3 or lvl == levels:
        c = 1.345 * sigma
        wgt = np.minimum(1.0, c / np.maximum(np.abs(e), 1e-12))
      else:
        c = 4.685 * sigma
        r = np.abs(e) / c
        wgt = np.where(r < 1, (1 - r * r) ** 2, 0.0)
      Jw = J * wgt[:, None]
      A = Jw.T @ J
      A += np.eye(len(idx)) * (1e-9 + 1e-6 * np.trace(A) / len(idx))
      delta = np.linalg.solve(A, -Jw.T @ e)
      theta[idx] += delta
      if np.max(np.abs(delta)) < 1e-3 * sc: break
  return theta

def correlation_shift(ref, other):
  a, b = to_gray(ref), to_gray(other)
  win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
  (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
  return dx, dy

def dominant_motion(ref, other, norm=None):
 """Pseudocode step 1-2: dominant quadratic motion ref -> other"""
 norm = norm or CoordNorm.from_shape(ref.shape)
 dx, dy = correlation_shift(ref, other)
 theta0 = np.array([dx, 0, 0, dy, 0, 0, 0, 0], np.float64)
 return gauss_newton(ref, other, theta0, "quadratic", norm=norm)

# -----------------------------------------------
# Misalignment
# -----------------------------------------------
def misalignment_map(ref,warped, win=7, noise=0.01, measure="normal_flow"):
  """normal_flow: residual displacement in pixels (threshold ~1.0)
     intensity: |I_ref - I_warped| in grey levels (threshold ~6)"""
  a = cv2.GaussianBlur(to_gray(ref), (0,0), 1.0)
  b = cv2.GaussianBlur(to_gray(warped), (0,0), 1.0)
  it = b - a
  if measure == "intensity":
    return cv2.boxFilter(np.abs(it), -1, (win, win)) * 255.0
  m = 0.5 * (a + b)
  gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3) / 8.0
  gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3) / 8.0
  num = cv2.boxFilter(it * it, -1, (win, win))
  den = cv2.boxFilter(gx * gx + gy * gy, -1, (win, win))
  return np.sqrt(np.maximum(num, 0) / (np.maximum(den, 0) + (3 * noise) ** 2))
