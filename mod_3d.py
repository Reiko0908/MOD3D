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
  def for_shape(shape):
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
        inside = cv2.erode(inside.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    return warped, inside
