from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


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
