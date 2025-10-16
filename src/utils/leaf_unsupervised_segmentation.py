
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unsupervised leaf segmentation pipeline (color cues + thresholding + GrabCut + watershed).
Author: ChatGPT
Dependencies: Python 3.8+, numpy, opencv-python
Usage:
  python leaf_unsupervised_segmentation.py --image input.jpg --out out_prefix
It will save:
  - *_V.png           : vegetation likelihood map [0..255]
  - *_mask0.png       : initial binary mask (after threshold + morphology)
  - *_mask_refined.png: refined mask after GrabCut
  - *_instances.png   : instance labels with random colors
  - *_overlay.png     : instance boundaries overlaid on original image
"""

import os
import cv2
import numpy as np
from typing import Tuple

# ---------------------------- Utilities ----------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / (sigma + eps)

def _normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, lbl = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        area = np.sum(lbl == i)
        if area >= min_area:
            out[lbl == i] = 1
    return out

def _rand_colors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = (rng.random((n, 3)) * 255).astype(np.uint8)
    colors[0] = 0  # background
    return colors

# ---------------------------- Step 1: Preprocess ----------------------------

def preprocess(img_bgr: np.ndarray, d: int = 7, sigmaColor: float = 50, sigmaSpace: float = 7) -> np.ndarray:
    """Bilateral filter to denoise while keeping edges."""
    smooth = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return smooth

# ---------------------------- Step 2: Vegetation likelihood ----------------------------

def vegetation_score(img_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Compute a fused vegetation likelihood V in [0,1] from ExG, Lab a*, HSV Hue.
    V = alpha*z(ExG) - beta*z(a*) + gamma*I[Hue in green range]
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32) / 255.0)
    exg = 2*g - r - b  # can be negative

    # Lab a* (green negative)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1]  # OpenCV L*a*b* a in [0..255], 128 is zero
    a = (a - 128.0) / 128.0  # roughly normalize to [-1,1]

    # HSV Hue in [0,180] in OpenCV (0-360 mapped to 0-180)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[..., 0] * 2.0  # back to degrees [0..360)
    S = hsv[..., 1] / 255.0
    hue_green = ((H >= 60) & (H <= 160)).astype(np.float32)
    # Penalize desaturated highlights (low S)
    hue_green *= (S > 0.2).astype(np.float32)

    # z-score features for fusion
    exg_z = _zscore(exg)
    a_neg_z = _zscore(-a)  # more green => larger
    # Fuse with weights
    alpha, beta, gamma = 1.0, 1.0, 0.5
    V = alpha*exg_z + beta*a_neg_z + gamma*hue_green
    V = _normalize01(V)

    dbg = {"exg": exg, "a": a, "hue_green": hue_green, "S": S}
    return V, dbg

# ---------------------------- Step 3: Thresholding to get initial mask ----------------------------

def threshold_V(V: np.ndarray, method: str = "otsu", percentile: float = 90.0) -> np.ndarray:
    V8 = _to_uint8(V)
    if method == "otsu":
        _, th = cv2.threshold(V8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (th > 0).astype(np.uint8)
    elif method == "percentile":
        t = np.percentile(V8, percentile)
        mask = (V8 >= t).astype(np.uint8)
    elif method == "adaptive":  # Sauvola-like using local mean & std
        Vf = V.astype(np.float32)
        win = 31
        mu = cv2.blur(Vf, (win, win))
        mu2 = cv2.blur(Vf*Vf, (win, win))
        var = np.maximum(mu2 - mu*mu, 0.0)
        sigma = np.sqrt(var)
        R = 0.5
        k = 0.34
        t = mu * (1 + k * (sigma / (R + 1e-6) - 1))
        mask = (Vf >= t).astype(np.uint8)
    else:
        raise ValueError("Unknown method")
    return mask

def refine_mask_morph(mask: np.ndarray, close_r: int = 3, open_r: int = 2, min_area_ratio: float = 0.0005) -> np.ndarray:
    h, w = mask.shape[:2]
    area_min = int(min_area_ratio * h * w)
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
    se_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1, 2*open_r+1))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  se_open)
    m = _remove_small_components(m, area_min)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_close)
    return m

# ---------------------------- Step 4: Edge-aware refinement (GrabCut) ----------------------------

def edge_aware_refine(img_bgr: np.ndarray, mask0: np.ndarray, iter_count: int = 2) -> np.ndarray:
    """
    Use GrabCut initialized from mask0.
    Sure FG: eroded mask0; Sure BG: dilated inverse; others are probable.
    """
    h, w = mask0.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    sure_fg = cv2.erode(mask0, se, iterations=1)
    sure_bg = cv2.dilate((1 - mask0), se, iterations=2)
    gc_mask[sure_fg == 1] = cv2.GC_FGD
    gc_mask[sure_bg == 1] = cv2.GC_BGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return refined

# ---------------------------- Step 5: Separate touching leaves (Watershed) ----------------------------

def separate_instances(img_bgr: np.ndarray, mask: np.ndarray, min_area_ratio: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance-transform -> local maxima as markers -> watershed on gradient.
    Returns (label_image, colored_vis)
    """
    h, w = mask.shape[:2]
    area_min = int(min_area_ratio * h * w)
    # Distance transform on binary mask
    dist = cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    # Normalize for visualization / stability
    dist_n = _normalize01(dist)
    # Find local maxima via dilate-equals-max trick
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    local_max = (dist_n == cv2.dilate(dist_n, k)).astype(np.uint8)
    local_max = cv2.morphologyEx(local_max, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # Keep maxima only inside the mask
    markers_mask = (local_max & (mask>0)).astype(np.uint8)
    # Connected components as markers (start from 1)
    num_markers, markers = cv2.connectedComponents(markers_mask)
    if num_markers <= 1:
        # No peaks found, treat entire mask as a single instance
        lbl = (mask>0).astype(np.int32)
        return lbl, colorize_labels(lbl)
    # Prepare gradient image for watershed
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Use morphological gradient (edge) to guide watershed
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # Watershed expects markers int32, background 0
    markers_ws = markers.copy().astype(np.int32)
    # Ensure background is 0, unknown is 0 where mask=0
    markers_ws[mask == 0] = 0
    img_for_ws = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_ws, markers_ws)
    # markers_ws: -1 boundaries, >=1 labels
    lbl = markers_ws.copy()
    lbl[lbl < 0] = 0
    # Remove tiny instances
    out = np.zeros_like(lbl)
    for i in range(1, lbl.max()+1):
        area = np.sum(lbl == i)
        if area >= area_min:
            out[lbl == i] = i
    return out, colorize_labels(out)

def colorize_labels(lbl: np.ndarray) -> np.ndarray:
    n = int(lbl.max()) + 1
    colors = _rand_colors(n, seed=42)
    h, w = lbl.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        vis[lbl == i] = colors[i]
    return vis

def overlay_boundaries(img_bgr: np.ndarray, lbl: np.ndarray, color=(0,255,255)) -> np.ndarray:
    """Draw thin boundaries between labels."""
    boundaries = cv2.morphologyEx((lbl>0).astype(np.uint8), cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    out = img_bgr.copy()
    out[boundaries>0] = color
    return out

# ---------------------------- Pipeline ----------------------------

def run_pipeline(img_bgr: np.ndarray,
                 thr_method: str = "otsu",
                 save_prefix: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = preprocess(img_bgr)
    V, _ = vegetation_score(img)
    mask0 = threshold_V(V, method=thr_method)
    mask1 = refine_mask_morph(mask0)
    mask_ref = edge_aware_refine(img, mask1, iter_count=2)
    labels, labels_vis = separate_instances(img, mask_ref)
    overlay = overlay_boundaries(img_bgr, labels)
    if save_prefix is not None:
        os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
        cv2.imwrite(f"{save_prefix}_V.png", _to_uint8(V))
        cv2.imwrite(f"{save_prefix}_mask0.png", (mask0*255).astype(np.uint8))
        cv2.imwrite(f"{save_prefix}_mask_refined.png", (mask_ref*255).astype(np.uint8))
        cv2.imwrite(f"{save_prefix}_instances.png", labels_vis)
        cv2.imwrite(f"{save_prefix}_overlay.png", overlay)
    return V, mask0, mask1, mask_ref, labels,labels_vis, overlay

# ---------------------------- CLI ----------------------------

# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Unsupervised leaf segmentation")
#     parser.add_argument("--image", type=str, required=True, help="Path to input image")
#     parser.add_argument("--out", type=str, default="result/leaf", help="Output prefix (folder/prefix)")
#     parser.add_argument("--thr", type=str, default="otsu", choices=["otsu","percentile","adaptive"], help="Thresholding method on V")
#     args = parser.parse_args()
#     img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
#     if img_bgr is None:
#         raise FileNotFoundError(f"Cannot read image: {args.image}")
#     V, mask0, labels, overlay = run_pipeline(img_bgr, thr_method=args.thr, save_prefix=args.out)
#     print(f"Saved outputs with prefix: {args.out}_*.png")

# if __name__ == "__main__":
#     main()
