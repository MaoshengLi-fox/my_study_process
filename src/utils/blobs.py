"""
Task 2: Laplacian of Gaussian (LoG) Blob Detection for "flowers.png"
--------------------------------------------------------------------

What this module provides:
1) A from-scratch multi-scale LoG blob detector (no use of OpenCV's blob detector to do the task).
2) Two modes:
   - scale-normalized (sigma^2 * Laplacian)  -> scale-invariant
   - non-normalized (just Laplacian)        -> not scale-invariant
3) Threshold sweep helper to compare detections at different thresholds.
4) Optional comparison with OpenCV's SimpleBlobDetector (for reference only).

Typical Notebook usage:
-----------------------
from task2_log_blobs import (
    detect_blobs_log, draw_blobs, sweep_thresholds, compare_with_opencv
)

img_path = "flowers.png"
blobs = detect_blobs_log(img_path, sigmas=np.linspace(1.0, 6.0, 12),
                         threshold=0.04, normalize_scale=True)
draw_blobs(img_path, blobs, out_path="flowers_log_norm.png")

# Non-normalized baseline:
blobs_non = detect_blobs_log(img_path, sigmas=np.linspace(1.0, 6.0, 12),
                             threshold=0.04, normalize_scale=False)
draw_blobs(img_path, blobs_non, out_path="flowers_log_nonorm.png")

# Threshold sweep
results = sweep_thresholds(img_path, thresholds=[0.02, 0.03, 0.04, 0.06],
                           sigmas=np.linspace(1.0, 6.0, 12))

# (Optional) Comparison with OpenCV (if installed)
compare_with_opencv(img_path, out_path="flowers_opencv_blobs.png")
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2  # for optional comparison only
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ------------------------------
# Basic image utilities
# ------------------------------

def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(imgf: np.ndarray) -> np.ndarray:
    arr = imgf.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


# ------------------------------
# Convolution & kernels
# ------------------------------

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    assert size % 2 == 1
    ax = np.arange(-(size//2), size//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2)/(2*sigma*sigma))
    k /= k.sum()
    return k


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return out


# 2D Laplacian operator (discrete). We'll optionally use LoG via (Laplacian of Gaussian):
LAPLACIAN_KERNEL = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float32)


def laplacian(img: np.ndarray) -> np.ndarray:
    return convolve2d(img, LAPLACIAN_KERNEL)


# ------------------------------
# LoG Multi-scale detection
# ------------------------------

from dataclasses import dataclass

@dataclass
class Blob:
    y: int
    x: int
    sigma: float
    response: float

    @property
    def radius(self) -> float:
        # For LoG, radius ~ sqrt(2) * sigma
        return float(np.sqrt(2.0) * self.sigma)


def log_response(img_gray: np.ndarray, sigma: float, normalize_scale: bool = True) -> np.ndarray:
    """Compute LoG response for one sigma:
        LoG ≈ Laplacian( GaussianBlur(img, sigma) )
    Optionally multiply by sigma^2 for scale normalization.
    """
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    g = gaussian_kernel(ksize, sigma)
    smoothed = convolve2d(img_gray, g)
    lap = laplacian(smoothed)
    if normalize_scale:
        lap = (sigma**2) * lap
    # Take absolute to detect bright/dark blobs; keep positive responses
    return np.abs(lap)


def nms_3d(scale_space: np.ndarray, sigmas: np.ndarray, threshold: float) -> list:
    """Non-maximum suppression in 3D (y, x, scale).
    threshold: relative to max of scale_space (0..1).
    """
    H, W, S = scale_space.shape
    M = scale_space.max()
    if M <= 1e-12:
        return []
    abs_thr = threshold * M

    blobs = []
    for s in range(1, S-1):
        # 3x3x3 neighborhood
        for i in range(1, H-1):
            for j in range(1, W-1):
                val = scale_space[i, j, s]
                if val < abs_thr:
                    continue
                patch = scale_space[i-1:i+2, j-1:j+2, s-1:s+2]
                if val >= np.max(patch):
                    blobs.append(Blob(y=i, x=j, sigma=float(sigmas[s]), response=float(val)))
    return blobs


def detect_blobs_log(
    image_path: str,
    sigmas: np.ndarray,
    threshold: float = 0.03,
    normalize_scale: bool = True
) -> list:
    """Run LoG blob detection over a set of sigmas.
    - threshold: relative to global max response in scale-space (0..1)
    - normalize_scale: True for sigma^2 * Laplacian (scale-invariant), False otherwise
    """
    img = load_gray(image_path)
    H, W = img.shape
    S = len(sigmas)

    # Build scale space (H, W, S)
    ss = np.zeros((H, W, S), dtype=np.float32)
    for k, sigma in enumerate(sigmas):
        ss[..., k] = log_response(img, float(sigma), normalize_scale=normalize_scale)

    blobs = nms_3d(ss, sigmas, threshold)
    return blobs


# ------------------------------
# Utilities: draw / sweep / compare
# ------------------------------

def draw_blobs(image_path: str, blobs: list, out_path: Optional[str] = None) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in blobs:
        r = b.radius
        x0, y0 = b.x - r, b.y - r
        x1, y1 = b.x + r, b.y + r
        draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    if out_path is not None:
        img.save(out_path)
    return img


def sweep_thresholds(
    image_path: str,
    thresholds: list,
    sigmas: np.ndarray,
    normalize_scale: bool = True
) -> dict:
    """Run detections at multiple thresholds; return mapping threshold -> blobs."""
    results = {}
    for t in thresholds:
        blobs = detect_blobs_log(image_path, sigmas=sigmas, threshold=t, normalize_scale=normalize_scale)
        results[t] = blobs
    return results


def compare_with_opencv(image_path: str, out_path: Optional[str] = None):
    """Optional reference: use cv2.SimpleBlobDetector to detect blobs.
    This does NOT replace our LoG method; it's just for comparison.
    If OpenCV is not available, the function reports and returns None.
    """
    if not _HAS_CV2:
        print("OpenCV not installed; skipping comparison.")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image via OpenCV.")
        return None

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10

    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 5000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    # Draw keypoints
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if out_path is not None:
        cv2.imwrite(out_path, vis)

    # Return as a simple list of tuples for parity with Blob
    opencv_blobs = [(int(k.pt[0]), int(k.pt[1]), float(k.size/2.0)) for k in keypoints]
    return opencv_blobs
