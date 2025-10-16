"""
Task 1: From-scratch Canny edge detection + fence post counting
----------------------------------------------------------------

- No use of cv2.Canny or other "full task" functions.
- Dependencies: numpy, pillow (PIL) for image I/O and basic handling.
  (You may remove PIL usage if you already have the image as a NumPy array.)

USAGE (example):
----------------
python task1_canny_and_post_count.py --image fence.jpg --low 30 --high 90 --sigma 1.4 --show

This will:
1) Run a from-scratch Canny edge detector.
2) Save intermediate stages and the final edge map next to the input image.
3) Print an estimated count of fence posts.

Notes on the post counting method:
----------------------------------
We use a simple, robust heuristic:
- Compute a column-wise sum of the binary edge map (vertical projection).
- Smooth the 1D signal with a moving average.
- Detect peaks (local maxima) with a minimum distance and prominence.
This works well when posts are approximately vertical and well-separated.
"""

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

# ------------------------------
# Utility functions
# ------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    # Convert RGB/RGBA to grayscale using BT.601 luma
    return (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel normalized to sum 1."""
    assert size % 2 == 1, "Kernel size must be odd."
    ax = np.arange(-(size//2), size//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2*sigma*sigma))
    kernel /= kernel.sum()
    return kernel


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution with zero padding (same output size)."""
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    out = np.zeros_like(img, dtype=np.float32)
    # Flip kernel for convolution
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return out


def sobel_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute image gradients using Sobel kernels."""
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = convolve2d(img, Kx)
    gy = convolve2d(img, Ky)
    return gx, gy


def non_maximum_suppression(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Thin edges by keeping local maxima along gradient direction."""
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float32)
    # Map angles to 0,45,90,135 deg sectors
    angle = np.rad2deg(ang) % 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 0.0
            r = 0.0

            # 0 degrees
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # 45 degrees
            elif (22.5 <= angle[i, j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # 90 degrees
            elif (67.5 <= angle[i, j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # 135 degrees
            elif (112.5 <= angle[i, j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i, j] >= q and mag[i, j] >= r:
                Z[i, j] = mag[i, j]
            else:
                Z[i, j] = 0.0
    return Z


def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Apply double thresholding; output labels: 0=non-edge, 1=weak, 2=strong."""
    strong = (img >= high).astype(np.uint8) * 2
    weak = ((img >= low) & (img < high)).astype(np.uint8)
    return strong + weak


def hysteresis(thresh_img: np.ndarray) -> np.ndarray:
    """Track edges by hysteresis: promote weak pixels connected to strong ones."""
    H, W = thresh_img.shape
    strong_val = 2
    weak_val = 1

    out = np.zeros_like(thresh_img, dtype=np.uint8)
    # Any weak connected (8-neigh) to strong becomes edge=1
    for i in range(1, H-1):
        for j in range(1, W-1):
            if thresh_img[i, j] == strong_val:
                out[i, j] = 1
            elif thresh_img[i, j] == weak_val:
                window = thresh_img[i-1:i+2, j-1:j+2]
                if np.any(window == strong_val):
                    out[i, j] = 1
                else:
                    out[i, j] = 0
    return out


@dataclass
class CannyParams:
    low: float = 30.0
    high: float = 90.0
    sigma: float = 1.4
    gaussian_size: int = 5

def canny(img: np.ndarray, params: CannyParams) -> Tuple[np.ndarray, dict]:
    """Full Canny pipeline; returns binary edges and intermediates."""
    gray = to_grayscale(img)

    # 1) Gaussian blur
    gk = gaussian_kernel(params.gaussian_size, params.sigma)
    smooth = convolve2d(gray, gk)

    # 2) Gradients
    gx, gy = sobel_gradients(smooth)
    mag = np.hypot(gx, gy)
    # Normalize magnitude to [0,255] for thresholding convenience
    mag = mag / (mag.max() + 1e-8) * 255.0
    ang = np.arctan2(gy, gx)

    # 3) Non-maximum suppression
    nms = non_maximum_suppression(mag, ang)

    # 4) Double threshold + hysteresis
    timg = double_threshold(nms, params.low, params.high)
    edges = hysteresis(timg)

    intermediates = {
        "gray": gray,
        "smooth": smooth,
        "grad_x": gx,
        "grad_y": gy,
        "mag": mag,
        "nms": nms,
        "timg": timg
    }
    return edges, intermediates


# ------------------------------
# Post counting (simple heuristic)
# ------------------------------

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    c = np.convolve(xp, np.ones(k, dtype=np.float32)/k, mode="valid")
    return c


def find_peaks(signal: np.ndarray, min_distance: int = 10, prominence: float = 0.1) -> List[int]:
    """Very small peak detector; returns indices of peaks.
    - min_distance: minimum required separation between peaks (pixels)
    - prominence: fraction of (max - min) used as minimal peak height difference
    """
    smin, smax = float(signal.min()), float(signal.max())
    prom = prominence * (smax - smin)
    thresh = smin + prom

    peaks = []
    last_idx = -min_distance
    for i in range(1, len(signal)-1):
        if i - last_idx < min_distance:
            continue
        if signal[i] > signal[i-1] and signal[i] >= signal[i+1] and signal[i] >= thresh:
            peaks.append(i)
            last_idx = i
    return peaks


def count_posts(edge_map: np.ndarray,
                smooth_win: int = 15,
                min_distance: int = 25,
                prominence: float = 0.15) -> Tuple[int, List[int]]:
    """Estimate number of vertical posts in a fence from an edge map.
    Returns the count and the x-indexes of detected posts.
    """
    # Column-wise sum (vertical projection)
    col_strength = edge_map.sum(axis=0).astype(np.float32)
    col_strength_s = moving_average(col_strength, smooth_win)
    peaks = find_peaks(col_strength_s, min_distance=min_distance, prominence=prominence)
    return len(peaks), peaks


# ------------------------------
# CLI for convenience
# ------------------------------

def save_image(arr: np.ndarray, path: str):
    # Normalize to 0..255 for saving
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 1e-8:
        arr = arr / arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
