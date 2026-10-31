"""
Feature Extractor for Image Blur Detection.

Extracts 7 low-level sharpness and blur metrics from an image using OpenCV, NumPy,
and scikit-image with optimized memory management for CPU-only systems.

Exact Feature Order:
1. laplacian_variance
2. edge_strength
3. noise
4. brenner_sharpness
5. canny_edge_density
6. fft_high_frequency_ratio
7. blur_effect
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Union, Optional

try:
    from skimage.measure import blur_effect as skimage_blur_effect
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


FEATURE_NAMES = [
    "laplacian_variance",
    "edge_strength",
    "noise",
    "brenner_sharpness",
    "canny_edge_density",
    "fft_high_frequency_ratio",
    "blur_effect"
]


def _calc_laplacian_variance(gray: np.ndarray) -> float:
    """Calculates variance of Laplacian operator (focus measure)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    val = float(lap.var())
    del lap
    return val


def _calc_edge_strength(gray: np.ndarray) -> float:
    """Calculates mean magnitude of Sobel gradient vectors."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_strength = float(mag.mean())
    del gx, gy, mag
    return edge_strength


def _calc_noise(gray: np.ndarray) -> float:
    """Estimates high-frequency sensor noise standard deviation via Gaussian residual."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_image = gray.astype(np.float32) - blur.astype(np.float32)
    noise_val = float(noise_image.std())
    del blur, noise_image
    return noise_val


def _calc_brenner_sharpness(gray: np.ndarray) -> float:
    """Calculates Brenner focus measure based on second-difference along horizontal axis."""
    gray_float = gray.astype(np.float32)
    diff = gray_float[:, 2:] - gray_float[:, :-2]
    brenner_sharpness = float(np.mean(diff ** 2))
    del gray_float, diff
    return brenner_sharpness


def _calc_canny_edge_density(gray: np.ndarray) -> float:
    """Calculates proportion of edge pixels detected by Canny detector."""
    edges = cv2.Canny(gray, 100, 200)
    density = float(np.mean(edges > 0))
    del edges
    return density


def _calc_fft_high_frequency_ratio(gray: np.ndarray, fft_size: int = 512, r_ratio: float = 0.1) -> float:
    """
    Calculates ratio of high-frequency energy to total spectral energy in 2D FFT.
    
    Memory optimization: Resizes temporary grayscale array if image exceeds fft_size,
    preventing large FFT matrix allocations.
    """
    h, w = gray.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > fft_size:
        scale = fft_size / max_dim
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        gray_fft = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_fft = gray

    # Compute 2D Fast Fourier Transform
    fft_raw = np.fft.fft2(gray_fft.astype(np.float32))
    fft_shifted = np.fft.fftshift(fft_raw)
    magnitude = np.abs(fft_shifted)

    rows, cols = gray_fft.shape[:2]
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r = min(crow, ccol) * r_ratio
    high_freq_mask = ((x - ccol) ** 2 + (y - crow) ** 2) > (r ** 2)

    total_energy = float(np.sum(magnitude))
    high_freq_energy = float(np.sum(magnitude[high_freq_mask]))

    # Avoid division by zero
    ratio = float(high_freq_energy / (total_energy + 1e-10))

    # Free memory
    del fft_raw, fft_shifted, magnitude, high_freq_mask
    if gray_fft is not gray:
        del gray_fft

    return ratio


def _fallback_blur_effect(gray: np.ndarray) -> float:
    """
    Fallback implementation of blur effect (reblur comparison metric)
    used if skimage is unavailable or raises an exception.
    """
    h, w = gray.shape[:2]
    # Resize to modest resolution for memory safety if needed
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        g_small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        g_small = gray

    # Blur image with low-pass filter
    blurred = cv2.GaussianBlur(g_small, (11, 11), 0)
    diff = np.abs(g_small.astype(np.float32) - blurred.astype(np.float32))
    # Normalized change
    metric = float(np.mean(diff) / (np.std(g_small.astype(np.float32)) + 1e-5))
    if g_small is not gray:
        del g_small
    del blurred, diff
    return min(1.0, max(0.0, 1.0 - metric))


def _calc_blur_effect(gray: np.ndarray) -> float:
    """
    Calculates blur effect metric using skimage.measure.blur_effect.
    Falls back to reblur comparison metric if skimage fails or is absent.
    """
    if SKIMAGE_AVAILABLE:
        try:
            # blur_effect expects 2D grayscale array
            return float(skimage_blur_effect(gray))
        except Exception:
            return _fallback_blur_effect(gray)
    return _fallback_blur_effect(gray)


def extract_features(image: np.ndarray, fft_size: int = 512, as_dict: bool = False) -> Union[List[float], Dict[str, float]]:
    """
    Extracts all 7 sharpness/blur features from an in-memory image array.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR/RGB image array.
    fft_size : int, optional
        Maximum dimension for FFT calculation (default: 512).
    as_dict : bool, optional
        If True, returns a dictionary keyed by feature names.
        If False, returns a list in exact FEATURE_NAMES order.

    Returns
    -------
    Union[List[float], Dict[str, float]]
        Extracted features.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided to extract_features.")

    # Convert to grayscale if 3-channel
    if len(image.shape) == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        needs_del_gray = True
    else:
        gray = image
        needs_del_gray = False

    # Extract all 7 features in exact order
    lap_var = _calc_laplacian_variance(gray)
    edge_str = _calc_edge_strength(gray)
    noise_val = _calc_noise(gray)
    brenner = _calc_brenner_sharpness(gray)
    canny_dens = _calc_canny_edge_density(gray)
    fft_ratio = _calc_fft_high_frequency_ratio(gray, fft_size=fft_size)
    blur_eff = _calc_blur_effect(gray)

    if needs_del_gray:
        del gray

    features_ordered = [
        lap_var,
        edge_str,
        noise_val,
        brenner,
        canny_dens,
        fft_ratio,
        blur_eff
    ]

    if as_dict:
        return dict(zip(FEATURE_NAMES, features_ordered))
    return features_ordered


def extract_features_from_path(image_path: str, fft_size: int = 512, as_dict: bool = False) -> Union[List[float], Dict[str, float]]:
    """
    Loads an image from disk in grayscale (memory-efficient) and extracts all 7 features.
    
    Parameters
    ----------
    image_path : str
        Path to image file.
    fft_size : int, optional
        Maximum dimension for FFT calculation (default: 512).
    as_dict : bool, optional
        If True, returns a dictionary keyed by feature names.
        If False, returns a list in exact FEATURE_NAMES order.

    Returns
    -------
    Union[List[float], Dict[str, float]]
        Extracted features.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load directly in grayscale to avoid 3x RGB RAM allocation
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to decode image from path: {image_path}")

    try:
        features = extract_features(gray, fft_size=fft_size, as_dict=as_dict)
    finally:
        del gray

    return features
