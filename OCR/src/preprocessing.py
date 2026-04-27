"""
Preprocessing module.

This module prepares a raw input image before it is passed to the OCR engine.
Preprocessing is critical: a clean binary image with good contrast dramatically
improves both text detection and recognition accuracy.

Pipeline:
    raw BGR image -> grayscale -> denoise -> binarization
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a color image to a single-channel grayscale image.

    Grayscale conversion removes color information, reducing the image from
    three channels (B, G, R) to one. This is standard in OCR because text
    recognition depends on luminance contrast, not color.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format (OpenCV default) or already grayscale.

    Returns
    -------
    np.ndarray
        Single-channel grayscale image (uint8).
    """
    if image is None:
        raise ValueError("Input image is None.")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Reduce noise using a median blur filter.

    Median filtering replaces each pixel with the median of its neighborhood.
    It is particularly effective at removing salt-and-pepper noise while
    preserving edges, which is exactly what we need before binarization.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    kernel_size : int
        Size of the median filter kernel. Must be odd. Default: 3.

    Returns
    -------
    np.ndarray
        Denoised grayscale image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def binarize(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert a grayscale image to a black-and-white binary image.

    Binarization separates foreground (text) from background. Two methods
    are supported:

    - "otsu"     : global thresholding using Otsu's method; fast and works
                   well when lighting is uniform.
    - "adaptive" : computes a different threshold for small regions of the
                   image; robust to uneven illumination, recommended for
                   real-world photos (e.g. a blind user's camera).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    method : str
        Either "otsu" or "adaptive".

    Returns
    -------
    np.ndarray
        Binary image with values in {0, 255}.
    """
    if method == "otsu":
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    return cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def preprocess(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Full preprocessing pipeline: grayscale -> denoise -> binarize.

    This is the function called by the detection and recognition modules.
    Returning a clean binary image ensures Tesseract operates on a
    well-defined foreground/background separation.

    Parameters
    ----------
    image : np.ndarray
        Raw BGR input image.
    method : str
        Binarization method ("adaptive" or "otsu").

    Returns
    -------
    np.ndarray
        Preprocessed binary image, ready for OCR.
    """
    gray = to_grayscale(image)
    clean = denoise(gray, kernel_size=3)
    binary = binarize(clean, method=method)
    return binary
