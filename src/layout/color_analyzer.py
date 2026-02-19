"""
Color Analyzer
===============
Extracts dominant colors from images using K-Means clustering in LAB space.
Used for background matching in album layouts.
"""

import cv2
import numpy as np
from typing import Tuple, List
from pathlib import Path
from PIL import Image


def extract_dominant_color(image_path: Path, k: int = 3,
                            thumbnail_size: int = 64) -> Tuple[float, float, float]:
    """
    Extract the dominant color from an image using K-Means in LAB space.

    Args:
        image_path: Path to an image file.
        k: Number of color clusters.
        thumbnail_size: Resize to this before clustering (speed).

    Returns:
        Dominant color as (L, A, B) tuple in LAB space.
    """
    # Read as thumbnail for speed
    img = Image.open(image_path).convert("RGB")
    img = img.resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
    rgb = np.array(img)

    # Convert to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    # K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Find the cluster with the most members → dominant
    counts = np.bincount(labels.flatten())
    dominant_idx = counts.argmax()

    return tuple(centers[dominant_idx].tolist())


def compute_average_color(image_paths: List[Path],
                           thumbnail_size: int = 64) -> Tuple[float, float, float]:
    """
    Compute the average LAB color across multiple images.

    Args:
        image_paths: List of image paths.
        thumbnail_size: Resize to this before averaging.

    Returns:
        Average color as (L, A, B) tuple.
    """
    if not image_paths:
        return (255.0, 128.0, 128.0)  # Neutral white in LAB

    lab_sum = np.array([0.0, 0.0, 0.0])
    for p in image_paths:
        try:
            dominant = extract_dominant_color(p, thumbnail_size=thumbnail_size)
            lab_sum += np.array(dominant)
        except Exception:
            continue

    lab_avg = lab_sum / max(len(image_paths), 1)
    return tuple(lab_avg.tolist())


def delta_e(lab1: Tuple[float, float, float],
            lab2: Tuple[float, float, float]) -> float:
    """
    CIE76 ΔE distance between two LAB colors.
    Lower = more similar.
    """
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))
