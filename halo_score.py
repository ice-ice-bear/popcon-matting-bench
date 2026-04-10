"""Halo score computation for matting quality assessment.

Measures white fringe artifacts around character edges when composited
on a dark background. Designed for LINE animated emoji QA where
transparent backgrounds must look clean on both light and dark themes.

This module is designed to be copy-pasteable into the popcon pipeline
for dark-mode preview QA.
"""

import numpy as np
from PIL import Image


def compute_halo_score(rgba_image: Image.Image, edge_band_px: int = 3, luminance_threshold: float = 0.85) -> float:
    """Compute halo score for an RGBA image.

    Composites the image on black, finds the outer alpha edge contour,
    and measures bright pixels within `edge_band_px` of that contour.

    Args:
        rgba_image: PIL Image in RGBA mode.
        edge_band_px: Pixels outward from alpha edge to check. Default 3.
        luminance_threshold: Pixels brighter than this (0-1) are counted
            as halo artifacts. Default 0.85.

    Returns:
        Halo score (float). Lower is better. Clean frame < 0.05.
        Returns 0.0 for fully transparent or fully opaque images.
    """
    img = rgba_image.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    # Find the alpha boundary region: where alpha transitions between
    # fully opaque and fully transparent. Halos live HERE, not outside.
    #
    #   ┌─────────────────────────────┐
    #   │ alpha = 0 (transparent)     │
    #   │   ┌───────────────────┐     │
    #   │   │ EDGE BAND (halo?) │     │  ← We check this ring
    #   │   │ ┌───────────────┐ │     │
    #   │   │ │ alpha = 255   │ │     │
    #   │   │ │ (solid core)  │ │     │
    #   │   │ └───────────────┘ │     │
    #   │   └───────────────────┘     │
    #   └─────────────────────────────┘
    #
    from scipy.ndimage import binary_erosion

    binary_mask = (alpha > 0).astype(np.uint8)

    # No edge if fully transparent or fully opaque
    if binary_mask.sum() == 0 or binary_mask.sum() == binary_mask.size:
        return 0.0

    # Erode the mask to find the solid interior
    eroded = binary_erosion(binary_mask, iterations=edge_band_px).astype(np.uint8)

    # Edge band = original mask minus eroded interior
    # These are the pixels at the INNER edge of the alpha boundary
    edge_band = (binary_mask - eroded).astype(bool)

    edge_band_count = edge_band.sum()
    if edge_band_count == 0:
        return 0.0

    # Composite on black: RGB * (alpha/255)
    rgb = arr[:, :, :3].astype(np.float64)
    alpha_norm = alpha.astype(np.float64) / 255.0
    composited = rgb * alpha_norm[:, :, np.newaxis]

    # Luminance of composited pixels in the edge band
    # Using standard luminance: 0.299R + 0.587G + 0.114B
    luminance = (
        0.299 * composited[:, :, 0]
        + 0.587 * composited[:, :, 1]
        + 0.114 * composited[:, :, 2]
    ) / 255.0

    # Count bright pixels in edge band
    bright_in_band = (luminance[edge_band] > luminance_threshold).sum()

    # Normalize by edge perimeter (approximate as edge_band_count / edge_band_px)
    perimeter_approx = max(1, edge_band_count / edge_band_px)
    score = bright_in_band / perimeter_approx

    return float(score)


def compute_coverage_ratio(rgba_image: Image.Image, baseline_image: Image.Image) -> float:
    """Compute foreground coverage ratio vs a baseline.

    Compares the alpha mask area of two RGBA images. A ratio < 0.9 means
    the model is eating >10% of the foreground detail.

    Args:
        rgba_image: The model's output (RGBA).
        baseline_image: The baseline output (RGBA), typically rembg.

    Returns:
        Ratio of model's foreground area to baseline's foreground area.
        Values < 0.9 indicate potential detail loss.
        Returns 0.0 if baseline has no foreground.
    """
    model_alpha = np.array(rgba_image.convert("RGBA"))[:, :, 3]
    baseline_alpha = np.array(baseline_image.convert("RGBA"))[:, :, 3]

    baseline_fg = (baseline_alpha > 0).sum()
    if baseline_fg == 0:
        return 0.0

    model_fg = (model_alpha > 0).sum()
    return float(model_fg / baseline_fg)


def composite_on_dark(rgba_image: Image.Image, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Composite an RGBA image on a dark background for visual inspection.

    Args:
        rgba_image: PIL Image in RGBA mode.
        bg_color: Background RGB color. Default black (0,0,0).

    Returns:
        RGB PIL Image composited on the dark background.
    """
    bg = Image.new("RGB", rgba_image.size, bg_color)
    bg.paste(rgba_image, mask=rgba_image.split()[3])
    return bg
