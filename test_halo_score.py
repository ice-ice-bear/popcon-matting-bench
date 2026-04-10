"""Unit tests for halo_score module."""

import numpy as np
import pytest
from PIL import Image

from halo_score import compute_coverage_ratio, compute_halo_score, composite_on_dark


def _make_rgba(w: int, h: int, rgb: tuple, alpha_arr: np.ndarray) -> Image.Image:
    """Helper: create RGBA image with solid RGB color and given alpha array."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = rgb[0]
    arr[:, :, 1] = rgb[1]
    arr[:, :, 2] = rgb[2]
    arr[:, :, 3] = alpha_arr
    return Image.fromarray(arr, mode="RGBA")


class TestComputeHaloScore:
    def test_clean_edges_low_score(self):
        """Character with clean hard edges on transparent bg should score near 0."""
        alpha = np.zeros((100, 100), dtype=np.uint8)
        # Solid opaque square in center, no semi-transparent fringe
        alpha[20:80, 20:80] = 255
        img = _make_rgba(100, 100, (128, 64, 64), alpha)
        score = compute_halo_score(img)
        assert score < 0.05, f"Clean edges should score < 0.05, got {score}"

    def test_white_halo_high_score(self):
        """Character with white-contaminated edge pixels should score high.

        Real rembg halos: the mask extends slightly beyond the true character.
        Edge pixels have bright RGB (white bg contamination) + semi-transparent
        alpha. On dark bg, these glow as a visible white ring.

        The halo score checks the INNER edge band of the alpha mask:
        pixels that have alpha > 0 but are within edge_band_px of the
        boundary. If those composited pixels are bright, that's the halo.
        """
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        # Solid dark character core
        arr[30:70, 30:70, :3] = 80
        arr[30:70, 30:70, 3] = 255
        # White-contaminated edge ring (semi-transparent, bright RGB)
        # These pixels are INSIDE the alpha mask at its inner edge
        for offset in range(1, 4):
            y_lo, y_hi = 30 - offset, 70 + offset
            x_lo, x_hi = 30 - offset, 70 + offset
            # High alpha + white RGB = visible bright fringe on dark bg
            arr[y_lo, x_lo:x_hi, :3] = 255
            arr[y_lo, x_lo:x_hi, 3] = 220
            arr[y_hi - 1, x_lo:x_hi, :3] = 255
            arr[y_hi - 1, x_lo:x_hi, 3] = 220
            arr[y_lo:y_hi, x_lo, :3] = 255
            arr[y_lo:y_hi, x_lo, 3] = 220
            arr[y_lo:y_hi, x_hi - 1, :3] = 255
            arr[y_lo:y_hi, x_hi - 1, 3] = 220
        img = Image.fromarray(arr, mode="RGBA")
        score = compute_halo_score(img)
        assert score > 0.05, f"White halo should score > 0.05, got {score}"

    def test_fully_transparent_returns_zero(self):
        """Fully transparent image should return 0."""
        alpha = np.zeros((50, 50), dtype=np.uint8)
        img = _make_rgba(50, 50, (255, 255, 255), alpha)
        assert compute_halo_score(img) == 0.0

    def test_fully_opaque_returns_zero(self):
        """Fully opaque image (no alpha edge at all) should return 0."""
        alpha = np.full((50, 50), 255, dtype=np.uint8)
        img = _make_rgba(50, 50, (128, 128, 128), alpha)
        assert compute_halo_score(img) == 0.0

    def test_tiny_image_no_crash(self):
        """Very small image should not crash."""
        alpha = np.zeros((5, 5), dtype=np.uint8)
        alpha[2, 2] = 255
        img = _make_rgba(5, 5, (100, 100, 100), alpha)
        score = compute_halo_score(img)
        assert isinstance(score, float)


class TestComputeCoverageRatio:
    def test_identical_images_ratio_one(self):
        """Same image as model and baseline should give ratio ~1.0."""
        alpha = np.zeros((50, 50), dtype=np.uint8)
        alpha[10:40, 10:40] = 255
        img = _make_rgba(50, 50, (128, 128, 128), alpha)
        assert abs(compute_coverage_ratio(img, img) - 1.0) < 0.01

    def test_smaller_mask_detected(self):
        """Model with smaller mask should give ratio < 1.0."""
        baseline_alpha = np.zeros((50, 50), dtype=np.uint8)
        baseline_alpha[10:40, 10:40] = 255
        baseline = _make_rgba(50, 50, (128, 128, 128), baseline_alpha)

        model_alpha = np.zeros((50, 50), dtype=np.uint8)
        model_alpha[15:35, 15:35] = 255  # smaller
        model = _make_rgba(50, 50, (128, 128, 128), model_alpha)

        ratio = compute_coverage_ratio(model, baseline)
        assert ratio < 0.9, f"Smaller mask should give ratio < 0.9, got {ratio}"

    def test_empty_baseline_returns_zero(self):
        """Empty baseline should return 0.0."""
        empty_alpha = np.zeros((50, 50), dtype=np.uint8)
        empty = _make_rgba(50, 50, (0, 0, 0), empty_alpha)
        model_alpha = np.full((50, 50), 255, dtype=np.uint8)
        model = _make_rgba(50, 50, (128, 128, 128), model_alpha)
        assert compute_coverage_ratio(model, empty) == 0.0


class TestCompositeOnDark:
    def test_output_is_rgb(self):
        """Output should be RGB, not RGBA."""
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        img = _make_rgba(10, 10, (128, 128, 128), alpha)
        result = composite_on_dark(img)
        assert result.mode == "RGB"

    def test_transparent_region_shows_bg(self):
        """Transparent regions should show the background color."""
        alpha = np.zeros((10, 10), dtype=np.uint8)
        img = _make_rgba(10, 10, (255, 255, 255), alpha)
        result = composite_on_dark(img, bg_color=(26, 26, 46))
        pixel = result.getpixel((5, 5))
        assert pixel == (26, 26, 46)
