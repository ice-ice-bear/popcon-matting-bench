"""Matting model benchmark for popcon emoji pipeline.

Compares background removal approaches on white-bg character frames:
  1. rembg (baseline)
  2. rembg + enhanced cleanup
  3. MODNet (ONNX, trimap-free)
  4. ViTMatte (HuggingFace, trimap-based, 3 dilation widths)
  5. RVM (video-native, temporal consistency)

Usage:
    python benchmark.py --samples-dir samples/ --output-dir results/
"""

import argparse
import csv
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from halo_score import composite_on_dark, compute_coverage_ratio, compute_halo_score

# ─── Diagnostic Step 0 ──────────────────────────────────────────────────────

def run_diagnostic(frame_path: Path, output_dir: Path) -> None:
    """Visualize raw rembg alpha mask BEFORE post-processing.

    This diagnoses whether halos come from bad masks or bad color decontamination.
    """
    from rembg import remove

    with open(frame_path, "rb") as f:
        raw_output = remove(f.read())
    rgba = Image.open(io.BytesIO(raw_output)).convert("RGBA")
    alpha = rgba.split()[3]

    # Save raw alpha as grayscale
    alpha_path = output_dir / f"{frame_path.stem}_raw_alpha.png"
    alpha.save(alpha_path)

    # Composite raw rembg output (no cleanup) on black
    composite = composite_on_dark(rgba)
    composite_path = output_dir / f"{frame_path.stem}_raw_on_black.png"
    composite.save(composite_path)


# ─── Model runners ───────────────────────────────────────────────────────────

def run_rembg(frame_path: Path) -> Image.Image:
    """Run rembg background removal. Returns RGBA image."""
    from rembg import remove

    with open(frame_path, "rb") as f:
        output = remove(f.read())
    return Image.open(io.BytesIO(output)).convert("RGBA")


def run_rembg_enhanced(frame_path: Path) -> Image.Image:
    """Run rembg with enhanced post-processing cleanup.

    Applies stronger color decontamination, wider morphological cleanup,
    and edge-aware alpha refinement compared to popcon's current clean_rembg_output.
    """
    rgba = run_rembg(frame_path)
    r, g, b, a = rgba.split()

    # Step 1: Stronger color decontamination
    a_arr = np.array(a, dtype=np.float64)
    alpha_norm = a_arr / 255.0
    rgb_arr = np.stack([np.array(c, dtype=np.float64) for c in (r, g, b)], axis=-1)

    decontam_mask = alpha_norm > 0.05
    for c in range(3):
        ch = rgb_arr[:, :, c]
        ch[decontam_mask] = np.clip(
            (ch[decontam_mask] - 255.0 * (1.0 - alpha_norm[decontam_mask]))
            / np.maximum(alpha_norm[decontam_mask], 0.01),
            0, 255,
        )

    # Step 2: Wider morphological close
    a_clean = a.filter(ImageFilter.MaxFilter(5))
    a_clean = a_clean.filter(ImageFilter.MinFilter(5))

    # Step 3: Binary alpha with lower threshold
    a_binary = a_clean.point(lambda x: 255 if x > 50 else 0)

    # Step 4: Edge-aware feathering with wider blur
    a_feathered = a_binary.filter(ImageFilter.GaussianBlur(radius=1.0))
    a_feathered = a_feathered.point(lambda x: 0 if x < 3 else (255 if x > 252 else x))

    # Step 5: Erode mask by 1px to push edge inward (removes white fringe)
    a_eroded = a_feathered.filter(ImageFilter.MinFilter(3))

    r_out = Image.fromarray(rgb_arr[:, :, 0].astype(np.uint8))
    g_out = Image.fromarray(rgb_arr[:, :, 1].astype(np.uint8))
    b_out = Image.fromarray(rgb_arr[:, :, 2].astype(np.uint8))

    return Image.merge("RGBA", (r_out, g_out, b_out, a_eroded))


def generate_trimap(frame_path: Path, dilation: int = 10) -> np.ndarray:
    """Auto-generate trimap from white-bg frame.

    White = definite foreground, Black = definite background,
    Gray (128) = unknown transition zone.

    Args:
        frame_path: Path to white-bg RGB frame.
        dilation: Width of unknown zone in pixels.

    Returns:
        Trimap as uint8 ndarray (H, W) with values 0, 128, 255.
    """
    img = cv2.imread(str(frame_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: pixels darker than 240 are likely foreground
    _, fg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Erode for definite foreground
    kernel = np.ones((dilation, dilation), np.uint8)
    definite_fg = cv2.erode(fg_mask, kernel, iterations=1)

    # Dilate for definite background boundary
    definite_bg_boundary = cv2.dilate(fg_mask, kernel, iterations=1)

    # Build trimap
    trimap = np.zeros_like(gray)
    trimap[definite_bg_boundary == 0] = 0  # definite background
    trimap[definite_fg > 0] = 255  # definite foreground
    # Unknown zone: between eroded fg and dilated boundary
    unknown = (definite_bg_boundary > 0) & (definite_fg == 0)
    trimap[unknown] = 128

    # Sanity check: if foreground < 5% of image, warn
    fg_ratio = (trimap == 255).sum() / trimap.size
    if fg_ratio < 0.05:
        print(f"  WARNING: Trimap foreground is only {fg_ratio:.1%} of image. "
              f"Character may be too light for auto-trimap (dilation={dilation}).")

    return trimap


def run_vitmatte(frame_path: Path, dilation: int = 10) -> Image.Image:
    """Run ViTMatte with auto-generated trimap. Returns RGBA image."""
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
    model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

    image = Image.open(frame_path).convert("RGB")
    trimap_arr = generate_trimap(frame_path, dilation=dilation)
    trimap_pil = Image.fromarray(trimap_arr, mode="L")

    inputs = processor(images=image, trimaps=trimap_pil, return_tensors="pt")
    import torch
    with torch.no_grad():
        output = model(**inputs)

    alpha = output.alphas[0, 0].cpu().numpy()
    alpha = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)

    # Combine original RGB with predicted alpha
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba


def run_modnet(frame_path: Path) -> Image.Image | None:
    """Run MODNet ONNX inference. Returns RGBA image or None if model missing."""
    import onnxruntime as ort

    model_path = Path(__file__).parent / "models" / "modnet_photographic_portrait_matting.onnx"
    if not model_path.exists():
        print("  MODNet model not found. Run setup.sh first.")
        return None

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    image = Image.open(frame_path).convert("RGB")
    orig_w, orig_h = image.size

    # MODNet expects 512x512 input
    img_resized = image.resize((512, 512), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # Add batch dim

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    arr = (arr - mean) / std

    result = session.run(None, {input_name: arr})
    matte = result[0][0, 0]  # (512, 512)
    matte = (np.clip(matte, 0, 1) * 255).astype(np.uint8)

    # Resize matte back to original size
    matte_pil = Image.fromarray(matte, mode="L").resize((orig_w, orig_h), Image.BILINEAR)

    rgba = image.convert("RGBA")
    rgba.putalpha(matte_pil)
    return rgba


def run_rvm(frame_paths: list[Path]) -> list[Image.Image] | None:
    """Run RVM on frames via lossless video. Returns list of RGBA images."""
    import torch

    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found. Skipping RVM.")
        return None

    # Reassemble frames into lossless video
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / "input.avi"
        matted_dir = tmp / "matted"
        matted_dir.mkdir()

        # Create lossless video from frames
        # First, copy frames with sequential naming
        for i, fp in enumerate(frame_paths):
            shutil.copy2(fp, tmp / f"frame_{i:03d}.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", "12",
            "-i", str(tmp / "frame_%03d.png"),
            "-c:v", "ffv1",  # lossless codec
            str(video_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Load RVM model
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")

        # Process video frame by frame using RVM's recurrent state
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
        ])

        results = []
        rec = [None] * 4  # recurrent states
        downsample_ratio = 0.25

        for fp in frame_paths:
            frame = Image.open(fp).convert("RGB")
            src = transform(frame).unsqueeze(0)
            if torch.cuda.is_available():
                src = src.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                src = src.to("mps")

            with torch.no_grad():
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

            # Convert alpha to PIL
            alpha = (pha[0, 0].cpu().numpy() * 255).astype(np.uint8)
            alpha_pil = Image.fromarray(alpha, mode="L")

            rgba = frame.convert("RGBA")
            rgba.putalpha(alpha_pil)
            results.append(rgba)

        return results


# ─── Main benchmark ──────────────────────────────────────────────────────────

def benchmark_emoji(
    emoji_dir: Path,
    output_dir: Path,
    run_models: list[str] | None = None,
) -> list[dict]:
    """Run all models on one emoji's frames and return per-frame metrics."""
    raw_frames = sorted(emoji_dir.glob("frame_*.png"))
    if not raw_frames:
        print(f"  No frames found in {emoji_dir}")
        return []

    emoji_name = emoji_dir.name
    emoji_output = output_dir / emoji_name
    diagnostic_dir = emoji_output / "diagnostic"
    composites_dir = emoji_output / "composites"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    composites_dir.mkdir(parents=True, exist_ok=True)

    all_models = run_models or ["rembg", "rembg_enhanced", "modnet", "vitmatte_5", "vitmatte_10", "vitmatte_20", "rvm"]

    # Step 0: Diagnostic — visualize raw rembg output
    print(f"  Step 0: Diagnostic (raw rembg alpha visualization)...")
    for fp in raw_frames[:3]:  # first 3 frames only
        run_diagnostic(fp, diagnostic_dir)

    # Run rembg baseline first (needed for coverage ratio)
    print(f"  Running rembg baseline...")
    rembg_results = {fp.stem: run_rembg(fp) for fp in raw_frames}

    rows = []
    for model_name in all_models:
        print(f"  Running {model_name}...")

        if model_name == "rembg":
            model_results = rembg_results
        elif model_name == "rembg_enhanced":
            model_results = {fp.stem: run_rembg_enhanced(fp) for fp in raw_frames}
        elif model_name == "modnet":
            model_results = {}
            for fp in raw_frames:
                result = run_modnet(fp)
                if result is None:
                    break
                model_results[fp.stem] = result
            if not model_results:
                continue
        elif model_name.startswith("vitmatte_"):
            dilation = int(model_name.split("_")[1])
            model_results = {fp.stem: run_vitmatte(fp, dilation=dilation) for fp in raw_frames}
        elif model_name == "rvm":
            rvm_outputs = run_rvm(raw_frames)
            if rvm_outputs is None:
                continue
            model_results = {fp.stem: out for fp, out in zip(raw_frames, rvm_outputs)}
        else:
            print(f"  Unknown model: {model_name}")
            continue

        for fp in raw_frames:
            frame_name = fp.stem
            if frame_name not in model_results:
                continue
            rgba = model_results[frame_name]

            halo = compute_halo_score(rgba)
            coverage = compute_coverage_ratio(rgba, rembg_results[frame_name])
            clean = halo < 0.05

            # Save composite on black
            comp = composite_on_dark(rgba)
            comp.save(composites_dir / f"{frame_name}_{model_name}_on_black.png")

            rows.append({
                "emoji": emoji_name,
                "frame": frame_name,
                "model": model_name,
                "halo_score": round(halo, 6),
                "coverage_ratio": round(coverage, 4),
                "clean": clean,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Matting model benchmark for popcon emoji pipeline")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"), help="Directory with emoji sample folders")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory for results")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run (default: all)")
    args = parser.parse_args()

    if not args.samples_dir.exists():
        print(f"Samples directory not found: {args.samples_dir}")
        print("Copy raw frame folders from popcon jobs: /tmp/popcon/jobs/<job_id>/frames/<emoji_name>/raw/")
        sys.exit(1)

    emoji_dirs = sorted([d for d in args.samples_dir.iterdir() if d.is_dir()])
    if not emoji_dirs:
        print(f"No emoji directories found in {args.samples_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_models = args.models.split(",") if args.models else None

    print(f"Benchmarking {len(emoji_dirs)} emoji sets...")
    all_rows = []
    for emoji_dir in emoji_dirs:
        print(f"\n[{emoji_dir.name}]")
        rows = benchmark_emoji(emoji_dir, args.output_dir, run_models=run_models)
        all_rows.extend(rows)

    # Write summary CSV
    csv_path = args.output_dir / "summary.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["emoji", "frame", "model", "halo_score", "coverage_ratio", "clean"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to {csv_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        from collections import defaultdict
        by_model = defaultdict(list)
        for row in all_rows:
            by_model[row["model"]].append(row)

        for model, rows in sorted(by_model.items()):
            clean_count = sum(1 for r in rows if r["clean"])
            total = len(rows)
            avg_halo = sum(r["halo_score"] for r in rows) / total
            avg_coverage = sum(r["coverage_ratio"] for r in rows) / total
            pct = clean_count / total * 100
            status = "PASS" if pct >= 80 else "FAIL"
            print(f"  {model:20s}  clean: {clean_count}/{total} ({pct:.0f}%)  "
                  f"avg_halo: {avg_halo:.4f}  avg_coverage: {avg_coverage:.3f}  [{status}]")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
