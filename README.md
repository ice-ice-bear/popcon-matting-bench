# popcon-matting-bench

Benchmark tool for comparing background removal models on LINE animated emoji frames.

Part of the [popcon](https://github.com/ice-ice-bear/popcon) emoji generation pipeline. Popcon generates animated emoji from a single character image, but removing the white background cleanly is the hardest step. LINE requires transparent PNG backgrounds that look clean on both light and dark chat themes.

This repo tests which model produces the best alpha matte for cartoon/illustrated characters on white backgrounds.

## Models Tested

| Model | Type | Size | Trimap? | Description |
|-------|------|------|---------|-------------|
| **rembg** | Segmentation | ~170MB | No | General-purpose background removal (u2net). Current popcon default. |
| **rembg_enhanced** | Segmentation + cleanup | ~170MB | No | rembg with stronger color decontamination, morphological cleanup, and 1px erosion. |
| **MODNet** | Matting | 25MB | No | Trimap-free portrait matting (ONNX). Lightweight, real-time. |
| **ViTMatte** | Matting | ~330MB | Yes (auto) | Vision Transformer matting via HuggingFace. Trimap auto-generated from white-bg threshold. Tested at 3 dilation widths (5px, 10px, 20px). |
| **RVM** | Video matting | 103MB | No | Robust Video Matting with temporal consistency via recurrent architecture. |

## Metrics

### Halo Score
Measures white fringe artifacts at the inner edge of the alpha mask when composited on a black background. Lower is better. A "clean" frame scores < 0.05.

```
Composite RGBA on black (#000) -> find inner alpha edge (3px band)
-> count bright pixels (luminance > 0.85) -> normalize by perimeter
```

### Coverage Ratio
Compares foreground area to the rembg baseline. Values < 0.9 indicate the model is eating character detail (outlines, thin features). Values > 1.0 mean the model preserves more detail than rembg (e.g., motion lines).

## Diagnostic Step

Before running any models, the benchmark visualizes rembg's **raw alpha mask** on black. This diagnoses whether quality issues come from bad masks or bad post-processing (color spill). If the raw mask looks clean, you may not need a new model at all.

## Quick Start

```bash
# Clone
git clone https://github.com/ice-ice-bear/popcon-matting-bench.git
cd popcon-matting-bench

# Install
uv venv && uv pip install -e ".[dev]"

# Download MODNet model (~25MB)
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR" -O /tmp/modnet/
mkdir -p models
cp /tmp/modnet/PretrainedModels/modnet_photographic_portrait_matting.onnx models/

# Add sample frames
# Copy raw white-bg frames from popcon jobs:
# cp -r /tmp/popcon/jobs/<JOB_ID>/frames/<EMOJI_NAME>/raw/ samples/<name>/

# Run benchmark
python benchmark.py --samples-dir samples/

# Run specific models only
python benchmark.py --samples-dir samples/ --models rembg,vitmatte_20

# Run tests
pytest test_halo_score.py -v
```

## Output

```
results/
  summary.csv                          # Per-frame halo scores + coverage ratios
  <emoji>/
    diagnostic/                        # Raw rembg alpha masks on black (Step 0)
      frame_001_raw_alpha.png
      frame_001_raw_on_black.png
    composites/                        # Each model's output composited on black
      frame_001_rembg_on_black.png
      frame_001_modnet_on_black.png
      frame_001_vitmatte_20_on_black.png
      ...
```

## Sample Results

Tested on a cartoon bear character (thick outlines, high contrast):

```
============================================================
SUMMARY
============================================================
  rembg                 clean: 24/24 (100%)  avg_halo: 0.0000  avg_coverage: 1.000  [PASS]
  rembg_enhanced        clean: 24/24 (100%)  avg_halo: 0.0000  avg_coverage: 0.965  [PASS]
  modnet                clean: 23/24 (96%)   avg_halo: 0.0046  avg_coverage: 0.860  [PASS]
  vitmatte_20           clean: 24/24 (100%)  avg_halo: 0.0310  avg_coverage: 1.016  [PASS]
  vitmatte_10           clean: 21/24 (88%)   avg_halo: 0.0389  avg_coverage: 1.015  [PASS]
  vitmatte_5            clean: 16/24 (67%)   avg_halo: 0.0433  avg_coverage: 1.012  [FAIL]
  rvm                   clean: 24/24 (100%)  avg_halo: 0.0000  avg_coverage: 0.630  [PASS]
```

### Key Findings

- **rembg wins** on high-contrast cartoon characters. Zero halos, full detail preservation.
- **ViTMatte (20px dilation)** preserves the most detail (motion lines, effects) with clean edges. Best matting model for this use case.
- **MODNet** eats 14% of foreground. Trained on portrait photos, it strips cartoon outlines.
- **RVM** destroys cartoon content entirely (37% detail loss). Designed for real human video, not illustration.
- **rembg_enhanced** adds no benefit over vanilla rembg on this character, just erodes detail.

### Interpretation

For simple cartoon characters with thick outlines, rembg is already sufficient. The benchmark needs harder test cases (thin lines, pastel colors, motion blur) to find where rembg fails and ViTMatte becomes necessary.

## Project Structure

```
popcon-matting-bench/
  benchmark.py         # Main benchmark script (6 conditions + diagnostic)
  halo_score.py        # Reusable metric module (halo score + coverage ratio)
  test_halo_score.py   # 10 unit tests
  setup.sh             # MODNet download instructions
  pyproject.toml       # Dependencies
  samples/             # Add your own white-bg frames here
  models/              # Model weights (gitignored)
  results/             # Benchmark output (gitignored)
```

## Adding Your Own Samples

Place raw white-background PNG frames in `samples/<emoji_name>/`:

```
samples/
  my_character_wave/
    frame_001.png
    frame_002.png
    ...
  my_character_cry/
    frame_001.png
    ...
```

Frames should be the raw output from video frame extraction, before any background removal. White background, character centered.

## Context

This benchmark was designed during a [/office-hours](https://github.com/garrytan/gstack) session to validate whether replacing rembg with specialized matting models would improve LINE emoji quality. The Codex outside voice correctly predicted that rembg might already be sufficient for high-contrast cartoon characters, and that the diagnostic step (checking raw alpha masks) should come before model swapping.

## License

MIT
