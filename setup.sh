#!/usr/bin/env bash
# Download model weights for popcon-matting-bench.
# Run once before benchmark.py.

set -euo pipefail

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

# MODNet ONNX model (~24MB)
# The pretrained ONNX model must be exported manually from the checkpoint.
# 1. Download checkpoint from: https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
# 2. Clone https://github.com/ZHKKKe/MODNet
# 3. Run: python -m onnx.export_onnx --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt --output-path=pretrained/modnet_photographic_portrait_matting.onnx
# 4. Copy the .onnx file to: models/modnet_photographic_portrait_matting.onnx
MODNET_PATH="$MODELS_DIR/modnet_photographic_portrait_matting.onnx"
if [ -f "$MODNET_PATH" ]; then
    echo "MODNet model found: $MODNET_PATH"
else
    echo "MODNet model NOT found. See setup.sh comments for manual download instructions."
    echo "Benchmark will skip MODNet and run other models."
fi

# RVM model (downloaded via torch.hub on first run of benchmark.py)
echo ""
echo "Note: ViTMatte weights are downloaded automatically by HuggingFace Transformers."
echo "Note: RVM weights are downloaded automatically via torch.hub on first benchmark run."
echo ""
echo "Setup complete. Run: python benchmark.py --samples-dir samples/"
