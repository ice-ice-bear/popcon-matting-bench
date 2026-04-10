#!/usr/bin/env bash
# Download model weights for popcon-matting-bench.
# Run once before benchmark.py.

set -euo pipefail

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

# MODNet ONNX model (~24MB)
MODNET_URL="https://github.com/ZHKKKe/MODNet/raw/master/pretrained/modnet_photographic_portrait_matting.onnx"
MODNET_PATH="$MODELS_DIR/modnet_photographic_portrait_matting.onnx"

if [ -f "$MODNET_PATH" ]; then
    echo "MODNet model already downloaded: $MODNET_PATH"
else
    echo "Downloading MODNet ONNX model..."
    curl -fSL "$MODNET_URL" -o "$MODNET_PATH"
    echo "Downloaded: $MODNET_PATH ($(du -h "$MODNET_PATH" | cut -f1))"
fi

# RVM model (downloaded via torch.hub on first run of benchmark.py)
echo ""
echo "Note: ViTMatte weights are downloaded automatically by HuggingFace Transformers."
echo "Note: RVM weights are downloaded automatically via torch.hub on first benchmark run."
echo ""
echo "Setup complete. Run: python benchmark.py --samples-dir samples/"
