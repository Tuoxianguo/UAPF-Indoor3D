#!/usr/bin/env bash
# Download the MUSt3R / DUSt3R backbone weights used by the full pipeline.
# The dense-geometry backbone and pretrained weights are provided by NAVER under a
# non-commercial license; please review their terms before downloading.
set -e

WEIGHTS_DIR="${1:-weights}"
mkdir -p "${WEIGHTS_DIR}"

echo "Please obtain the MUSt3R checkpoints from the official repository:"
echo "  https://github.com/naver/must3r"
echo ""
echo "Place the checkpoint (e.g. MUSt3R_512.pth) under: ${WEIGHTS_DIR}/"
echo ""
echo "Then run, for example:"
echo "  python scripts/run_slam.py --chkpt ${WEIGHTS_DIR}/MUSt3R_512.pth \\"
echo "      --input /path/to/images --output ./results --res 512 --use_improved_kf"
