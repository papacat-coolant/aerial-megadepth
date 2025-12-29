#!/bin/bash
#
# Batch process AerialMegaDepth scenes: download, convert, upload to R2, cleanup
#
# Usage:
#   ./batch_process.sh                    # Process scenes 0001-0100
#   ./batch_process.sh 1 50               # Process scenes 0001-0050
#   ./batch_process.sh 10 20              # Process scenes 0010-0020
#   ./batch_process.sh 5 5                # Process only scene 0005

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/work"

# Default range
START=${1:-1}
END=${2:-100}

echo "========================================"
echo "Batch Processing AerialMegaDepth Scenes"
echo "========================================"
echo "Range: $(printf '%04d' $START) - $(printf '%04d' $END)"
echo "Work directory: $WORK_DIR"
echo "========================================"

# Create work directory
mkdir -p "$WORK_DIR"

# Track success/failure
SUCCESS=()
FAILED=()

for i in $(seq $START $END); do
    SCENE=$(printf '%04d' $i)
    echo ""
    echo "========================================"
    echo "Processing scene: $SCENE ($(($i - $START + 1)) of $(($END - $START + 1)))"
    echo "========================================"

    if python "$SCRIPT_DIR/process_and_upload.py" \
        --scene "$SCENE" \
        --work_dir "$WORK_DIR" \
        --cleanup; then
        SUCCESS+=("$SCENE")
        echo "Scene $SCENE completed successfully!"
    else
        FAILED+=("$SCENE")
        echo "Scene $SCENE failed!"
    fi
done

echo ""
echo "========================================"
echo "Batch Processing Complete"
echo "========================================"
echo "Successful: ${#SUCCESS[@]} scenes"
if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "  ${SUCCESS[*]}"
fi
echo ""
echo "Failed: ${#FAILED[@]} scenes"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  ${FAILED[*]}"
fi
echo "========================================"
