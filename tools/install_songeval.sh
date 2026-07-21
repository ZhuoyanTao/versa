#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
SONGEVAL_DIR="$REPO_ROOT/versa_cache/SongEval"
SONGEVAL_REPO="https://github.com/ASLP-lab/SongEval.git"
SONGEVAL_REVISION="848fb2ff3a2a9d64dcb20d46f07238c28abd7add"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m pip install \
    "einops" \
    "muq==0.1.0" \
    "omegaconf>=2.3,<3" \
    "safetensors"

if [ ! -d "$SONGEVAL_DIR" ]; then
    git clone "$SONGEVAL_REPO" "$SONGEVAL_DIR"
elif [ ! -d "$SONGEVAL_DIR/.git" ]; then
    echo "ERROR: $SONGEVAL_DIR exists but is not a Git checkout."
    echo "Move it aside or provide that directory through the metric's model_dir option."
    exit 1
fi

if [ -n "$(git -C "$SONGEVAL_DIR" status --porcelain)" ]; then
    echo "ERROR: $SONGEVAL_DIR has local changes; refusing to change its revision."
    exit 1
fi

if ! git -C "$SONGEVAL_DIR" cat-file -e "$SONGEVAL_REVISION^{commit}"; then
    git -C "$SONGEVAL_DIR" fetch origin main
fi
git -C "$SONGEVAL_DIR" checkout --detach "$SONGEVAL_REVISION"

if [ ! -f "$SONGEVAL_DIR/config.yaml" ]; then
    echo "ERROR: SongEval config.yaml is missing from $SONGEVAL_DIR."
    exit 1
fi

if [ ! -f "$SONGEVAL_DIR/ckpt/model.safetensors" ]; then
    echo "ERROR: SongEval checkpoint is missing from $SONGEVAL_DIR/ckpt."
    echo "Check that the SongEval repository checkout includes ckpt/model.safetensors."
    exit 1
fi

echo "SongEval dependencies and pinned assets ($SONGEVAL_REVISION) are ready."
