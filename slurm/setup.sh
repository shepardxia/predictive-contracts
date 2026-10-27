#!/bin/bash
# Prepare the project on Mila's login node; training uses a node-local copy of the packed environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export HF_HOME="$SCRATCH/hf"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
ROOT="$SCRATCH/predictive-contracts"
ENV="$SCRATCH/venvs/predictive-contracts"
mkdir -p "$ROOT"/{data/leetcode,runs,logs,prepared} "$SCRATCH/venvs"
if [ ! -x "$ENV/bin/python" ]; then uv venv --python 3.12 --relocatable "$ENV"; fi
uv pip install --python "$ENV/bin/python" --index-strategy unsafe-best-match -r slurm/requirements.txt
uv pip freeze --python "$ENV/bin/python" > "$ROOT/environment.txt"
tar cf "$ROOT/venv.tar.tmp" -C "$ENV" .
mv "$ROOT/venv.tar.tmp" "$ROOT/venv.tar"

SOURCE=https://raw.githubusercontent.com/ariahw/rl-rewardhacking/73695ff5533b566f7cc99b02bfeb9168936e740d/results/data
for split in leetcode_train_medhard_filtered leetcode_test_medhard; do
    curl -fL --retry 3 "$SOURCE/$split.jsonl" -o "$ROOT/data/leetcode/$split.jsonl"
done
"$ENV/bin/hf" download Qwen/Qwen3-4B
printf 'Prepared %s\n' "$ROOT"
