#!/bin/bash
# Usage: sbatch slurm/job.sbatch <run-name> <stage> [stage arguments]
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
RUN_NAME=$1; shift
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/predictive-contracts}"
export RUN_ROOT="$SCRATCH/predictive-contracts"
export HF_HOME="$SCRATCH/hf"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hfds"
export WANDB_PROJECT=predictive-contracts
export WANDB_NAME="$RUN_NAME"
export WANDB_CACHE_DIR="$SCRATCH/wandb-cache"
export WANDB_DATA_DIR="$SCRATCH/wandb-data"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export VENV="$SLURM_TMPDIR/venv"
export OUT="$RUN_ROOT/runs/$RUN_NAME"
mkdir -p "$OUT"
printf '%s\n' "$SOURCE_ARCHIVE" > "$OUT/source_archive"
export PATH="$VENV/bin:$PATH"
cd "$PROJECT_ROOT"
# Requeue before the allocation expires; afterok dependencies remain attached to this job ID.
trap 'echo "[allocation] requeuing before time limit" >> "$OUT/run.log"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
printf '[allocation] job=%s restart=%s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" >> "$OUT/run.log"
# Mila gives each Slurm step its own /tmp; stage inside the step that uses it.
srun --gres-flags=allow-task-sharing bash -c '
    set -euo pipefail
    mkdir -p "$VENV" "$SLURM_TMPDIR/code"
    tar xf "$RUN_ROOT/venv.tar" -C "$VENV"
    tar xf "$SOURCE_ARCHIVE" -C "$SLURM_TMPDIR/code"
    cd "$SLURM_TMPDIR/code"
    exec bash slurm/run.sh "$@"
' bash "$@" >> "$OUT/run.log" 2>&1 &
wait $!
