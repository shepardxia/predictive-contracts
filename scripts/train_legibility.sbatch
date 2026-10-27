#!/bin/bash
#
# Legibility training job: the last allocated GPU serves the frozen monitor; the others train.
#   sbatch --time=24:00:00 scripts/train_legibility.sbatch venv_train/bin/python -m contract.legibility.train --reward legibility ...
#   sbatch --gpus=3 scripts/train_legibility.sbatch venv_train/bin/python -m contract.legibility.train --reward legibility ...
#
#SBATCH --job-name=contract_legible
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-low
#SBATCH --requeue   # a pre-empted job goes back in the queue; training resumes from its latest checkpoint (open_logs / train)
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch-oc40/jda/contract/logs/%x_%j.log
#SBATCH --error=/data/scratch-oc40/jda/contract/logs/%x_%j.err
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --open-mode=append

cd /data/scratch-oc40/jda/contract
export HF_HOME=/data/scratch-oc40/jda/contract/hf_cache
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR=/data/scratch-oc40/jda/contract/.uv_cache
if ! venv_train/bin/python -c 'import wandb' >/dev/null 2>&1; then
    bin/uv pip install --python venv_train/bin/python wandb
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))
export VLLM_GROUP_PORT=$((40000 + SLURM_JOB_ID % 10000))
echo "GPUs: $(nvidia-smi -L)"

# Keep SLURM's device IDs (which may be non-contiguous or UUIDs).
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_GPUS=${TRAIN_GPUS:-$((${#GPUS[@]} - 1))}
if (( TRAIN_GPUS < 1 || TRAIN_GPUS >= ${#GPUS[@]} )); then
    echo "Need TRAIN_GPUS training devices plus one monitor device" >&2
    exit 1
fi
TRAIN_DEVICES=$(IFS=,; echo "${GPUS[*]:0:TRAIN_GPUS}")
MONITOR_DEVICE=${GPUS[${#GPUS[@]}-1]}

# the monitor: the serving venv's vLLM, thinking off at request time, same model as the policy's base
MONITOR_MODEL=${MONITOR_MODEL:-Qwen/Qwen3-4B}
PORT=$((20000 + SLURM_JOB_ID % 10000))
CUDA_VISIBLE_DEVICES=$MONITOR_DEVICE venv/bin/vllm serve $MONITOR_MODEL --port $PORT --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --enable-prefix-caching --max-num-seqs 256 > logs/monitor_$SLURM_JOB_ID.log 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
until curl -sf localhost:$PORT/health > /dev/null; do
    sleep 10
    kill -0 $MONITOR_PID 2>/dev/null || { echo "monitor vllm died, see logs/monitor_$SLURM_JOB_ID.log"; exit 1; }
done
export MONITOR_URL=http://localhost:$PORT/v1
echo "monitor ready at $MONITOR_URL"

export CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES
if (( TRAIN_GPUS > 1 )); then
    venv_train/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$TRAIN_GPUS" --no-python "$@"
else
    "$@"
fi
