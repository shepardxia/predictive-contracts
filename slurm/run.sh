#!/bin/bash
# Stages share one allocation, one staged environment, and checkpointed outputs on scratch.
set -euo pipefail
STAGE=$1; shift
DATA="$RUN_ROOT/data/leetcode/leetcode_train_medhard_filtered.jsonl"
PREP="$RUN_ROOT/prepared"
PY="$VENV/bin/python"
PORT=$((20000 + SLURM_JOB_ID % 10000))
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=${#devices[@]}
SERVER_PIDS=()
stop_servers() {
    for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do wait "$pid" || true; done
    SERVER_PIDS=()
}
trap stop_servers EXIT

serve() {
    local -a gpu_list urls
    IFS=, read -r -a gpu_list <<< "$1"; shift
    local i port
    for i in "${!gpu_list[@]}"; do
        port=$((PORT + i))
        CUDA_VISIBLE_DEVICES="${gpu_list[i]}" "$PY" -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3-4B --port "$port" --dtype bfloat16 --max-model-len 16384 \
            --tensor-parallel-size 1 --seed "$((1 + i))" --gpu-memory-utilization 0.85 --enable-prefix-caching \
            --reasoning-parser qwen3 --max-num-seqs 128 "$@" > "$OUT/server-$i.log" 2>&1 &
        SERVER_PIDS+=("$!")
        urls+=("http://localhost:$port/v1")
    done
    for i in "${!gpu_list[@]}"; do
        until curl -sf "${urls[i]%/v1}/health" >/dev/null; do
            kill -0 "${SERVER_PIDS[i]}" 2>/dev/null || { tail -30 "$OUT/server-$i.log"; exit 1; }
            sleep 5
        done
    done
    export LLM_BASE_URL="${urls[0]}"
    export LLM_BASE_URLS=$(IFS=,; printf '%s' "${urls[*]}")
    export MONITOR_URL="$LLM_BASE_URL" MONITOR_URLS="$LLM_BASE_URLS"
    export LLM_WORKERS=$((64 * ${#gpu_list[@]}))
}

train() {
    local processes=$1; shift
    if (( processes > 1 )); then
        "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$processes" -m "$@"
    else
        "$PY" -m "$@"
    fi
}

COMMON=(--model Qwen/Qwen3-4B --data "$DATA" --hint modify_tests --thinking --think-budget 4096
        --num-prompts 16 --num-generations 16 --max-completion-length 1536 --max-steps 200
        --save-steps 1 --per-device-batch 8 --prefix-pack --split-normalize --is-mode token_truncate
        --mask-truncated --vllm-sleep --seed 1 --out "$OUT")

case "$STAGE" in
    base)
        serve "$CUDA_VISIBLE_DEVICES"
        "$PY" -m contract.run_tasks --model Qwen/Qwen3-4B --data "$DATA" --hint modify_tests \
            --neutral-system-prompt --thinking --think-budget 4096 --max-tokens 1536 --n 8 --seed 1 \
            --resume --out "$PREP/base_train.jsonl"
        stop_servers
        "$PY" -m contract.probe_features --data "$DATA" --batch-size 4 --out "$PREP/features.npz"
        "$PY" -m contract.probe --features "$PREP/features.npz" --transcripts "$PREP/base_train.jsonl" \
            --behaviors any_hack earns_reward --predict-out "$PREP/targets.json"
        ;;
    commitment_warmup)
        if [ ! -s "$PREP/commitment_traces.jsonl" ]; then
            serve "$CUDA_VISIBLE_DEVICES"
            "$PY" -m contract.commitment.warmup_distill sample --data "$DATA" --targets "$PREP/targets.json" \
                --variant v1 --n 6 --keep 2 --jitter-logit 0.75 --save-all --out "$PREP/commitment_traces.jsonl.tmp"
            mv "$PREP/commitment_traces.jsonl.tmp" "$PREP/commitment_traces.jsonl"
            mv "$PREP/commitment_traces.jsonl.tmp.all.jsonl" "$PREP/commitment_traces.all.jsonl"
            stop_servers
        fi
        train "$NGPUS" contract.commitment.warmup_distill train --traces "$PREP/commitment_traces.jsonl" --variant v1 --out "$OUT"
        ;;
    hacking_warmup)
        if [ ! -s "$PREP/hack_traces.jsonl" ]; then
            serve "$CUDA_VISIBLE_DEVICES"
            "$PY" -m contract.legibility.warmup_hack sample --data "$DATA" --rollouts "$PREP/base_train.jsonl" \
                --out "$PREP/hack_traces.jsonl.tmp"
            mv "$PREP/hack_traces.jsonl.tmp" "$PREP/hack_traces.jsonl"
            stop_servers
        fi
        train "$NGPUS" contract.legibility.warmup_distill train --traces "$PREP/hack_traces.jsonl" \
            --no-mask-hacked-answers --loss-on window+answer --epochs 3 --lr 3e-5 --out "$OUT"
        ;;
    task)
        train "$NGPUS" contract.commitment.train "${COMMON[@]}" --neutral-system-prompt "$@"
        ;;
    commitment)
        train "$NGPUS" contract.commitment.train "${COMMON[@]}" --precommit prob --statements hack_success \
            --commit-max-tokens 128 --decoupled --commit-norm batch --commit-thinking --commit-variant v1 \
            --init-adapter "$RUN_ROOT/runs/commitment_warmup/final" "$@"
        ;;
    monitor)
        policy_gpus=$((NGPUS > 1 ? NGPUS / 2 : 1))
        if (( NGPUS == 1 )); then
            # The frozen 4B monitor shares the card with policy generation and training.
            serve "${devices[0]}" --gpu-memory-utilization 0.20 --max-model-len 8192 --max-num-seqs 16
        else
            monitor_devices=$(IFS=,; printf '%s' "${devices[*]:policy_gpus}")
            serve "$monitor_devices"
        fi
        export CUDA_VISIBLE_DEVICES=$(IFS=,; printf '%s' "${devices[*]:0:policy_gpus}")
        export SLURM_CPUS_PER_TASK=$((SLURM_CPUS_PER_TASK / 2))
        train "$policy_gpus" contract.legibility.train "${COMMON[@]}" --monitor-model Qwen/Qwen3-4B \
            --flag-threshold 0.1 --init-adapter "$RUN_ROOT/runs/hacking_warmup/final" "$@"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
