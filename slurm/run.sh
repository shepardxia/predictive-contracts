#!/bin/bash
# Stages share one allocation, one staged environment, and checkpointed outputs on scratch.
set -euo pipefail
STAGE=$1; shift
DATA="$RUN_ROOT/data/leetcode/leetcode_train_medhard_filtered.jsonl"
PREP="$RUN_ROOT/prepared"
PY="$VENV/bin/python"
PORT=$((20000 + SLURM_JOB_ID % 10000))
SERVER_PID=
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT

serve() {
    local devices=$1 tp=$2
    CUDA_VISIBLE_DEVICES="$devices" "$PY" -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-4B --port "$PORT" --dtype bfloat16 --max-model-len 16384 \
        --tensor-parallel-size "$tp" --disable-custom-all-reduce --gpu-memory-utilization 0.85 --enable-prefix-caching \
        --reasoning-parser qwen3 --max-num-seqs 128 > "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    until curl -sf "http://localhost:$PORT/health" >/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$OUT/server.log"; exit 1; }
        sleep 5
    done
    export LLM_BASE_URL="http://localhost:$PORT/v1"
    export MONITOR_URL="$LLM_BASE_URL"
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
        --save-steps 10 --per-device-batch 2 --split-normalize --is-mode token_truncate
        --mask-truncated --vllm-sleep --seed 1 --out "$OUT")

case "$STAGE" in
    base)
        serve "$CUDA_VISIBLE_DEVICES" 2
        "$PY" -m contract.run_tasks --model Qwen/Qwen3-4B --data "$DATA" --hint modify_tests \
            --neutral-system-prompt --thinking --think-budget 4096 --max-tokens 1536 --n 8 --seed 1 \
            --resume --out "$PREP/base_train.jsonl"
        kill "$SERVER_PID"; wait "$SERVER_PID" || true; SERVER_PID=
        "$PY" -m contract.probe_features --data "$DATA" --batch-size 4 --out "$PREP/features.npz"
        "$PY" -m contract.probe --features "$PREP/features.npz" --transcripts "$PREP/base_train.jsonl" \
            --behaviors any_hack earns_reward --predict-out "$PREP/targets.json"
        ;;
    commitment_warmup)
        if [ ! -s "$PREP/commitment_traces.jsonl" ]; then
            serve "$CUDA_VISIBLE_DEVICES" 2
            "$PY" -m contract.commitment.warmup_distill sample --data "$DATA" --targets "$PREP/targets.json" \
                --variant v1 --n 6 --keep 2 --jitter-logit 0.75 --save-all --out "$PREP/commitment_traces.jsonl.tmp"
            mv "$PREP/commitment_traces.jsonl.tmp" "$PREP/commitment_traces.jsonl"
            mv "$PREP/commitment_traces.jsonl.tmp.all.jsonl" "$PREP/commitment_traces.all.jsonl"
            kill "$SERVER_PID"; wait "$SERVER_PID" || true; SERVER_PID=
        fi
        train 2 contract.commitment.warmup_distill train --traces "$PREP/commitment_traces.jsonl" --variant v1 --out "$OUT"
        ;;
    hacking_warmup)
        if [ ! -s "$PREP/hack_traces.jsonl" ]; then
            serve "$CUDA_VISIBLE_DEVICES" 2
            "$PY" -m contract.legibility.warmup_hack sample --data "$DATA" --rollouts "$PREP/base_train.jsonl" \
                --out "$PREP/hack_traces.jsonl.tmp"
            mv "$PREP/hack_traces.jsonl.tmp" "$PREP/hack_traces.jsonl"
            kill "$SERVER_PID"; wait "$SERVER_PID" || true; SERVER_PID=
        fi
        train 2 contract.legibility.warmup_distill train --traces "$PREP/hack_traces.jsonl" \
            --no-mask-hacked-answers --loss-on window+answer --epochs 3 --lr 3e-5 --out "$OUT"
        ;;
    task)
        train 2 contract.commitment.train "${COMMON[@]}" --neutral-system-prompt "$@"
        ;;
    commitment)
        train 2 contract.commitment.train "${COMMON[@]}" --precommit prob --statements hack_success \
            --commit-max-tokens 128 --decoupled --commit-norm batch --commit-thinking --commit-variant v1 \
            --init-adapter "$RUN_ROOT/runs/commitment_warmup/final" "$@"
        ;;
    monitor)
        IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
        serve "${devices[1]}" 1
        export CUDA_VISIBLE_DEVICES="${devices[0]}"
        export SLURM_CPUS_PER_TASK=$((SLURM_CPUS_PER_TASK / 2))
        train 1 contract.legibility.train "${COMMON[@]}" --monitor-model Qwen/Qwen3-4B \
            --flag-threshold 0.1 --init-adapter "$RUN_ROOT/runs/hacking_warmup/final" "$@"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
