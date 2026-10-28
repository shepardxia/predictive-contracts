#!/bin/bash
# Submit four-A100 jobs on short-unkillable; each requeues until its stage finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$SCRATCH/predictive-contracts"
mkdir -p "$ROOT/logs"
export SOURCE_ARCHIVE="$ROOT/source-$(date +%Y%m%d-%H%M%S).tar"
tar cf "$SOURCE_ARCHIVE" --exclude=__pycache__ contract slurm
submit() {
    local name=$1 dependency=$2; shift 2
    local deps=()
    if [ -n "$dependency" ]; then deps=(--dependency="afterok:$dependency"); fi
    local id
    id=$(sbatch --parsable --job-name="pc_$name" --output="$ROOT/logs/%x_%j.out" \
        "${deps[@]}" slurm/job.sbatch "$name" "$@")
    printf '%s\t%s\t%s\n' "$name" "$id" "$dependency" >> "$ROOT/jobs.tsv"
    echo "$id"
}
BASE=$(submit base_prepare '' base)
CW=$(submit commitment_warmup "$BASE" commitment_warmup)
HW=$(submit hacking_warmup "$BASE" hacking_warmup)
submit commitment_task_s1 '' task
submit commitment_prediction_s1 "$CW" commitment
submit commitment_agreement_s1 "$CW" commitment --attempt-agreement
submit legibility_task_s1 "$HW" task --init-adapter "$ROOT/runs/hacking_warmup/final"
submit legibility_penalty_s1 "$HW" monitor --reward penalty --penalty 2
submit legibility_reward_s1 "$HW" monitor --reward legibility
cat "$ROOT/jobs.tsv"
