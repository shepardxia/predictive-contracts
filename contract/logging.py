"""W&B scalars and complete rollout tables, written by rank zero."""
import collections
import json
import math
import os
from pathlib import Path

from transformers import TrainerCallback


def flatten(values, prefix=""):
    """Flatten nested metric names; retain lists and text as values."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def record_rollout(rollout):
    """Keep the exact templated prompt and completion, including spliced turns and thinking."""
    if rollout is None:
        return None

    def recorded(prompts, trainer):
        out = rollout(prompts, trainer)
        tok = trainer.processing_class
        out["prompt_text"] = tok.batch_decode(out["prompt_ids"], skip_special_tokens=False)
        out["completion_text"] = tok.batch_decode(out["completion_ids"], skip_special_tokens=False)
        out["prompt_tokens"] = [len(p) for p in out["prompt_ids"]]
        out["completion_tokens"] = [len(c) for c in out["completion_ids"]]
        if "env_mask" in out:
            out["generated_tokens"] = [sum(m) for m in out["env_mask"]]
        return out

    return recorded


ROLLOUT_FIELDS = ("prompt_text", "completion_text", "prompt_tokens", "completion_tokens", "generated_tokens",
                  "think_forced", "think_tokens", "n_attempts", "n_examples", "forced", "commit_text", "proposal_prompt")


def rollout_rows(prompts, completions, task_id, rewards, extra):
    """One row per episode; custom rollout text takes precedence over the dataset prompt."""
    rows = []
    for i, (prompt, completion, tid, reward) in enumerate(zip(prompts, completions, task_id, rewards)):
        row = {"task_id": tid, "reward": reward,
               "prompt": extra["prompt_text"][i] if "prompt_text" in extra else json.dumps(prompt, ensure_ascii=False),
               "completion": extra["completion_text"][i] if "completion_text" in extra else completion[0]["content"]}
        row.update({k: extra[k][i] for k in ROLLOUT_FIELDS if k in extra and k not in ("prompt_text", "completion_text")})
        rows.append(row)
    return rows


class ResearchLog(TrainerCallback):
    """All batches are retained as immutable W&B table versions and local JSONL rows.

    W&B history is append-only across restarts. `launch` distinguishes replayed batches, while
    `train/global_step` is the chart axis and may rewind to the last training checkpoint.
    """

    def __init__(self, args, logs):
        self.logs, self.run = logs, None
        if int(os.environ.get("RANK", 0)) != 0:
            return
        import wandb

        self.wandb = wandb
        identity = Path(args.out) / "wandb_run.json"
        previous = json.loads(identity.read_text()) if identity.exists() else {}
        self.launch = previous.get("launch", -1) + 1
        self.run = wandb.init(
            project=previous.get("project", os.environ.get("WANDB_PROJECT", "predictive-contracts")),
            entity=previous.get("entity") or os.environ.get("WANDB_ENTITY"),
            name=os.environ.get("WANDB_NAME", Path(args.out).name),
            id=previous.get("id") or os.environ.get("WANDB_RUN_ID"),
            resume="allow", dir=args.out, config=vars(args),
        )
        identity.write_text(json.dumps({"id": self.run.id, "project": self.run.project,
                                        "entity": self.run.entity, "launch": self.launch}) + "\n")
        self.run.define_metric("train/global_step")
        self.run.define_metric("*", step_metric="train/global_step")
        self.run.config.update({**vars(args), "training_gpus": int(os.environ.get("WORLD_SIZE", 1))},
                               allow_val_change=True)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.run is not None:
            self.run.config.update({"trainer": args.to_dict()}, allow_val_change=True)
            self.run.summary["resume_from_step"] = state.global_step

    def log_batch(self, metrics, rows, global_step):
        if self.run is None:
            return
        self.logs["reward_log"].write(json.dumps(metrics) + "\n")
        for i, row in enumerate(rows):
            row.update(call=metrics["call"], global_step=global_step, launch=self.launch, rollout=i)
            self.logs["rollouts"].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.logs["reward_log"].flush()
        self.logs["rollouts"].flush()

        numeric = collections.defaultdict(list)
        flat_rows = [flatten(row) for row in rows]
        for row in flat_rows:
            for key, value in row.items():
                if key not in ("task_id", "call", "global_step", "launch", "rollout") and isinstance(value, (int, float)):
                    numeric[key].append(float(value))
        scalars = flatten(metrics, "research")
        for key, values in numeric.items():
            mean = sum(values) / len(values)
            scalars.update({f"rollout/{key}/mean": mean, f"rollout/{key}/min": min(values),
                            f"rollout/{key}/max": max(values),
                            f"rollout/{key}/std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))})
        scalars.update({"train/global_step": global_step, "research/launch": self.launch, "research/rollouts": len(rows)})
        columns = list(dict.fromkeys(key for row in flat_rows for key in row))
        # Bound each table, never the number of logged rollouts. Every chunk gets its own artifact version.
        for start in range(0, len(rows), 10000):
            data = [[row.get(k) for k in columns] for row in flat_rows[start:start + 10000]]
            table = self.wandb.Table(columns=columns, data=data, allow_mixed_types=True)
            self.run.log({**scalars, "rollouts": table})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.run is not None and logs:
            self.run.log({**flatten(logs, "train"), "train/global_step": state.global_step})

    def on_train_end(self, args, state, control, **kwargs):
        if self.run is not None:
            self.run.finish()
