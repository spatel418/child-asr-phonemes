"""Upload val eval results (eval_val_*.json) to W&B as small "eval" runs, so pasketti-phonetic and the
best-checkpoint scores sit next to the training runs in the same project.

    $PY -I scripts/log_eval_wandb.py                    # every RUNS/*/eval_val_*.json
    $PY -I scripts/log_eval_wandb.py RUNS/pasketti-phonetic/eval_val_plain.json

Numbers only: string fields other than model/audio/ckpt are dropped, hyps files are never read.
The W&B run ID comes from the file path, so re-running updates the same run instead of adding a copy.
"""
import glob
import hashlib
import json
import os
import sys

import wandb

PROJECT = "child-asr-phonemes"
KEEP_STR = {"model", "audio", "ckpt", "weights"}


def main():
    paths = sys.argv[1:] or sorted(glob.glob("RUNS/*/eval_val_*.json"))
    for p in paths:
        d = json.load(open(p))
        run = os.path.basename(os.path.dirname(p))
        tag = os.path.basename(p)[len("eval_val_"):-len(".json")]
        nums = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        cfg = {k: d[k] for k in KEEP_STR if isinstance(d.get(k), str)}
        cfg.update(source_run=run, eval_file=os.path.basename(p))
        rid = "eval-" + hashlib.sha1(os.path.abspath(p).encode()).hexdigest()[:10]
        w = wandb.init(project=PROJECT, id=rid, resume="allow", name=f"{run}:eval_{tag}", group=run,
                       job_type="eval", config=cfg, reinit=True)
        w.log(nums)
        for k, v in nums.items():
            w.summary[k] = v
        w.finish()
        print(f"{p}: {len(nums)} numbers -> {rid}")


if __name__ == "__main__":
    main()
