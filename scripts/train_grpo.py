"""GRPO (RLVR) on top of a CTC SFT checkpoint. Plan and reasons: RESEARCH_LOG.md, 2026-11-15 "RLVR plan".

    $PY -I scripts/train_grpo.py --init RUNS/sft-r16-plain-word-s0 --run-dir RUNS/grpo-s0
    $PY -I scripts/train_grpo.py --init RUNS/sft-r16-plain-word-s0 --run-dir RUNS/probe --probe 3000

How one step works:
  1. Draw training clips (phone-labeled SFT train set, SSD share as in SFT). Clips on the skip list
     (all guesses perfect last time) are skipped, except for a small re-check chance.
  2. One encoder pass per clip. Sample G paths from the frame odds (temperature T); collapse each path
     (merge repeats, drop blanks) to a phone string.
  3. Reward = -PER against the human phones.
  4. Group advantage = (reward - group mean) / group std. Groups with zero spread are dropped
     (no task signal; RAGEN-2 SNR filtering) and recorded (solved vs same-but-wrong).
  5. Loss = policy-gradient term (PPO-style ratio clip; one update per batch, so the ratio is 1)
     + beta * exact per-frame KL(policy || frozen SFT copy), both over kept groups only.
--probe N: no training. Sample on N train + N val clips and report zero-spread shares and greedy PER.
"""
import argparse
import collections
import json
import math
import os
import random
import secrets
import sys
import time

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402


def args_parser():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--init", required=True, help="SFT run dir; its best.pt (EMA or live, as chosen there) is the start and the KL anchor")
    a("--run-dir", required=True)
    a("--seed", type=int, default=0)
    a("--group", type=int, default=16, help="guesses per clip")
    a("--temperature", type=float, default=1.0)
    a("--beta", type=float, default=0.03, help="KL weight to the frozen SFT copy")
    a("--clip-eps", type=float, default=0.2, help="PPO ratio clip (inactive with one update per batch)")
    a("--lr", type=float, default=2e-5)
    a("--warmup-steps", type=int, default=20)
    a("--grad-clip", type=float, default=1.0)
    a("--batch-seconds", type=float, default=60.0, help="target audio seconds of KEPT groups per step")
    a("--max-batch-seconds", type=float, default=150.0, help="cap on drawn audio when refilling for dropped groups")
    a("--recheck-prob", type=float, default=0.1, help="chance a skip-listed clip is drawn anyway")
    a("--ssd-share", type=float, default=None, help="default: the SFT run's value")
    a("--max-steps", type=int, default=2000, help="cap on steps (epoch mode: stops at whichever comes first)")
    a("--epochs", type=int, default=0,
      help="0: random weighted draws (old mode). N>0: N plain passes, every train clip once per pass, shuffled, no SSD weighting")
    a("--curve-every", type=int, default=200, help="fixed-clip reward curve every N steps (0: off)")
    a("--curve-clips", type=int, default=500, help="fixed clips per split for the curve")
    a("--val-every", type=int, default=250)
    a("--val-limit", type=int, default=0)
    a("--patience", type=int, default=4)
    a("--train-limit", type=int, default=0)
    a("--probe", type=int, default=0)
    a("--mi-clips", type=int, default=500, help="fixed val clips for entropy / mutual information")
    a("--time-limit-min", type=float, default=230.0)
    a("--wandb", default="on", choices=["on", "off"])
    a("--workers", type=int, default=6)
    return p


# ---------------------------------------------------------------- pure pieces (tested in tests/test_grpo.py)


def frame_logits(model, x, mask):
    """Phone logits [B, T, V] and valid frame counts [B] from one encoder pass."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hid = base.wavlm(input_values=x, attention_mask=mask)[0]
    return base.lm_head(base.dropout(hid)), base._get_feat_extract_output_lengths(mask.sum(-1))


def frame_mask(n, T_):
    return torch.arange(T_, device=n.device)[None, :] < n[:, None]


def sample_paths(logits, n, group, temperature, generator=None):
    """G frame paths per clip from the (temperature-scaled) frame odds. Returns [B, G, T] ids;
    frames past n are set to blank."""
    B, T_, V = logits.shape
    probs = torch.softmax(logits.detach().float() / temperature, -1).reshape(B * T_, V)
    s = torch.multinomial(probs, group, replacement=True, generator=generator)
    s = s.reshape(B, T_, group).permute(0, 2, 1).contiguous()
    return s.masked_fill(~frame_mask(n, T_)[:, None, :], 0)


def collapse(paths):
    """[B, G, T] frame ids -> list (B) of lists (G) of phone-id lists. CTC rule: merge repeats, drop blank."""
    prev = torch.nn.functional.pad(paths[..., :-1], (1, 0), value=-1)
    keep = (paths != 0) & (paths != prev)
    p, k = paths.cpu().numpy(), keep.cpu().numpy()
    return [[p[b, g][k[b, g]].tolist() for g in range(p.shape[1])] for b in range(p.shape[0])]


def per(ref, hyp):
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def group_stats(r, eps=1e-6):
    """r [B, G] rewards -> advantages [B, G], keep [B] (has spread), solved [B] (all perfect)."""
    mean, std = r.mean(1, keepdims=True), r.std(1, keepdims=True)
    keep = std[:, 0] > eps
    adv = np.where(keep[:, None], (r - mean) / (std + 1e-4), 0.0)
    solved = (~keep) & (np.abs(r).max(1) < eps)
    return adv, keep, solved


def grpo_loss(logits, n, paths, adv, keep, ref_logits, beta, temperature, clip_eps):
    """Policy-gradient + beta * KL, averaged over kept groups. logits/ref_logits [B,T,V], paths [B,G,T],
    adv [B,G] tensor, keep [B] bool tensor."""
    fm = frame_mask(n, logits.shape[1]).float()                      # [B, T]
    lp = torch.log_softmax(logits.float() / temperature, -1)          # sampling policy
    lp_tok = lp.gather(-1, paths.permute(0, 2, 1)).permute(0, 2, 1)   # [B, G, T]
    ratio = torch.exp(lp_tok - lp_tok.detach())
    a = adv[:, :, None]
    pg = -torch.min(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
    pg_seq = (pg * fm[:, None, :]).sum(-1) / fm.sum(-1, keepdim=True).clamp(min=1)   # [B, G]
    lp1 = torch.log_softmax(logits.float(), -1)
    lq = torch.log_softmax(ref_logits.float(), -1)
    kl_t = (lp1.exp() * (lp1 - lq)).sum(-1)                           # exact KL per frame [B, T]
    kl_clip = (kl_t * fm).sum(-1) / fm.sum(-1).clamp(min=1)           # [B]
    k = keep.float()
    nk = k.sum().clamp(min=1)
    pg_mean = (pg_seq.mean(1) * k).sum() / nk
    kl_mean = (kl_clip * k).sum() / nk
    ent_t = -(lp1.exp() * lp1).sum(-1)
    stats = {"kl": float(kl_mean), "kl_all": float((kl_clip).mean()),
             "entropy": float((ent_t * fm).sum() / fm.sum()),
             "pg": float(pg_mean)}
    return pg_mean + beta * kl_mean, stats


def mutual_info(logits, n):
    """Over frames whose top choice is a phone: H(mean phone odds) - mean H(phone odds). Blank excluded."""
    fm = frame_mask(n, logits.shape[1])
    p = torch.softmax(logits.float(), -1)
    sel = fm & (p.argmax(-1) != 0)
    if not bool(sel.any()):
        return 0.0, 0.0
    q = p[sel][:, 1:]
    q = q / q.sum(-1, keepdim=True)
    h = lambda d: -(d * torch.log(d.clamp(min=1e-12))).sum(-1)  # noqa: E731
    cond = float(h(q).mean())
    return float(h(q.mean(0))) - cond, cond


# ---------------------------------------------------------------- model + data


def load_sft(init_dir, device):
    ck = torch.load(os.path.join(init_dir, "best.pt"), map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**ck["config"])
    model, head, lora = T.build_model(cfg, device)
    whead = T.word_head_params(model)
    use_ema = ck["state"].get("best_is_ema", False)
    src = ck["ema"] if use_ema else ck["params"]
    assert len(src) == len(head + lora + whead), "checkpoint does not match the model"
    for p, v in zip(head + lora + whead, src):
        p.data.copy_(v.to(p.dtype))
    return model, head, lora, whead, ck["config"], use_ema


class Draw:
    """Training clips for step k: a pure function of (seed, k, skip list)."""

    def __init__(self, items, weights, seed):
        self.dur = np.array([it["dur"] for it in items])
        self.w, self.seed = weights, seed

    def __call__(self, k, seconds, skip, recheck):
        rng = np.random.default_rng([self.seed, k])
        pool = max(16, int(4 * seconds / max(self.dur.mean(), 0.5)))
        cand = rng.choice(len(self.dur), size=pool, replace=True, p=self.w)
        cand = [int(c) for c in cand if c not in skip or rng.random() < recheck]
        cand = sorted(set(cand), key=lambda c: self.dur[c])
        start = int(rng.integers(0, max(1, len(cand))))
        batch, longest = [], 0.0
        for j in list(range(start, len(cand))) + list(range(start - 1, -1, -1)):
            l2 = max(longest, self.dur[cand[j]])
            if batch and l2 * (len(batch) + 1) > seconds:
                break
            batch.append(cand[j])
            longest = l2
        return batch


def load_batch(ds, idx, device):
    x, mask, lab, _, _ = T.collate([ds[i] for i in idx])
    return x.to(device), mask.to(device)


def val_subset(val, limit, rng):
    by_tag = collections.defaultdict(list)
    for i, it in enumerate(val):
        by_tag[it["tag"]].append(i)
    per_tag = max(1, limit // len(by_tag))
    return [i for t in sorted(by_tag) for i in rng.sample(by_tag[t], min(per_tag, len(by_tag[t])))]


def batches_by_seconds(ids, items, seconds):
    ids = sorted(ids, key=lambda i: items[i]["dur"])
    out, cur, longest = [], [], 0.0
    for i in ids:
        if cur and max(longest, items[i]["dur"]) * (len(cur) + 1) > seconds:
            out.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, items[i]["dur"])
    return out + ([cur] if cur else [])


def epoch_batches(n_items, items, seconds, seed, epoch, skip, recheck, chunk=512):
    """One plain pass: shuffle all clips, cut into chunks, length-sort inside a chunk (less padding),
    batch by seconds, shuffle batch order. Skip-listed (all-perfect) clips are left out unless re-checked."""
    rng = np.random.default_rng([seed, 1000 + epoch])
    ids = [int(i) for i in rng.permutation(n_items) if int(i) not in skip or rng.random() < recheck]
    out = []
    for c in range(0, len(ids), chunk):
        out += batches_by_seconds(ids[c:c + chunk], items, seconds)
    order = rng.permutation(len(out))
    return [out[j] for j in order], len(ids)


@torch.no_grad()
def probe_set(model, ds, items, ids, args, device):
    """Sample G guesses per clip; zero-spread shares under -PER and 0/1 exact match."""
    c = collections.Counter()
    ref_n = greedy_err = 0
    sample_per, best_err = [], 0
    for b in batches_by_seconds(ids, items, args.batch_seconds):
        x, mask = load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = frame_logits(model, x, mask)
        g, _ = T.greedy(logits.float(), n.tolist())
        hyps = collapse(sample_paths(logits, n, args.group, args.temperature))
        for k, i in enumerate(b):
            ref = items[i]["labels"]
            ref_n += len(ref)
            greedy_err += Levenshtein.distance(ref, g[k])
            r = np.array([[-per(ref, h) for h in hyps[k]]])
            _, keep, solved = group_stats(r)
            exact = [h == ref for h in hyps[k]]
            sample_per.append(-r.mean())
            best_err += min(Levenshtein.distance(ref, h) for h in hyps[k])  # oracle: best of the G guesses
            c["clips"] += 1
            c["zero_spread"] += int(not keep[0])
            c["solved"] += int(solved[0])
            c["same_wrong"] += int(not keep[0] and not solved[0])
            c["exact_zero_spread"] += int(all(exact) or not any(exact))
            c["distinct"] += len({tuple(h) for h in hyps[k]})
    n_ = max(c["clips"], 1)
    return {"clips": c["clips"], "greedy_per": greedy_err / max(ref_n, 1), "mean_sample_per": float(np.mean(sample_per)),
            "best_of_g_per": best_err / max(ref_n, 1),
            "zero_spread_share": c["zero_spread"] / n_, "solved_share": c["solved"] / n_,
            "same_wrong_share": c["same_wrong"] / n_, "exact01_zero_spread_share": c["exact_zero_spread"] / n_,
            "mean_distinct_guesses": c["distinct"] / n_}


@torch.no_grad()
def diagnostics(model, ds, items, ids, args, device):
    """Entropy and mutual information on a fixed val subset, plus a silence check."""
    mis, conds, ents = [], [], []
    for b in batches_by_seconds(ids, items, args.batch_seconds):
        x, mask = load_batch(ds, b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = frame_logits(model, x, mask)
        mi, cond = mutual_info(logits, n)
        lp = torch.log_softmax(logits.float(), -1)
        fm = frame_mask(n, logits.shape[1]).float()
        ents.append(float(((-(lp.exp() * lp).sum(-1)) * fm).sum() / fm.sum()))
        mis.append(mi)
        conds.append(cond)
    sil = torch.zeros(4, 3 * 16000, device=device)
    sm = torch.ones_like(sil, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, n = frame_logits(model, sil, sm)
    g, _ = T.greedy(logits.float(), n.tolist())
    return {"diag/mi_phone": float(np.mean(mis)), "diag/entropy_phone_frames": float(np.mean(conds)),
            "diag/entropy_all_frames": float(np.mean(ents)), "diag/silence_phones": float(np.mean([len(h) for h in g]))}


# ---------------------------------------------------------------- main


def main():
    args = args_parser().parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, head, lora, whead, sft_cfg, init_ema = load_sft(args.init, device)
    model.eval()  # dropout / SpecAugment / layerdrop are 0 in our SFT configs; eval keeps the forward fixed
    for p in whead:
        p.requires_grad_(False)  # word head: not used in RL, saved unchanged
    torch.manual_seed(args.seed)
    cfg = argparse.Namespace(**sft_cfg)
    items, checks = T.load_items(cfg)
    train, val = items["train"], items["val"]
    rng = random.Random(args.seed)
    if args.train_limit:
        train = rng.sample(train, min(args.train_limit, len(train)))
    train_ds, val_ds = T.Clips(train), T.Clips(val)

    if args.probe:
        tr_ids = rng.sample(range(len(train)), min(args.probe, len(train)))
        va_ids = val_subset(val, args.probe, rng)
        out = {"init": args.init, "init_weights": "ema" if init_ema else "live", "group": args.group,
               "temperature": args.temperature,
               "train": probe_set(model, train_ds, train, tr_ids, args, device),
               "val": probe_set(model, val_ds, val, va_ids, args, device)}
        json.dump(out, open(os.path.join(args.run_dir, "probe.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return

    ref_model, _, _, _, _, _ = load_sft(args.init, device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if args.val_limit:
        val = [val[i] for i in val_subset(val, args.val_limit, rng)]
        val_ds = T.Clips(val)
    mi_ids = val_subset(val, args.mi_clips, random.Random(1234))  # same clips in every run
    crng = random.Random(4321)  # fixed-clip curve: the same clips at every point and in every run
    curve_tr = crng.sample(range(len(train)), min(args.curve_clips, len(train)))
    curve_va = val_subset(val, args.curve_clips, crng)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_sampler=batches_by_seconds(range(len(val)), val, 200.0),
                                             collate_fn=T.collate, num_workers=args.workers)
    ssd_share = args.ssd_share if args.ssd_share is not None else sft_cfg["ssd_share"]
    draw = Draw(train, T.clip_weights(train, ssd_share, sft_cfg["child_cap"]), args.seed)

    params = head + lora
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    state = {"step": 0, "best": math.inf, "bad": 0, "wandb_id": None, "slurm_jobs": [], "skip": {},
             "keep_ema": 1.0, "best_is_ema": False, "epoch": 0, "pos": 0, "plan": None, "epoch_clips": []}
    last = os.path.join(args.run_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        for p, v in zip(head + lora + whead, ck["params"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"resumed at step {state['step']}", flush=True)
    if os.environ.get("SLURM_JOB_ID"):
        state["slurm_jobs"].append(os.environ["SLURM_JOB_ID"])

    git, code_sha = T.code_version()
    rl = {**vars(args), "ssd_share": ssd_share, "init_weights": "ema" if init_ema else "live",
          "git_commit": git, "code_sha256": code_sha, "n_train_clips": len(train), "n_val_clips": len(val),
          "slurm_jobs": state["slurm_jobs"], "torch": torch.__version__, **checks}
    config = {**sft_cfg, "run_dir": args.run_dir, "init_run": args.init, "rl": rl}  # SFT keys stay flat: eval_ckpt can load it
    json.dump(config, open(os.path.join(args.run_dir, "config.json"), "w"), indent=1)
    run = None
    if args.wandb == "on":
        import wandb
        state["wandb_id"] = state["wandb_id"] or secrets.token_hex(4)
        name = os.path.basename(args.run_dir.rstrip("/"))
        kw = dict(project=os.environ.get("WANDB_PROJECT", "child-asr-phonemes"), entity=os.environ.get("WANDB_ENTITY"),
                  id=state["wandb_id"], resume="allow", name=name, group=name, job_type="grpo",
                  config={**rl, "sft": {k: v for k, v in sft_cfg.items() if not isinstance(v, (list, dict))}},
                  dir=os.environ.get("WANDB_DIR", args.run_dir))
        try:
            run = wandb.init(**kw)
        except Exception as e:
            print(f"wandb online init failed ({type(e).__name__}); logging offline", flush=True)
            run = wandb.init(mode="offline", **kw)
        run.define_metric("*", step_metric="trainer/step")

    def log(m):
        show = ("rl/reward_kept", "rl/reward_all", "rl/kept_frac", "rl/kl", "val/per/all", "val/select_score",
                "rl/greedy_per_train", "train/grad_norm")
        print(f"STEP {state['step']} " + " ".join(f"{k}={m[k]:.4g}" for k in show if k in m), flush=True)
        with open(os.path.join(args.run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": state["step"], **{k: v for k, v in m.items() if isinstance(v, (int, float))}}) + "\n")
        if run:
            run.log({**m, "trainer/step": state["step"]})

    def save(path):
        torch.save({"params": [p.detach() for p in head + lora + whead], "opt": opt.state_dict(),
                    "ema": [p.detach() for p in head + lora + whead], "state": state, "config": config}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def curve():
        m = {}
        for name, ds_, its, ids in (("train", train_ds, train, curve_tr), ("val", val_ds, val, curve_va)):
            pr = probe_set(model, ds_, its, ids, args, device)
            m.update({f"curve/{name}_reward": -pr["mean_sample_per"], f"curve/{name}_greedy_per": pr["greedy_per"],
                      f"curve/{name}_distinct": pr["mean_distinct_guesses"],
                      f"curve/{name}_best_of_g_per": pr["best_of_g_per"], f"curve/{name}_solved": pr["solved_share"]})
        print(f"CURVE {state['step']} " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in m.items()), flush=True)
        return m

    def run_val():
        m = T.evaluate(model, val_loader, val, device)
        model.eval()  # evaluate() leaves the model in train mode
        m.update(diagnostics(model, val_ds, val, mi_ids, args, device))
        return m

    if state["step"] == 0:
        m = run_val()  # = the SFT start: the line every RL number is compared with
        m.update({k.replace("val/", "sft_start/"): v for k, v in m.items() if k.startswith("val/")})
        log(m)
        state["best"] = m["val/select_score"]
        if args.curve_every:
            m.update(curve())
            log(m)
        save(os.path.join(args.run_dir, "best.pt"))

    t0, clipped = time.time(), collections.deque(maxlen=10)
    while state["step"] < args.max_steps:  # epoch mode also breaks after the last epoch
        if (time.time() - t0) / 60 > args.time_limit_min:
            save(last)
            print("time limit: saved last.pt", flush=True)
            return
        k = state["step"]
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (k + 1) / max(1, args.warmup_steps))
        seconds = min(args.max_batch_seconds, args.batch_seconds / max(state["keep_ema"], 1e-3))
        skip = state["skip"]
        if args.epochs:
            if state["plan"] is None or state["pos"] >= len(state["plan"]):
                if state["plan"] is not None:
                    state["epoch"] += 1
                if state["epoch"] >= args.epochs:
                    break
                state["plan"], used = epoch_batches(len(train), train, seconds, args.seed, state["epoch"], skip,
                                                    args.recheck_prob)
                state["pos"] = 0
                state["epoch_clips"].append(used)
                print(f"EPOCH {state['epoch']} {used} clips ({len(train) - used} skip-listed), "
                      f"{len(state['plan'])} batches", flush=True)
            idx = state["plan"][state["pos"]]
            state["pos"] += 1
        else:
            idx = draw(k, seconds, skip, args.recheck_prob)
        ts = time.time()
        x, mask = load_batch(train_ds, idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, n = frame_logits(model, x, mask)
            with torch.no_grad():
                ref_logits, _ = frame_logits(ref_model, x, mask)
        paths = sample_paths(logits, n, args.group, args.temperature)
        hyps = collapse(paths)
        r = np.array([[-per(train[i]["labels"], h) for h in hyps[b]] for b, i in enumerate(idx)])
        adv, keep, solved = group_stats(r)
        for b, i in enumerate(idx):  # skip list: all-perfect clips; a re-checked clip with spread comes off it
            if solved[b]:
                skip[i] = k
            elif keep[b]:
                skip.pop(i, None)
        loss, st = grpo_loss(logits, n, paths, torch.tensor(adv, device=device, dtype=torch.float32),
                             torch.tensor(keep, device=device), ref_logits, args.beta, args.temperature, args.clip_eps)
        opt.zero_grad(set_to_none=True)
        if keep.any():
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            clipped.append(gn > args.grad_clip)
            opt.step()
        else:
            gn = 0.0
        state["keep_ema"] = 0.9 * state["keep_ema"] + 0.1 * float(keep.mean())
        state["step"] += 1

        g_hyp, _ = T.greedy(logits.detach().float(), n.tolist())
        refs = [train[i]["labels"] for i in idx]
        hl = sum(len(h) for row in hyps for h in row) / len(hyps[0])
        m = {"rl/reward_all": float(r.mean()), "rl/reward_kept": float(r[keep].mean()) if keep.any() else float("nan"),
             "rl/greedy_per_train": sum(Levenshtein.distance(a, b) for a, b in zip(refs, g_hyp)) / max(1, sum(map(len, refs))),
             "rl/kept_frac": float(keep.mean()), "rl/solved_frac": float(solved.mean()),
             "rl/same_wrong_frac": float(((~keep) & (~solved)).mean()), "rl/skip_list_size": len(skip),
             "rl/clips": len(idx), "rl/batch_seconds": float(mask.sum()) / 16000,
             "rl/sample_len_ratio": hl / max(1, sum(map(len, refs))),
             "rl/distinct_guesses": float(np.mean([len({tuple(h) for h in row}) for row in hyps])),
             "rl/kl": st["kl"], "rl/kl_all": st["kl_all"], "rl/entropy": st["entropy"], "rl/pg": st["pg"],
             "train/loss": float(loss), "train/grad_norm": gn, "train/clip_frac": sum(clipped) / max(1, len(clipped)),
             "train/lr": opt.param_groups[0]["lr"], "sys/step_time": time.time() - ts,
             "rl/epoch": state["epoch"] + (state["pos"] / max(1, len(state["plan"])) if args.epochs else 0.0)}
        if args.curve_every and state["step"] % args.curve_every == 0:
            m.update(curve())
        if state["step"] % args.val_every == 0 or state["step"] == args.max_steps:
            m.update(run_val())
            if m["val/select_score"] < state["best"]:
                state["best"], state["bad"] = m["val/select_score"], 0
                save(os.path.join(args.run_dir, "best.pt"))
            else:
                state["bad"] += 1
            m["val/best_select_score"] = state["best"]
            save(last)
            log(m)
            if state["bad"] >= args.patience:
                print("early stop", flush=True)
                break
        elif state["step"] % 10 == 0 or any(k_.startswith("curve/") for k_ in m):
            log(m)
    if args.epochs and state["step"] % args.val_every and state["bad"] < args.patience:  # end of last epoch: one final val
        m = run_val()
        if m["val/select_score"] < state["best"]:
            state["best"] = m["val/select_score"]
            save(os.path.join(args.run_dir, "best.pt"))
        m["val/best_select_score"] = state["best"]
        if args.curve_every:
            m.update(curve())
        log(m)
    save(last)
    open(os.path.join(args.run_dir, "DONE"), "w").write(f"step {state['step']}\n")
    print(f"done at step {state['step']}, best select_score {state['best']:.4f}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
