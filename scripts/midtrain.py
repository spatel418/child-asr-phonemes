"""Mid-training: continued self-supervised training of WavLM-Large on unlabeled child audio (+15% adult).
Design and decisions: MIDTRAINING.md. No labels are read except to skip nothing: IPA/words are ignored.

    $PY -I scripts/midtrain.py --build-pool                         # CPU: pool manifest + leak audit
    $PY -I scripts/midtrain.py --kmeans  --out RUNS/midtrain-v1     # GPU: k-means targets
    $PY -I scripts/midtrain.py --train   --out RUNS/midtrain-v1     # GPU: masked prediction (resumable)

Pool (train children only, or corpora in no eval split):
  SFT train clips (prov2 recipe: all train corpora + Providence age 2), word-only clips (MyST train part only),
  unlabeled CHI turns + adult turns from train children's CHAT sessions, SEED mid-training half, CSLU Kids.
Method (HuBERT-style):
  targets = k-means (K=500) on the output of transformer layer 12 of the ORIGINAL WavLM-Large (frozen teacher,
  run on the clean input every step); student = WavLM-Large, CNN frozen, spans of frames masked
  (HF mask_prob 0.65, span 10: about half the frames); a linear head predicts the cluster of masked frames.
  Head-only warmup, then the whole transformer at a low LR (cosine decay). Clips cropped to 15 s.
Output: <out>/wavlm/ (HF save_pretrained of the student backbone) for SFT --base-weights; metrics.jsonl.
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402

POOL = os.path.join(T.MANIFESTS, "midtrain", "pool.jsonl")
MID_INDEX = os.path.join(T.DATA, "processed", "midtrain_index")
CSLU = os.path.join(T.DATA, "raw", "CSLU_Kids", "cslu_kids")
SFT_CFG = "RUNS/sft-r16-plain-word-s0-prov2/config.json"
CROP, WIN, MIN_S = 15.0, 15.0, 0.5
K, LAYER = 500, 12


# ---------------------------------------------------------------- pool

def heldout(who):
    return int(hashlib.sha1(str(who).encode()).hexdigest(), 16) % 100 == 0  # 1% of speakers, for monitoring


def build_pool():
    cfg = argparse.Namespace(**json.load(open(SFT_CFG)))
    split_dir = os.path.join(T.DATA, "splits", "phonetic", "v2")
    split_of = {c: s for s in ("train", "val", "testA", "testB", "testC")
                for c in open(os.path.join(split_dir, f"{s}.txt")).read().split()}
    seed_mid = set(open(os.path.join(T.DATA, "splits", "seed", "midtrain.txt")).read().split())
    seed_test = set(open(os.path.join(T.DATA, "splits", "seed", "test.txt")).read().split())
    rows = []

    def add(path, dur, kind, source, who, start=None, end=None):
        if dur >= MIN_S:
            rows.append({"path": path, "start": start, "end": end, "dur": round(min(dur, CROP), 3), "full_dur": round(dur, 3),
                         "kind": kind, "source": source, "who": who, "heldout": heldout(who)})

    items, _ = T.load_items(cfg)  # train + val; only train is used
    for it in items["train"]:
        assert split_of.get(it["child"]) == "train"
        add(it["path"], it["dur"], "child", f"sft:{it['corpus']}", it["child"])
    for it in T.load_word_items(cfg):
        assert it["child"] not in split_of, "word-corpus child in a phonetic split"
        add(it["path"], it["dur"], "child", f"words:{it['corpus']}", it["child"])
    for p in sorted(glob.glob(os.path.join(MID_INDEX, "shard-*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if r["corpus"] == "SEED":
                assert r["who"] in seed_mid and r["who"] not in seed_test
            else:
                assert split_of.get(r["who"]) == "train", "turn from a non-train session"
            add(r["path"], r["duration"], "adult" if r["kind"] == "adult" else "child",
                f"{'seed' if r['corpus'] == 'SEED' else 'turns'}:{r['corpus']}:{r['kind']}", r["who"])
    # CSLU Kids: scripted (drop verify code 3 = target word not said) + spontaneous cut into 15 s windows
    for p in sorted(glob.glob(os.path.join(CSLU, "speech", "scripted", "**", "*.wav"), recursive=True)):
        v = p.replace(os.sep + "speech" + os.sep, os.sep + "verify" + os.sep)[:-4] + ".txt"
        if os.path.exists(v) and open(v).read().strip()[:1] == "3":
            continue
        add(p, sf.info(p).duration, "child", "cslu:scripted", "cslu_" + os.path.basename(os.path.dirname(p)))
    for p in sorted(glob.glob(os.path.join(CSLU, "speech", "spontaneous", "**", "*.wav"), recursive=True)):
        d = sf.info(p).duration
        for s in np.arange(0, d, WIN):
            e = min(s + WIN, d)
            if e - s >= 2.0:
                add(p, e - s, "child", "cslu:spontaneous", "cslu_" + os.path.basename(os.path.dirname(p)),
                    round(float(s), 3), round(float(e), 3))
    os.makedirs(os.path.dirname(POOL), exist_ok=True)
    with open(POOL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.chmod(POOL, 0o640)
    h = collections.Counter()
    for r in rows:
        h[(r["kind"], r["source"])] += r["dur"] / 3600
    summ = {"rows": len(rows), "hours_cropped": {f"{k} | {s}": round(v, 1) for (k, s), v in sorted(h.items())},
            "child_h": round(sum(v for (k, _), v in h.items() if k == "child"), 1),
            "adult_h": round(sum(v for (k, _), v in h.items() if k == "adult"), 1),
            "heldout_rows": sum(r["heldout"] for r in rows),
            "audit": "every child-corpus row is a train child; SEED rows are mid-training-half speakers; "
                     "word corpora children are in no phonetic split (asserted while building)"}
    json.dump(summ, open(os.path.join(os.path.dirname(POOL), "pool_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def is_heldout(r):
    """Children: 1% of speakers. Adults: 1% of clips (adult turns come from only a few sessions, so a
    speaker-level hold-out would leave no adult monitor clips)."""
    return heldout(r["path"] if r["kind"] == "adult" else r["who"])


def load_pool(heldout_set=False):
    return [r for r in map(json.loads, open(POOL)) if is_heldout(r) == heldout_set]


class PoolClips(torch.utils.data.Dataset):
    def __init__(self, rows, crop_seed):
        self.rows, self.crop_seed = rows, crop_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["start"] is None:
            y, sr = sf.read(r["path"], dtype="float32", always_2d=True)
        else:
            sr = sf.info(r["path"]).samplerate
            y, sr = sf.read(r["path"], start=int(r["start"] * sr), stop=int(r["end"] * sr), dtype="float32", always_2d=True)
        assert sr == 16000, r["path"]
        y = y.mean(axis=1)
        n = int(CROP * sr)
        if len(y) > n:
            rng = np.random.default_rng([self.crop_seed, i, len(y)])
            a = int(rng.integers(0, len(y) - n))
            y = y[a:a + n]
        y = (y - y.mean()) / (y.std() + 1e-7)
        return torch.from_numpy(y), i


def collate(batch):
    n = max(len(b[0]) for b in batch)
    x = torch.zeros(len(batch), n)
    m = torch.zeros(len(batch), n, dtype=torch.long)
    for k, (w, _) in enumerate(batch):
        x[k, : len(w)] = w
        m[k, : len(w)] = 1
    return x, m, [b[1] for b in batch]


def mix_weights(rows, adult_share):
    """Draw probabilities so that adult audio is `adult_share` of the drawn seconds, uniform per clip within a kind."""
    d = np.array([r["dur"] for r in rows])
    a = np.array([r["kind"] == "adult" for r in rows])
    p = np.where(a, adult_share / d[a].sum(), (1 - adult_share) / d[~a].sum())
    return p / p.sum()


# ---------------------------------------------------------------- models

def wavlm(device, trainable):
    from huggingface_hub import snapshot_download
    from transformers import WavLMModel
    path = snapshot_download(T.BASE_MODEL, revision=T.BASE_REVISION)
    kw = dict(mask_time_prob=0.05, mask_feature_prob=0.0, apply_spec_augment=True,  # 0.05 only so the
              # pretrained mask embedding loads; random masking never runs (mask_time_indices is always passed)
              layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0)
    m = WavLMModel.from_pretrained(path, **kw).to(device)
    m.freeze_feature_encoder()
    if not trainable:
        m.encoder.layers = m.encoder.layers[:LAYER]  # only what the targets need
        m.eval().requires_grad_(False)
    return m


class Teacher:
    """Output of transformer layer LAYER of the original WavLM-Large (forward hook, no final layer norm)."""

    def __init__(self, device):
        self.m = wavlm(device, trainable=False)
        self.out = None
        self.m.encoder.layers[LAYER - 1].register_forward_hook(lambda mod, inp, out: setattr(self, "out", out[0]))

    @torch.no_grad()
    def __call__(self, x, mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.m(x, attention_mask=mask)
        return self.out.float()


def frame_mask(model, mask, n_frames):
    return model._get_feature_vector_attention_mask(n_frames, mask).bool()


@torch.no_grad()
def assign(feats, cent):
    """Nearest centroid. feats [N, D], cent [K, D]."""
    d = (feats * feats).sum(1, keepdim=True) - 2 * feats @ cent.T + (cent * cent).sum(1)[None]
    return d.argmin(1)


# ---------------------------------------------------------------- k-means

def kmeans(a, device):
    rows = load_pool(False)
    rng = np.random.default_rng(a.seed)
    p = mix_weights(rows, a.adult_share)
    target, got, ids = a.kmeans_hours * 3600, 0.0, []
    for i in rng.choice(len(rows), size=len(rows), replace=False, p=p):
        ids.append(int(i))
        got += rows[i]["dur"]
        if got >= target:
            break
    teacher = Teacher(device)
    ds = PoolClips(rows, a.seed)
    feats = []
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    batches = [ids[j:j + 16] for j in range(0, len(ids), 16)]
    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=6)
    for x, m, _ in dl:
        x, m = x.to(device), m.to(device)
        h = teacher(x, m)
        fm = frame_mask(teacher.m, m, h.shape[1])
        f = h[fm]
        keep = torch.rand(len(f), generator=gen) < a.frame_keep
        feats.append(f[keep.to(device)].cpu())
    X = torch.cat(feats).to(device)
    print(f"k-means on {len(X)} frames from {len(ids)} clips ({got / 3600:.1f} h)", flush=True)
    cent = X[torch.randperm(len(X), generator=gen)[:K].to(device)].clone()
    for it in range(a.kmeans_iters):
        lab = torch.cat([assign(X[j:j + 200000], cent) for j in range(0, len(X), 200000)])
        cnt = torch.bincount(lab, minlength=K).float()
        s = torch.zeros_like(cent).index_add_(0, lab, X)
        empty = cnt == 0
        cent = torch.where(empty[:, None], X[torch.randint(len(X), (K,), generator=gen).to(device)], s / cnt.clamp(min=1)[:, None])
        inertia = ((X[:200000] - cent[lab[:200000]]) ** 2).sum(1).mean().item()
        print(f"iter {it}: empty {int(empty.sum())}, inertia(first 200k) {inertia:.2f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"centroids": cent.cpu(), "layer": LAYER, "k": K, "frames": len(X), "clips": len(ids),
                "hours": got / 3600, "base_revision": T.BASE_REVISION}, os.path.join(a.out, "kmeans.pt"))
    use = (cnt / cnt.sum()).cpu().numpy()
    json.dump({"frames": len(X), "hours": got / 3600, "cluster_share_max": float(use.max()),
               "cluster_share_min": float(use.min()), "clusters_under_0.01pct": int((use < 1e-4).sum())},
              open(os.path.join(a.out, "kmeans.json"), "w"), indent=1)


# ---------------------------------------------------------------- train

def train(a, device):
    from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices
    torch.manual_seed(a.seed)
    rows, hold = load_pool(False), load_pool(True)
    p = mix_weights(rows, a.adult_share)
    km = torch.load(os.path.join(a.out, "kmeans.pt"), map_location=device)
    cent = km["centroids"].to(device)
    teacher = Teacher(device)
    student = wavlm(device, trainable=True)
    if a.grad_ckpt:
        student.gradient_checkpointing_enable()
    head = torch.nn.Linear(student.config.hidden_size, K).to(device)
    body = [q for q in student.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": a.head_lr, "weight_decay": 0.0},
                             {"params": body, "lr": a.lr, "weight_decay": 0.01}], betas=(0.9, 0.98), eps=1e-6)
    pool_sec = sum(r["dur"] for r in rows)
    child_sec = sum(r["dur"] for r in rows if r["kind"] == "child")
    # a pass = the child audio once; adult draws come on top (adult_share of all drawn seconds); ~85% batch fill
    max_steps = a.max_steps or int(math.ceil(a.passes * child_sec / (1 - a.adult_share) / (a.batch_seconds * 0.85)))
    sched_steps = max_steps - a.head_only

    def lr_mult(step):  # body: 0 during head-only, then warmup + cosine
        if step < a.head_only:
            return 0.0
        s = step - a.head_only
        return min(1.0, s / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(sched_steps, 1))))

    step = 0
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    cfg = {**vars(a), "max_steps": max_steps, "pool_rows": len(rows), "pool_hours": pool_sec / 3600,
           "heldout_rows": len(hold), "kmeans": {k: km[k] for k in ("layer", "k", "frames", "hours")},
           "base_model": T.BASE_MODEL, "base_revision": T.BASE_REVISION, "code": T.code_version()}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
    ds = PoolClips(rows, a.seed)
    sampler = T.SecondsBatches([{"dur": r["dur"]} for r in rows], p, a.batch_seconds, a.seed, step)
    dl = iter(torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=8,
                                          prefetch_factor=4, persistent_workers=True))
    # fixed held-out monitor sets: up to N child and N adult clips (1% of speakers, never trained on)
    rng = np.random.default_rng(0)
    mon = {}
    for kind in ("child", "adult"):
        idx = [i for i, r in enumerate(hold) if r["kind"] == kind]
        mon[kind] = sorted(rng.choice(idx, size=min(a.monitor_clips, len(idx)), replace=False).tolist()) if idx else []
    hold_ds = PoolClips(hold, 0)

    def forward(x, m, masked):
        h_t = teacher(x, m)
        n = h_t.shape[1]
        fm = frame_mask(student, m, n)
        tgt = assign(h_t[fm], cent)  # same row-major order as logits[fm]
        mt = None
        if masked:
            mt = torch.from_numpy(_compute_mask_indices((len(x), n), a.mask_prob, a.mask_len, attention_mask=fm.long().cpu(),
                                                        min_masks=2)).to(device) & fm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(x, attention_mask=m, mask_time_indices=mt).last_hidden_state
        logits = head(out.float())[fm]
        sel = mt[fm] if masked else torch.ones_like(tgt, dtype=torch.bool)
        loss = torch.nn.functional.cross_entropy(logits[sel], tgt[sel])
        acc = (logits[sel].argmax(-1) == tgt[sel]).float().mean()
        return loss, acc, int(sel.sum())

    @torch.no_grad()
    def monitor():
        student.eval()
        res = {}
        for kind, ids in mon.items():
            L = A = N = 0.0
            for j in range(0, len(ids), 16):
                x, m, _ = collate([hold_ds[i] for i in ids[j:j + 16]])
                x, m = x.to(device), m.to(device)
                torch.manual_seed(1234 + j)
                np.random.seed(1234 + j)
                loss, acc, n = forward(x, m, masked=True)
                L, A, N = L + loss.item() * n, A + acc.item() * n, N + n
            if N:
                res[f"heldout/{kind}_loss"], res[f"heldout/{kind}_acc"] = L / N, A / N
        student.train()
        return res

    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    t_end = time.time() + a.time_limit_min * 60 if a.time_limit_min else None
    if step == 0:
        log.write(json.dumps({"step": 0, **monitor()}) + "\n")
        log.flush()
    student.train()
    t0, audio = time.time(), 0.0
    while step < max_steps:
        for g, mult in zip(opt.param_groups, (min(1.0, (step + 1) / 100), lr_mult(step))):
            g["lr"] = (a.head_lr if g is opt.param_groups[0] else a.lr) * mult
        np.random.seed([a.seed, step])
        x, m, _ = next(dl)
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        loss, acc, n = forward(x, m, masked=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(head.parameters()) + body, 1.0).item()
        opt.step()
        step += 1
        audio += m.sum().item() / 16000
        if step % 10 == 0:
            el = time.time() - t0
            log.write(json.dumps({"step": step, "train/loss": loss.item(), "train/masked_acc": acc.item(),
                                  "train/masked_frames": n, "train/grad_norm": gn, "train/lr_body": opt.param_groups[1]["lr"],
                                  "train/lr_head": opt.param_groups[0]["lr"], "sys/audio_sec_per_sec": audio / max(el, 1e-6),
                                  "sys/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}) + "\n")
            log.flush()
            if not math.isfinite(loss.item()):
                raise SystemExit(f"non-finite loss at step {step}")
        if step % a.eval_every == 0 or step == max_steps:
            log.write(json.dumps({"step": step, **monitor()}) + "\n")
            log.flush()
        if step % a.save_every == 0 or step == max_steps or (t_end and time.time() > t_end):
            torch.save({"student": student.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(), "step": step},
                       last + ".tmp")
            os.replace(last + ".tmp", last)
            if t_end and time.time() > t_end and step < max_steps:
                print(f"time limit: stopping at step {step}; resubmit to resume", flush=True)
                return
    student.save_pretrained(os.path.join(a.out, "wavlm"))
    torch.save(head.state_dict(), os.path.join(a.out, "head.pt"))
    print(f"done at step {step}; backbone in {a.out}/wavlm", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--kmeans", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--out", default="RUNS/midtrain-v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adult-share", type=float, default=0.15)
    ap.add_argument("--kmeans-hours", type=float, default=50.0)
    ap.add_argument("--frame-keep", type=float, default=0.1)
    ap.add_argument("--kmeans-iters", type=int, default=30)
    ap.add_argument("--passes", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = from --passes")
    ap.add_argument("--batch-seconds", type=float, default=200.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--head-only", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--mask-prob", type=float, default=0.65)
    ap.add_argument("--mask-len", type=int, default=10)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--monitor-clips", type=int, default=500)
    ap.add_argument("--time-limit-min", type=float, default=0)
    a = ap.parse_args()
    if a.build_pool:
        build_pool()
        return
    device = torch.device("cuda")
    if a.kmeans:
        kmeans(a, device)
    if a.train:
        train(a, device)


if __name__ == "__main__":
    main()
