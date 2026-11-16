"""SFT: WavLM-Large + LoRA + CTC head -> the 44 shared phones (ctc-sft skill).

    PY=/projects/p32576/simron/envs/pasketti-phonetic/bin/python
    $PY -I scripts/train_sft.py --run-dir RUNS/smoke --max-steps 50 --train-limit 2000 --val-limit 300 --wandb off
    $PY -I scripts/train_sft.py --run-dir RUNS/overfit5 --overfit 5 --max-steps 400 --wandb off
    $PY -I scripts/train_sft.py --run-dir RUNS/sft-r16-s0                   # real run (Slurm: slurm/train_sft.sbatch)

Resumes automatically from <run-dir>/last.pt (same W&B run, same LR schedule).
Stops cleanly before --time-limit-min so a chained Slurm job can continue it.
Writes <run-dir>/DONE when finished (max steps or early stop).

Privacy: W&B gets numbers only. No audio, no IPA strings, no child or clip IDs.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import secrets
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wavlm_lora  # noqa: E402
from build_manifest import INFO_FLAGS, OUT as MANIFESTS  # noqa: E402
from phoneset import PHONES, VERSION as PHONESET  # noqa: E402
from wordset import BLOCKING as WORD_BLOCKING, LETTERS, VERSION as WORDSET, normalize_chat, normalize_plain  # noqa: E402

DATA = os.path.dirname(MANIFESTS.rstrip("/"))
BASE_MODEL = "microsoft/wavlm-large"
BASE_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
VOCAB = ["<blank>"] + PHONES          # blank = 0 = pad
TOK = {p: i for i, p in enumerate(VOCAB)}
WORD_VOCAB = ["<blank>", " "] + LETTERS  # aux word head (letters), training only
WTOK = {c: i for i, c in enumerate(WORD_VOCAB)}
WORD_CORPORA = ["MyST", "Cameron", "EllisWeismer", "OCSC"]
SCORE_TAGS = ["indomain_typical", "indomain_SSD", "preston_SSD", "sprout_typical", "sprout_SSD"]
TRAIN_CORPORA = ["SPROUT", "ACAD", "Cummings", "PERCEPT-R", "PERCEPT-GFTA", "Penney", "Providence"]
VAL_CORPORA = TRAIN_CORPORA + ["Preston"]


def args_parser():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--run-dir", required=True)
    a("--splits", default="v2")
    a("--train-children", default=None,
      help="optional file of child IDs (one per line): train only on these, a subset of the split's train children")
    a("--audio", default="denoised", choices=["denoised", "plain"])
    a("--seed", type=int, default=0)
    # schedule (in steps)
    a("--epochs", type=float, default=10)
    a("--max-steps", type=int, default=0, help="overrides --epochs")
    a("--batch-seconds", type=float, default=200, help="padded audio seconds per batch")
    a("--warmup-steps", type=int, default=500)
    a("--head-only-steps", type=int, default=300)
    a("--val-per-epoch", type=float, default=4)
    a("--val-every", type=int, default=0, help="overrides --val-per-epoch")
    a("--patience", type=int, default=8, help="val passes without a better score before stopping")
    # optimizer
    a("--lr", type=float, default=3e-4, help="adapter LR")
    a("--head-lr-mult", type=float, default=3.0)
    a("--head-weight-decay", type=float, default=0.01)
    a("--grad-clip", type=float, default=1.0)
    a("--ema", type=float, default=0.999)
    # model
    a("--finetune", default="lora", choices=["lora", "full"],
      help="full: train every transformer weight (CNN stays frozen); use a lower --lr, e.g. 3e-5")
    a("--lora-r", type=int, default=16)
    a("--lora-alpha", type=int, default=32)
    a("--lora-targets", default="q_proj,k_proj,v_proj,out_proj")
    a("--blank-bias", type=float, default=-2.0)
    a("--layerdrop", type=float, default=0.0)
    a("--dropout", type=float, default=0.0)
    a("--mask-time-prob", type=float, default=0.0, help="SpecAugment; off until regularize stage")
    # data mix
    a("--ssd-share", type=float, default=0.35)
    a("--child-cap", type=int, default=1000, help="clips per child at full weight")
    # aux word head (pasketti-style): word-only clips also train the encoder
    a("--word-weight", type=float, default=0.0, help="weight of the word CTC loss; 0 = off (old code path)")
    a("--word-share", type=float, default=0.3, help="share of batch audio seconds from word-only clips")
    a("--word-corpora", default=",".join(WORD_CORPORA))
    # small runs
    a("--train-limit", type=int, default=0)
    a("--val-limit", type=int, default=0)
    a("--overfit", type=int, default=0, help="train and validate on these N train clips only")
    a("--time-limit-min", type=float, default=0, help="save and exit after this many minutes")
    a("--wandb", default="on", choices=["on", "off"])
    a("--workers", type=int, default=6)
    return ap


# ---------------------------------------------------------------- data


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tag_of(r):
    g = r["group"]
    if r["corpus"] == "Preston":
        return "preston_SSD"
    return f"sprout_{g}" if r["corpus"] == "SPROUT" else f"indomain_{g}"


def load_items(args):
    split_dir = os.path.join(DATA, "splits", "phonetic", args.splits)
    split_of = {}
    for s in ("train", "val"):  # test splits are never read here
        for c in open(os.path.join(split_dir, f"{s}.txt")).read().split():
            split_of[c] = s
    processed = {}
    for p in sorted(os.listdir(os.path.join(DATA, "processed", "index"))):
        for line in open(os.path.join(DATA, "processed", "index", p)):
            r = json.loads(line)
            processed[r["clip_id"]] = (r[f"{args.audio}_path"], r["duration"])
    items = {"train": [], "val": []}
    for corpus in VAL_CORPORA:
        for line in open(os.path.join(MANIFESTS, "phonetic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            s = split_of.get(r["child_id"])
            if s is None or set(r["flags"]) - INFO_FLAGS or r["clip_id"] not in processed:
                continue
            if r["sip_sealed"]:
                raise RuntimeError("sealed child reached train/val: stop")  # never expected
            if s == "train" and corpus not in TRAIN_CORPORA:
                continue
            path, dur = processed[r["clip_id"]]
            labels = [TOK[p] for p in r["ipa_normalized"].split() if p != "|"]
            items[s].append({"path": path, "dur": dur, "labels": labels, "child": r["child_id"],
                             "corpus": corpus, "group": r["group"], "tag": tag_of(r),
                             "wlabels": word_tokens(r["orthographic"], corpus)})
    checks = {f"split_{s}_sha256": sha256(os.path.join(split_dir, f"{s}.txt")) for s in ("train", "val", "testA", "testB", "testC")}
    sub = getattr(args, "train_children", None)
    if sub:
        keep = set(open(sub).read().split())
        assert keep <= {c for c, s in split_of.items() if s == "train"}, "train-children must be split train children"
        items["train"] = [it for it in items["train"] if it["child"] in keep]
        checks["train_children_sha256"] = sha256(sub)
        checks["train_children_n"] = len(keep)
    return items, checks


def word_tokens(text, corpus):
    """Letter ids for the word head; [] if the text is unusable."""
    if not isinstance(text, str):
        return []
    words, flags = (normalize_plain if corpus == "SPROUT" else normalize_chat)(text)
    return [] if flags & WORD_BLOCKING else [WTOK[c] for c in words]


def load_word_items(args):
    """Word-only training clips (no IPA). MyST: official train part only."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_word_manifest import usable
    idx = os.path.join(DATA, "processed", "index")
    processed = {}
    for p in sorted(os.listdir(idx)):
        if p.startswith("words-"):
            for line in open(os.path.join(idx, p)):
                r = json.loads(line)
                processed[r["clip_id"]] = (r[f"{args.audio}_path"], r["duration"])
    out = []
    for corpus in args.word_corpora.split(","):
        for line in open(os.path.join(MANIFESTS, "orthographic", f"{corpus}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if not usable(r) or r["clip_id"] not in processed or r.get("myst_part", "train") != "train":
                continue
            path, dur = processed[r["clip_id"]]
            out.append({"path": path, "dur": dur, "labels": [], "wlabels": [WTOK[c] for c in r["words"]],
                        "child": r["child_id"], "corpus": corpus, "group": r["group"], "tag": "words"})
    return out


class Clips(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        y, sr = sf.read(it["path"], dtype="float32")
        assert sr == 16000
        y = (y - y.mean()) / (y.std() + 1e-7)  # WavLM feature extractor: do_normalize
        return torch.from_numpy(y), torch.tensor(it["labels"]), torch.tensor(it.get("wlabels", [])), i


def collate(batch):
    n = max(len(b[0]) for b in batch)
    x = torch.zeros(len(batch), n)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    lab = torch.full((len(batch), max(1, max(len(b[1]) for b in batch))), -100, dtype=torch.long)
    wlab = torch.full((len(batch), max(1, max(len(b[2]) for b in batch))), -100, dtype=torch.long)
    for k, (w, l, wl, _) in enumerate(batch):
        x[k, : len(w)] = w
        mask[k, : len(w)] = 1
        lab[k, : len(l)] = l
        wlab[k, : len(wl)] = wl
    return x, mask, lab, wlab, [b[3] for b in batch]


def ctc_mean(logits, in_len, labels):
    """Same loss as HF WavLMForCTC (ctc_loss_reduction='mean', zero_infinity), over rows with targets.
    Returns (loss, n_rows); loss is 0 when no row has a target."""
    tgt_len = (labels >= 0).sum(-1)
    sel = tgt_len > 0
    if not bool(sel.any()):
        return logits.sum() * 0.0, 0
    lp = torch.nn.functional.log_softmax(logits[sel], dim=-1, dtype=torch.float32).transpose(0, 1)
    lab = labels[sel]
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.ctc_loss(lp, lab.masked_select(lab >= 0), in_len[sel], tgt_len[sel],
                                            blank=0, reduction="mean", zero_infinity=True)
    return loss, int(sel.sum())


def clip_weights(items, ssd_share, child_cap):
    """Per-clip sampling weight: cap clips per child, then scale SSD to ssd_share of the mass."""
    per_child = collections.Counter(it["child"] for it in items)
    w = np.array([min(1.0, child_cap / per_child[it["child"]]) for it in items])
    ssd = np.array([it["group"] == "SSD" for it in items])
    if 0 < ssd.sum() < len(items) and ssd_share > 0:
        a, b = w[ssd].sum(), w[~ssd].sum()
        w[ssd] *= ssd_share * b / ((1 - ssd_share) * a)
    return w / w.sum()


class SecondsBatches:
    """Batch k is a pure function of (seed, k), so resuming at step k gives the same data."""

    def __init__(self, items, weights, batch_seconds, seed, start):
        self.dur = np.array([it["dur"] for it in items])
        self.w, self.bs, self.seed, self.k = weights, batch_seconds, seed, start
        self.pool = max(8, int(3 * batch_seconds / max(self.dur.mean(), 0.5)))

    def __iter__(self):
        while True:
            rng = np.random.default_rng([self.seed, self.k])
            cand = rng.choice(len(self.dur), size=self.pool, replace=True, p=self.w)
            cand = cand[np.argsort(self.dur[cand])]  # similar lengths -> less padding
            start = rng.integers(0, len(cand))
            batch, longest = [], 0.0
            for j in list(range(start, len(cand))) + list(range(start - 1, -1, -1)):
                longest2 = max(longest, self.dur[cand[j]])
                if batch and longest2 * (len(batch) + 1) > self.bs:
                    break
                batch.append(int(cand[j]))
                longest = longest2
            self.k += 1
            yield batch


# ---------------------------------------------------------------- model


def build_model(args, device):
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import WavLMConfig, WavLMForCTC

    wavlm_lora.patch()  # without it, LoRA on WavLM attention does nothing (tests/test_wavlm_lora.py)
    path = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    d = args.dropout
    cfg = WavLMConfig.from_pretrained(
        path, vocab_size=len(VOCAB), pad_token_id=0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        layerdrop=args.layerdrop, hidden_dropout=d, attention_dropout=d, activation_dropout=d,
        feat_proj_dropout=d, final_dropout=d, mask_time_prob=args.mask_time_prob, mask_feature_prob=0.0)
    torch.manual_seed(args.seed)
    model = WavLMForCTC.from_pretrained(path, config=cfg)
    model.freeze_feature_encoder()
    with torch.no_grad():
        model.lm_head.bias[0] = args.blank_bias
    if getattr(args, "finetune", "lora") == "full":  # older configs have no key: LoRA
        model = model.to(device)
        add_word_head(model, args, device)
        head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("lm_head")]
        body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("lm_head")]
        assert head and body, "no trainable head or backbone"
        return model, head, body
    targets = "|".join(args.lora_targets.split(","))
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      target_modules=rf".*encoder\.layers\.\d+\.(attention|feed_forward)\.({targets})",
                      modules_to_save=["lm_head"])
    model = get_peft_model(model, lcfg).to(device)
    add_word_head(model, args, device)
    head = [p for n, p in model.named_parameters() if p.requires_grad and "lm_head" in n]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    assert head and lora, "no trainable head or adapters"
    return model, head, lora


def add_word_head(model, args, device):
    """Aux letter head on the last hidden state. Its params go LAST in every saved list, so
    loaders that know only head + lora (eval_ckpt) still line up."""
    if getattr(args, "word_weight", 0) > 0:
        hid = (model.get_base_model() if hasattr(model, "get_base_model") else model).config.hidden_size
        model.word_head = torch.nn.Linear(hid, len(WORD_VOCAB)).to(device)
        with torch.no_grad():
            model.word_head.bias[0] = args.blank_bias


def word_head_params(model):
    return list(model.word_head.parameters()) if hasattr(model, "word_head") else []


def dual_forward(model, x, mask, lab, wlab, word_weight):
    """Phone CTC + word_weight * word CTC from one encoder pass."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hid = base.dropout(base.wavlm(input_values=x, attention_mask=mask)[0])
    in_len = base._get_feat_extract_output_lengths(mask.sum(-1))
    lp, n_p = ctc_mean(base.lm_head(hid), in_len, lab)
    lw, n_w = ctc_mean(model.word_head(hid), in_len, wlab)
    return lp + word_weight * lw, lp, lw, n_p, n_w


class EMA:
    def __init__(self, params, decay):
        self.params, self.decay = params, decay
        self.shadow = [p.detach().clone().float() for p in params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self):  # swap live and EMA weights (call twice to undo)
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.copy_(s.to(p.dtype))
            s.copy_(tmp.float())


# ---------------------------------------------------------------- eval


def edit_ops(ref, hyp):
    """(substitutions, deletions, insertions) for the cheapest alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = de = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            de, i = de + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, de, ins


def greedy(logits, n_frames):
    ids = logits.argmax(-1).cpu().numpy()
    out = []
    for row, n in zip(ids, n_frames):
        prev, seq = -1, []
        for t in row[:n]:
            if t != prev and t != 0:
                seq.append(int(t))
            prev = t
        out.append(seq)
    return out, ids


@torch.no_grad()
def evaluate(model, loader, items, device):
    model.eval()
    acc = collections.defaultdict(lambda: np.zeros(6))  # ref, sub, del, ins, empty, clips
    blank = frames = 0
    hyp_len = ref_len = 0
    loss_sum = loss_n = 0.0
    for x, mask, lab, _, idx in loader:
        x, mask, lab = x.to(device), mask.to(device), lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_values=x, attention_mask=mask, labels=lab)
        loss_sum += float(out.loss) * len(idx)
        loss_n += len(idx)
        n_frames = (model.get_base_model() if hasattr(model, "get_base_model") else model)._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        hyps, ids = greedy(out.logits.float(), n_frames)
        for k, (h, i) in enumerate(zip(hyps, idx)):
            ref = items[i]["labels"]
            s, de, ins = edit_ops(ref, h)
            for key in (items[i]["tag"], f"corpus_{items[i]['corpus']}", "all"):
                acc[key] += (len(ref), s, de, ins, len(h) == 0, 1)
            blank += int((ids[k][: n_frames[k]] == 0).sum())
            frames += n_frames[k]
            hyp_len += len(h)
            ref_len += len(ref)
    model.train()
    m = {"val/loss": loss_sum / max(loss_n, 1), "val/blank_rate": blank / max(frames, 1),
         "val/len_ratio": hyp_len / max(ref_len, 1)}
    for key, (r, s, de, ins, emp, n) in acc.items():
        m[f"val/per/{key}"] = (s + de + ins) / max(r, 1)
        m[f"val/empty_rate/{key}"] = emp / max(n, 1)
        if key == "all":
            m.update({"val/sub_rate": s / r, "val/del_rate": de / r, "val/ins_rate": ins / r})
    present = [t for t in SCORE_TAGS if f"val/per/{t}" in m]
    m["val/select_score"] = float(np.mean([m[f"val/per/{t}"] for t in present]))  # lower is better
    m["val/select_groups"] = len(present)
    return m


def code_version():
    """Git commit read from .git (GPU nodes have no git) + a hash of the code actually run."""
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    commit = None
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            rp = os.path.join(repo, ".git", ref)
            if os.path.exists(rp):
                commit = open(rp).read().strip()
            else:
                for line in open(os.path.join(repo, ".git", "packed-refs")):
                    if line.strip().endswith(ref):
                        commit = line.split()[0]
        else:
            commit = head
    except OSError:
        pass
    h = hashlib.sha256()
    for d in ("scripts", "slurm"):
        for f in sorted(os.listdir(os.path.join(repo, d))):
            if f.endswith((".py", ".sbatch")):
                h.update(f.encode() + open(os.path.join(repo, d, f), "rb").read())
    return commit, h.hexdigest()


# ---------------------------------------------------------------- main


def main():
    args = args_parser().parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, checks = load_items(args)
    train, val = items["train"], items["val"]
    rng = random.Random(args.seed)
    if args.overfit:
        train = rng.sample(train, args.overfit)
        val = train
    else:
        if args.train_limit:
            train = rng.sample(train, min(args.train_limit, len(train)))
        if args.val_limit:  # keep every tag group in a small val set
            by_tag = collections.defaultdict(list)
            for it in val:
                by_tag[it["tag"]].append(it)
            per = max(1, args.val_limit // len(by_tag))
            val = [it for t in sorted(by_tag) for it in rng.sample(by_tag[t], min(per, len(by_tag[t])))]
    train_sec = sum(it["dur"] for it in train)
    word_on = args.word_weight > 0
    words = load_word_items(args) if word_on and not args.overfit else []
    if word_on and args.train_limit:
        words = rng.sample(words, min(args.train_limit, len(words)))
    # an epoch = one pass of phonetic audio; with word clips mixed in, each batch holds less of it
    ph_share = 1 - args.word_share if words else 1.0
    steps_per_epoch = max(1, int(train_sec / (args.batch_seconds * ph_share)))
    max_steps = args.max_steps or int(args.epochs * steps_per_epoch)
    val_every = args.val_every or max(1, int(steps_per_epoch / args.val_per_epoch))
    if args.overfit:
        val_every = args.val_every or 25

    model, head, lora = build_model(args, device)
    whead = word_head_params(model)
    for p in lora:
        p.requires_grad_(False)  # head-only first
    opt = torch.optim.AdamW([
        {"params": lora, "lr": args.lr, "weight_decay": 0.0, "name": "lora"},
        {"params": head + whead, "lr": args.lr * args.head_lr_mult, "weight_decay": args.head_weight_decay, "name": "head"},
    ], betas=(0.9, 0.98), eps=1e-8)
    base_lr = [g["lr"] for g in opt.param_groups]
    ema = EMA(head + lora + whead, args.ema)

    state = {"step": 0, "best": math.inf, "bad": 0, "wandb_id": None, "slurm_jobs": []}
    last = os.path.join(args.run_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        for p, v in zip(head + lora + whead, ck["params"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        ema.shadow = [s.to(device) for s in ck["ema"]]
        state = ck["state"]
        print(f"resumed at step {state['step']}", flush=True)
    if os.environ.get("SLURM_JOB_ID"):
        state["slurm_jobs"].append(os.environ["SLURM_JOB_ID"])

    git, code_sha = code_version()
    config = {**vars(args), "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "phoneset": PHONESET,
              "vocab_size": len(VOCAB), "git_commit": git, "code_sha256": code_sha, **checks,
              "n_train_clips": len(train), "n_val_clips": len(val), "train_hours": train_sec / 3600,
              "steps_per_epoch": steps_per_epoch, "wordset": WORDSET if words else None,
              "n_word_clips": len(words), "word_hours": sum(it["dur"] for it in words) / 3600,
              "n_train_clips_with_words": sum(bool(it["wlabels"]) for it in train), "max_steps": max_steps, "val_every": val_every,
              "slurm_jobs": state["slurm_jobs"], "torch": torch.__version__}
    json.dump(config, open(os.path.join(args.run_dir, "config.json"), "w"), indent=1)
    run = None
    if args.wandb == "on":
        import wandb
        state["wandb_id"] = state["wandb_id"] or secrets.token_hex(4)
        kw = dict(project=os.environ.get("WANDB_PROJECT", "child-asr-phonemes"),
                  entity=os.environ.get("WANDB_ENTITY"), id=state["wandb_id"], resume="allow",
                  name=os.path.basename(args.run_dir.rstrip("/")), group=os.path.basename(args.run_dir.rstrip("/")),
                  job_type="train", config=config, dir=os.environ.get("WANDB_DIR", args.run_dir))
        try:
            run = wandb.init(**kw)
        except Exception as e:  # no network or W&B down: log offline; the Slurm job syncs at the end
            print(f"wandb online init failed ({type(e).__name__}); logging offline", flush=True)
            run = wandb.init(mode="offline", **kw)
        run.define_metric("*", step_metric="trainer/step")

    def log(metrics):
        line = " ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float)) and
                        (k.startswith("train/loss") or k in ("val/select_score", "val/per/all", "val/blank_rate", "train/lr_lora", "sys/audio_sec_per_sec", "sys/gpu_mem_gb")))
        print(f"STEP {state['step']} {line}", flush=True)
        with open(os.path.join(args.run_dir, "metrics.jsonl"), "a") as f:  # local copy, no W&B needed
            f.write(json.dumps({"step": state["step"], **{k: v for k, v in metrics.items()
                                                          if isinstance(v, (int, float))}}) + "\n")
        if run:
            run.log({**metrics, "trainer/step": state["step"]})

    # same batching for val every time: sorted by length, fixed batches
    val_order = sorted(range(len(val)), key=lambda i: val[i]["dur"])
    val_batches, cur, longest = [], [], 0.0
    for i in val_order:
        if cur and max(longest, val[i]["dur"]) * (len(cur) + 1) > args.batch_seconds:
            val_batches.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, val[i]["dur"])
    val_batches.append(cur)
    val_loader = torch.utils.data.DataLoader(Clips(val), batch_sampler=val_batches, collate_fn=collate,
                                             num_workers=args.workers)

    def save(path):
        torch.save({"params": [p.detach() for p in head + lora + whead], "opt": opt.state_dict(),
                    "ema": ema.shadow, "state": state, "config": config}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def run_val():
        m = evaluate(model, val_loader, val, device)
        ema.swap()
        m.update({k.replace("val/", "val_ema/"): v for k, v in evaluate(model, val_loader, val, device).items()})
        ema.swap()
        return m

    if state["step"] == 0:
        m = run_val()
        m["train/step0_check"] = 1
        log(m)

    weights = clip_weights(train, args.ssd_share, args.child_cap)
    if words:  # mix by audio seconds: word clips get word_share of the expected batch seconds
        per_child = collections.Counter(it["child"] for it in words)
        ww = np.array([min(1.0, args.child_cap / per_child[it["child"]]) for it in words])
        ww /= ww.sum()
        d_ph = np.array([it["dur"] for it in train])
        d_w = np.array([it["dur"] for it in words])
        weights = np.concatenate([(1 - args.word_share) * weights / (weights @ d_ph),
                                  args.word_share * ww / (ww @ d_w)])
        weights /= weights.sum()
        train = train + words
    batches = SecondsBatches(train, weights, args.batch_seconds, args.seed, state["step"])
    loader = torch.utils.data.DataLoader(Clips(train), batch_sampler=batches, collate_fn=collate,
                                         num_workers=args.workers, prefetch_factor=4 if args.workers else None,
                                         persistent_workers=args.workers > 0)
    model.train()
    t_log, sec_log, clipped, nan_count = time.time(), 0.0, 0, 0
    for x, mask, lab, wlab, idx in loader:
        if state["step"] >= max_steps:
            break
        step = state["step"]
        adapters_on = step >= args.head_only_steps
        for p in lora:
            p.requires_grad_(adapters_on)
        warm_head = min(1.0, (step + 1) / args.warmup_steps)
        warm_lora = min(1.0, max(0, step - args.head_only_steps + 1) / args.warmup_steps)  # from switch-on
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * (warm_head if g["name"] == "head" else warm_lora)
        x, mask, lab = x.to(device, non_blocking=True), mask.to(device), lab.to(device)
        extra = {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if word_on:
                loss, lp, lw, n_p, n_w = dual_forward(model, x, mask, lab, wlab.to(device), args.word_weight)
                extra = {"train/loss_phone": float(lp), "train/loss_word": float(lw),
                         "train/rows_phone": n_p, "train/rows_word": n_w}
            else:  # old path, unchanged
                loss = model(input_values=x, attention_mask=mask, labels=lab).loss
        if not torch.isfinite(loss):
            nan_count += 1
            opt.zero_grad(set_to_none=True)
            if nan_count > 20:
                raise RuntimeError("loss not finite 20 times: stop")
            state["step"] += 1
            continue
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head + lora + whead, args.grad_clip))
        clipped += gn > args.grad_clip
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update()
        state["step"] += 1
        sec_log += float(mask.sum()) / 16000

        if state["step"] % 10 == 0:
            dt = time.time() - t_log
            groups = collections.Counter(train[i]["group"] for i in idx)
            corp = collections.Counter(train[i]["corpus"] for i in idx)
            m = {"train/loss": float(loss), **extra, "train/grad_norm": gn, "train/clip_frac": clipped / 10,
                 "train/lr_lora": opt.param_groups[0]["lr"], "train/lr_head": opt.param_groups[1]["lr"],
                 "train/adapters_on": int(adapters_on), "train/nan_count": nan_count,
                 "train/batch_clips": len(idx), "train/batch_mean_sec": float(mask.sum()) / 16000 / len(idx),
                 "train/batch_ssd_share": groups["SSD"] / len(idx),
                 "sys/audio_sec_per_sec": sec_log / dt, "sys/step_time": dt / 10,
                 "sys/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0}
            m.update({f"train/batch_share/{c}": n / len(idx) for c, n in corp.items()})
            log(m)
            t_log, sec_log, clipped = time.time(), 0.0, 0

        if state["step"] % val_every == 0 or state["step"] >= max_steps:
            m = run_val()
            m["val/epoch"] = state["step"] / steps_per_epoch
            score = min(m["val/select_score"], m["val_ema/select_score"])
            if score < state["best"] - 1e-4:
                state["best"], state["bad"] = score, 0
                state["best_is_ema"] = m["val_ema/select_score"] < m["val/select_score"]
                state["best_step"] = state["step"]
                save(os.path.join(args.run_dir, "best.pt"))
            else:
                state["bad"] += 1
            m["val/best_select_score"] = state["best"]
            log(m)
            save(last)
            if state["bad"] >= args.patience and not args.overfit:
                print("early stop", flush=True)
                break
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            save(last)
            print(f"time limit reached at step {state['step']}; resubmit to continue", flush=True)
            if run:
                run.finish()
            return
    save(last)
    open(os.path.join(args.run_dir, "DONE"), "w").write(f"step {state['step']} best {state['best']}\n")
    print(f"done at step {state['step']}, best select_score {state['best']:.4f}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
