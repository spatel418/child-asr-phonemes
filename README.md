# child-asr-phonemes

A phoneme-level speech recognizer for children's speech, built to help spot speech sound disorders (SSD) and low intelligibility.

## Idea

1. Start from a CTC speech encoder (e.g. WavLM).
2. Fine-tune it on child speech with hand-labeled IPA (SFT).
3. Improve it further with reinforcement learning from verifiable rewards (RLVR).

For diagnosis, we compare what the child **should** say (the target word as phonemes) with what the model hears the child **actually** say. The size and type of the differences are the signal.

A CTC model is used on purpose. Models that predict whole words tend to "fix" a child's errors toward the real word, which hides exactly what we need to measure.

## What's here

| File | What it is |
|---|---|
| `DATA.md` | Data sources, how to get them, manifest fields, train/val/test splits, preprocessing |
| `TRAINING_RECIPE.md` | Training plan: SFT, then RLVR, and how we evaluate |
| `scripts/tb_download.py` | Download TalkBank / PhonBank corpora with your own login |
| `scripts/check_jibo.py` | Check the JIBO Kids corpus download |
| `scripts/build_sprout_manifest.py` | Build a manifest from the SPROUT release |

No data is stored in this repo. Each corpus has its own license; get it from the original source (see `DATA.md`).

## Downloading TalkBank data

You need a TalkBank account. Credentials are never stored in the repo.

```bash
cp .env.example .env     # then put your email and data path in .env (git-ignored)
read -s -p "TalkBank password: " TALKBANK_PASSWORD; echo; export TALKBANK_PASSWORD

python scripts/tb_download.py --list Cummings   # dry run: sizes only
python scripts/tb_download.py all               # download (safe to re-run)
```

Please follow the [TalkBank ground rules](https://talkbank.org/0share/rules.html) and cite each corpus you use.

## Status

Early work in progress: gathering data and building splits.
