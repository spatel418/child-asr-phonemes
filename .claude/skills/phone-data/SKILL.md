---
name: phone-data
description: Use when building or changing anything between raw corpora and training input for the child phoneme model - manifests, cutting TalkBank sessions into clips, mapping corpus IPA to the shared phone set, flagging clips, preprocessing audio (denoise, 16 kHz), or making and freezing train/val/test splits. Also use when adding a new corpus.
---

# Phone data: raw corpora → manifests → frozen splits

Every training clip needs a person's IPA of what the child **actually said**. This skill turns raw corpora into clean manifests and frozen splits, the same way for every corpus.

Background: `DATA.md` (sources, splits, preprocessing). Paths: `CLAUDE.local.md`.

## Steps, in order

1. **Look first.** For a new corpus, run the inventory scripts before writing a reader:
   - `scripts/chat_label_hours.py` (IPA hours per corpus and age)
   - `scripts/phone_inventory.py` (every symbol used, with counts)
   - `scripts/clip_lengths.py` (lengths, long files)
   Open 10–20 clips in the local viewer and listen. Write what you found in the research log.
2. **Write a reader** that yields one manifest row per clip (fields below). Readers never change raw files.
3. **Map the phones** to the shared set (below). Unknown symbol = the script stops and prints the symbol, corpus and an example label.
4. **Flag, don't delete.** Add flags; training code decides what to skip.
5. **Preprocess audio** (below) into `processed/<corpus>/`.
6. **Validate** the manifest (checks below). Fix and rerun until clean.
7. **Make splits** by child, freeze them, log checksums.
8. **Run the `integrity-reviewer` agent** on the manifests and splits.

## Manifest row

One JSONL row per clip in `manifests/phonetic/<corpus>.jsonl` (git-ignored):

`path, duration, corpus, child_id, age, sex, task, group, target_word, ipa_original, ipa_normalized, flags, split_tag`

- `child_id`: one ID per child across corpora (e.g. children in both PERCEPT-R and PERCEPT-GFTA).
- `group`: `typical`, `SSD`, `language-only`, `unknown`. SPROUT mapping is in `DATA.md` §2.
- `target_word`: only where the task has one (picture naming). Never guess it.
- `ipa_original`: exactly as in the source. `ipa_normalized`: after mapping.

## Cutting

- Clip corpora (SPROUT v2, ACAD, SEED): use the clips as they are.
- TalkBank session corpora: cut at CHAT utterance times, **child (`*CHI`) lines with `%pho` only**.
- **Never cut audio shorter than its label.** If a time mark looks wrong, flag it.
- Cummings: many segments are exactly 3.000 s. Spot-check alignment before trusting them.

## Shared phone set

- Train on a **simplified set**; keep the detailed label in `ipa_original`.
- The mapping table is a file in the repo with tests. Every symbol seen in the inventory is listed with an action: keep, map to X, strip, or flag the clip.
- Known differences between labs (from the inventory):
  - stress marks ˈ ˌ (most Phon corpora, not SPROUT/JIBO)
  - tie bars (SPROUT/JIBO/ACAD/PERCEPT only)
  - length ː (ACAD, Penney)
  - plain `r` vs `ɹ`, rhotic hook ˞
  - CHAT/Phon markers (`*`, `( )`, `^`, `.`, `‹ ›`, `x`)
  - strays: tabs, no-break spaces, apostrophes, commas, `~`
- Normalize Unicode to NFC before mapping.
- TIMIT's 61 symbols get their own table to the same set.

## Flags

At least: `too_short`, `too_long` (over the max clip length, ~20 s), `label_too_long` (more phones than CTC can fit: WavLM gives 50 frames/s, so phones must be well under 50 × duration), `empty_label`, `unintelligible`, `low_snr`, `bad_timestamp`.

## Audio

- Denoise and normalize with the PedzSTAR QC settings (`noisereduce` spectral gating, −16 LUFS; peak normalize under 0.4 s).
- Session corpora: denoise the whole session, then cut. Clip corpora: denoise each clip.
- Resample to 16 kHz mono with a proper resampler (soxr or `resample_poly`), never sample dropping.
- Also keep a **plain 16 kHz copy** (no denoise).
- Same steps for train and test audio.

## Validation checks (must all pass)

- Every `path` exists, and the audio is 16 kHz mono with the stated duration.
- No unmapped symbols in `ipa_normalized`.
- No `child_id` missing; no child with two different groups.
- Row counts and hours per corpus printed and compared with `DATA.md`.

## Splits

- **By child, never by clip.** Splits as in `DATA.md` §3.
- Sealed SPROUT children go **only** to Test C (list path in `CLAUDE.local.md`). Check this in code, not by hand.
- Save split files to `splits/phonetic/`, then record their sha256 in the research log. **Once frozen, never edit them.** A change means a new version with a new name.
- Check: no `child_id` in two splits; every sealed child in Test C or unused.

## Never

- Commit manifests, split files or anything with child IDs.
- Drop an unknown symbol silently.
- Edit or move raw files.
- Use DrivenData competition data or TinyVox.
