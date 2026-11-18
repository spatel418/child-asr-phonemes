# Mid-training: design (approved 2026-11-16)

## The idea

Before SFT, keep training WavLM-Large on lots of child audio **without labels**, so the model is used to child voices before it learns IPA. Then run our usual SFT recipe from the new weights and compare it with the same recipe from plain WavLM.

Order: WavLM-Large → **mid-training (no labels)** → SFT (CTC on IPA) → RL.

Why try it: our errors sit in young children and connected speech (ages 3–4 give 85% of val errors; deletions are 4× insertions). More labeled data helped (Providence age 2: 0.3247 → 0.3165, bootstrap CI −0.011 to −0.005). Unlabeled child audio is the biggest pool we have left.

## Rule that cannot bend

Only **train** children's audio, plus corpora that are in no eval split. No audio from val, Test A, Test B or Test C children, even without labels. That includes younger Providence sessions of val/Test A children and the sealed SPROUT children. The k-means codebook (see Method) is also fit on this pool only.

## The audio pool

Hours from the manifests and indexes on disk, 2026-11-16.

| Source | Hours | Status | Notes |
|---|---|---|---|
| SFT train clips (all phonetic corpora, train children) | ~71 | on disk, cut | already seen in SFT, labels ignored here |
| Providence age 2, train children | 44 | on disk, cut | already in prov2 SFT |
| Word-only corpora (MyST 148, OCSC 55, Ellis Weismer 22, Cameron 5) | 230 | on disk, cut | different children from all our eval splits |
| CSLU Kids | ~100 | on disk, raw | scripted + spontaneous; needs cutting of the long spontaneous files |
| Adult speech (mothers/examiners) in train children's sessions + mid-training half of SEED adults | ~15% of the mix | needs cutting from CHAT timings | keeps adult skills (decided 2026-11-16); LibriSpeech only if short |
| **Total without new data** | **~450 child + ~15% adult** | | |

Not in the first run:

- **SEED** (child 4.9 h, adult 4.6 h): **half, split by child** (decided 2026-11-16). SSD/typical balanced halves; one half goes into mid-training, the other stays a clean test. First check no SEED child is a SPROUT val/test child.
- **SPROUT extra audio.** Every SPROUT v2 clip already has IPA and is already in SFT. The only extra SPROUT audio is the Secured full sessions (818 GB on S3, not copied). They hold examiner speech too, so they need child/adult separation first. It is the closest match to our weakest corpus, so it is the best phase-2 source, for train children only, after a permission check on copying it to Quest.
- **CHILDES-Aligned** (413 h, child-only clips, 38.6 GB, not downloaded). It likely contains Providence, so it needs child matching against our val/test children first. Phase 2.
- Providence under age 2: skipped (mostly babble). Maybe ages 18–23 months in phase 2.

## Method

HuBERT-style masked prediction, the same kind of objective WavLM was trained with:

1. **Targets.** Run plain WavLM-Large over a sample of the pool (~50 h), take one middle layer (layer 12), and fit k-means with 500 clusters. Each 20 ms frame gets a cluster number.
2. **Training.** Mask spans of the CNN features (WavLM defaults: 8% of frames start a 10-frame span). Predict the cluster number of masked frames with a small linear head. CNN frozen. Transformer fully trained at a low LR (2e-5, warmup, cosine decay). bf16, batches of 200 s of audio as in SFT.
3. **Length.** 2 passes over the child audio (~400 h after 15 s crops), with adult draws on top: about 20,000 steps of 200 s. (Earlier draft said 5,000: a math error.)
4. **Then SFT.** Same prov2 recipe and seed, starting from the mid-trained weights. The head is thrown away.

Leaving out for now: WavLM's utterance-mixing denoising, and a second k-means round on the new model's features.

## Cost (one H100)

| Step | Time |
|---|---|
| Cut Providence under 2 and CSLU (CPU array) | ~1 h |
| Features + k-means | ~1 h |
| Mid-training | ~3–4 h |
| SFT from the new weights | ~2.5 h |
| Error report + bootstrap | ~15 min |

Total ~8 h of GPU, spread over about one day.

## Checks along the way

- Smoke test: about 50 steps on a few clips; masked-frame accuracy must go up.
- Masked prediction loss and accuracy on held-out **train-pool** clips (not val audio).
- **Adult forgetting check:** masked-prediction loss on held-back adult clips (train-children sessions), before vs after. A big rise means raise the adult share.
- **SFT start check:** run the plain-WavLM CTC SFT start (step 0 of SFT) on val. If the mid-trained base starts much worse, that's a warning, not a stop.
- Integrity review before trusting the result: list of every child whose audio went in, checked against the val/test split files and the sealed list.

## Success rule (set now, before the run)

Mid-training counts as a win only if, against prov2 SFT (0.3165) on val:

1. the per-child bootstrap 95% CI of the select-score difference is fully below 0, **and**
2. no corpus gets worse by more than 0.01 pooled PER, **and**
3. SPROUT and Providence do not get worse.

Otherwise it's a failure, and post-hoc re-definitions go in the log labeled as such. As with the word head, one seed is a weak result; if it passes, a second seed is run before we build on it.

Every result is shown next to pasketti-phonetic (0.400) and next to prov2 SFT.

## Decided

1. SEED: half by child into mid-training, half stays test.
2. First run = audio on disk; SPROUT Secured sessions and CHILDES-Aligned are phase 2.
3. Providence under age 2: skipped.
4. 15% adult audio in the mix.
5. Order: mid-training → SFT (prov2 recipe) → RL.
