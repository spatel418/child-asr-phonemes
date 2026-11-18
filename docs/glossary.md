# Glossary (in depth)

Short versions: [glossary_short.md](glossary_short.md). Add a term to both files when it comes up.

## Scores

**PER (phone error rate).** We line up the model's phones with the label phones and count the fewest edits
that turn one into the other. PER = edits / number of label phones. 0 is perfect. It can go above 1 when the
model outputs far more phones than the label has (pasketti on Cummings: 1.18).

**sub / del / ins.** The three edit types. A substitution is a wrong phone (said /t/, model wrote /k/). A
deletion is a phone the model skipped. An insertion is a phone the model added. They sum to PER. Our models
lean to deletions. Pasketti leans to insertions.

**len ratio.** Total output phones / total label phones. Ours is about 0.86 (too short, matches the deletions).
Pasketti is about 1.06.

**empty rate.** Share of clips where the model output nothing at all.

**select score.** The number we use to pick the best checkpoint: PER averaged over five groups so that no one
big corpus drowns out the rest.

**target.** The dictionary pronunciation of the word the child meant to say (from CMUdict). Not what the
child said.

**child error.** A target consonant the child said differently, according to the human transcription.

**detected / target bias.** Of the child errors, detected = the model also wrote something other than the
target. Target bias = the model wrote the target instead, so it "corrected" the child. They add up to 1.
Target bias matters clinically: it hides speech errors.

**false alarm.** A consonant the child said correctly that the model marked as wrong (often by dropping it).
Too many false alarms make a typical child look disordered.

**PCC (percent consonants correct).** The share of target consonants a child says correctly. A standard
clinical score. We compute it twice per child: from the human transcription and from the model output.

## How the model learns

**CTC (connectionist temporal classification).** The model makes one guess every 20 ms. Labels say only which
phones were said, not when. CTC adds up every way the guesses could line up with the label, so the model can
learn without timing. The output is cleaned up by merging repeats and dropping blanks.

**blank / blank rate.** CTC needs a "nothing new here" symbol between phones. Phones are much rarer than 20 ms
frames, so a healthy model outputs blank about 90%+ of the time. Early in training a model often outputs only
blanks (blank rate 1, PER 1). That stage is normal and passes.

**loss.** The CTC score of how unlikely the label is under the model. Training pushes it down. Train loss
falling while val PER stops improving is the classic sign of overfitting.

**loss_phone / loss_word.** Only in word-head runs. loss_phone is the CTC loss of the phone head. loss_word is
the CTC loss of the word head on letters. Total loss = loss_phone + 0.3 x loss_word. If loss_word falls but
loss_phone stalls compared with a run without the word head, the word task is crowding out the phone task.

**gradient.** For every trainable weight, the direction and size of the change that would lower the loss on
this batch.

**grad norm.** The size of the whole gradient as one number (square root of the sum of squares over all
trainable weights). High at the start, then it settles. One spike is usually one odd batch. Repeated spikes or
a steady climb warn that training is going unstable, and the loss often blows up soon after.

**gradient clipping / clip_frac.** If grad norm is above the cap (1.0), the whole gradient is scaled down to
1.0. Same direction, smaller step. clip_frac is the share of the last 10 steps that got capped. In our runs it is
1.0: every step gets capped (grad norm about 2.3 for LoRA, about 5 for full fine-tune). So the real step size is
smaller than the LR suggests. Worth one test with a higher cap.

**learning rate (LR).** How far each step moves the weights. lr_lora is for the backbone (LoRA adapters or the
full model). lr_head is for the output layer, set 3x higher because it starts from random.

**warmup / head-only steps.** For the first 300 steps only the output head trains (adapters_on = 0), so the
random head doesn't send garbage gradients into the pretrained model. The LR then ramps up over the first 500
steps. Our LR stays constant after that. Pasketti decays it to zero (cosine).

**EMA (exponential moving average).** A second copy of the weights that follows the live weights slowly
(each step: 0.999 x old + 0.001 x new). It smooths out step-to-step noise and often scores a little better.
val_ema/ in W&B is the same scoring as val/, done on this copy.

## Model setups

**WavLM Large.** The pretrained speech model we start from (24 transformer layers, about 300M weights). It
learned speech from adult audio without labels.

**LoRA.** Freeze the big weight matrices and learn a small low-rank change for some of them (here q, k, v,
out in every attention layer, rank 16). Far fewer trained weights, smaller checkpoints, less overfitting.

**full fine-tune.** Train every weight except the first audio (CNN) layers. This is what pasketti did.

**word head.** A second output layer on the same encoder that learns letters (a-z, space, ') from clips that
have only word transcripts (MyST, OCSC, Ellis Weismer, Cameron, about 192 h). It gives the encoder more child
speech to learn from. It is thrown away at test time. The phone head is the only output we score.

**seed.** Fixes the random choices in a run: data order, new-layer starting values, dropout. Two runs that
differ only in seed show how much a score moves by luck.

## Data

**plain / denoised.** Both copies are the same recording, resampled to 16 kHz and set to a standard loudness.
Denoised also has a noise-removal model run over it. Plain has scored better so far (0.332 vs 0.370 val PER,
one seed each).

**splits.** Children, never clips, are split, so no child is in both train and val. Within each corpus, held-out
children are drawn per diagnostic group (typical / SSD / ...) to keep the corpus's mix. Rules: DATA.md section 3,
code: scripts/make_splits.py.

**val.** About 10% of children from each training corpus, 15 SPROUT children, and all of Preston. Used to
compare runs and pick checkpoints. Never trained on.

**Test A / B / C.** A: more held-out children from the training corpora. B: corpora never trained on
(TorringtonEaton, McAllister, JIBO). C: the 40 sealed SPROUT children plus 18 more, balanced SSD / typical.
None is opened until the final model.

**pasketti-phonetic.** DrivenData's public child phoneme model (WavLM Large, full fine-tune, word head). Our
baseline: every result is shown next to it on the same val clips. It trained on SPROUT, ACAD, PERCEPT-R,
PERCEPT-GFTA and others, so it has seen some of our val children's corpora. Preston is the fair comparison.
