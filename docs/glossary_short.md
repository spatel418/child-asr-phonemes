# Glossary (short)

One line each. Longer versions: [glossary.md](glossary.md).

| Term | Meaning |
|---|---|
| PER | Phone error rate. Share of phones the model got wrong. Lower is better. |
| sub / del / ins | The three kinds of PER errors: wrong phone, missed phone, extra phone. |
| CTC | The training method that lets the model learn phones without knowing where each one starts. |
| blank | CTC's "nothing here" output. Most frames are blank. |
| blank rate | Share of output frames that are blank. |
| len ratio | Output length / label length. Below 1 = too short (deletes), above 1 = too long (inserts). |
| loss | The number training tries to push down. |
| loss_phone / loss_word | The two parts of the loss in word-head runs. loss = loss_phone + 0.3 x loss_word. |
| gradient | For each weight, which way and how much to change it to lower the loss. |
| grad norm | One number for the size of a step's whole gradient. |
| clipping / clip_frac | Capping the gradient size at 1.0. clip_frac = share of recent steps that got capped. |
| learning rate (LR) | How big a step training takes. |
| warmup | Start with a tiny LR and raise it, so early steps don't wreck the model. |
| EMA | A smoothed copy of the weights (a running average). Often scores a bit better. |
| LoRA | Train small add-on matrices instead of the whole model. |
| full fine-tune | Train (almost) every weight in the model. |
| word head | Extra output layer that learns letters from word-only clips. Used in training only. |
| seed | The number that fixes the random choices in a run. |
| plain / denoised | Two copies of the same audio. Denoised has noise removal on top. |
| val | Held-out children used to compare runs. Never trained on. |
| Test A / B / C | Final test sets. Not opened until the final model. |
| pasketti-phonetic | DrivenData's public child phoneme model. Our baseline. |
