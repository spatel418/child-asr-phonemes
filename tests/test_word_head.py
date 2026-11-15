"""Aux word head: the dual-head path gives the same phone loss as HF's own CTC loss, rows without
targets are skipped, and gradients reach both heads and the adapters.
Run: python tests/test_word_head.py"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import WavLMConfig, WavLMForCTC  # noqa: E402

import train_sft as T  # noqa: E402
import wavlm_lora  # noqa: E402


def model_with_heads():
    wavlm_lora.patch()
    torch.manual_seed(0)
    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                      conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 4), num_conv_pos_embeddings=16,
                      num_conv_pos_embedding_groups=4, vocab_size=10, do_stable_layer_norm=True,
                      feat_extract_norm="layer", layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0,
                      activation_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0, mask_time_prob=0.0,
                      ctc_loss_reduction="mean", ctc_zero_infinity=True, pad_token_id=0)
    m = get_peft_model(WavLMForCTC(cfg), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, modules_to_save=["lm_head"],
        target_modules=r".*encoder\.layers\.\d+\.attention\.(q_proj|k_proj|v_proj|out_proj)"))
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)  # B=0 would hide adapter gradients from the check
    T.add_word_head(m, argparse.Namespace(word_weight=0.3, blank_bias=-2.0), "cpu")
    return m.eval()


x = torch.randn(3, 4000)
mask = torch.ones(3, 4000, dtype=torch.long)
mask[2, 3000:] = 0
lab = torch.tensor([[1, 2, 3], [4, 5, -100], [-100, -100, -100]])  # row 2: word-only clip
wlab = torch.tensor([[1, 5, 6, 7], [-100, -100, -100, -100], [8, 9, 1, 10]])  # row 1: no words


def test_phone_loss_matches_hf():
    m = model_with_heads()
    keep = torch.tensor([0, 1])  # HF can't take a row with no target; compare on rows that have one
    hf = m(input_values=x[keep], attention_mask=mask[keep], labels=lab[keep]).loss
    _, lp, _, n_p, n_w = T.dual_forward(m, x, mask, lab, wlab, 0.3)
    assert n_p == 2 and n_w == 2
    assert torch.allclose(hf, lp, atol=1e-5), (float(hf), float(lp))


def test_gradients_reach_everything():
    m = model_with_heads()
    loss, lp, lw, _, _ = T.dual_forward(m, x, mask, lab, wlab, 0.3)
    assert torch.allclose(loss, lp + 0.3 * lw)
    loss.backward()
    assert m.word_head.weight.grad is not None and m.word_head.weight.grad.abs().sum() > 0
    lora = [p.grad for n, p in m.named_parameters() if "lora_A" in n]
    assert lora and all(g is not None and g.abs().sum() > 0 for g in lora)
    head = [p.grad for n, p in m.named_parameters() if "lm_head" in n and p.requires_grad]
    assert head and all(g is not None and g.abs().sum() > 0 for g in head)


def test_word_only_batch_trains_encoder_not_phone_head():
    m = model_with_heads()
    none = torch.full_like(lab, -100)
    loss, lp, _, n_p, _ = T.dual_forward(m, x, mask, none, wlab, 0.3)
    assert n_p == 0 and float(lp) == 0.0
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for n, p in m.named_parameters() if "lm_head" in n and p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "lora_A" in n)


def test_word_tokens():
    assert [T.WORD_VOCAB[i] for i in T.word_tokens("the (be)cause [: x] .", "ACAD")] == list("the cause")
    assert T.word_tokens("I want xxx .", "ACAD") == []
    assert T.word_tokens(float("nan"), "SPROUT") == []


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
