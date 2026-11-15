"""Padding must not leak: clip A's loss and gradient do not depend on what else is in the batch.
Same model settings as training (layer-norm feature extractor, attention mask, LoRA patch).
Run: python tests/test_padding.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from test_wavlm_lora import TARGET, tiny  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

import wavlm_lora  # noqa: E402
from train_sft import collate  # noqa: E402


def loss_and_grad(model, batch):
    x, mask, lab, _ = collate(batch)
    model.zero_grad()
    out = model(input_values=x, attention_mask=mask, labels=lab)
    # per-clip loss for clip 0 only: recompute with only its row contributing
    logp = out.logits.log_softmax(-1).transpose(0, 1)
    n = model.base_model.model._get_feat_extract_output_lengths(mask.sum(-1))
    l = torch.nn.functional.ctc_loss(logp[:, :1], lab[:1].clamp(min=0), n[:1], (lab[:1] >= 0).sum(-1), blank=0)
    l.backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return l.detach(), g


def test_padding_does_not_leak():
    wavlm_lora.patch()
    torch.manual_seed(0)
    m = get_peft_model(tiny(), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGET,
                                          modules_to_save=["lm_head"]))
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    m.eval()
    a = (torch.randn(8000), torch.tensor([1, 2, 3]), 0)
    short_b = (torch.randn(3000), torch.tensor([4, 5]), 1)
    long_b = (torch.randn(16000) * 5, torch.tensor([6, 7, 8, 9]), 2)
    la, ga = loss_and_grad(m, [a])
    l1, g1 = loss_and_grad(m, [a, short_b])
    l2, g2 = loss_and_grad(m, [a, long_b])   # A is now padded
    for l, g in ((l1, g1), (l2, g2)):
        assert torch.allclose(l, la, atol=1e-4), (l, la)
        assert torch.allclose(g, ga, atol=1e-4), (g - ga).abs().max()


if __name__ == "__main__":
    test_padding_does_not_leak()
    print("ok test_padding_does_not_leak")
