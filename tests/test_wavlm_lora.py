"""LoRA on WavLM attention: the patch is needed and works. Run: python tests/test_wavlm_lora.py"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import WavLMConfig, WavLMForCTC  # noqa: E402
from transformers.models.wavlm import modeling_wavlm  # noqa: E402

import wavlm_lora  # noqa: E402

ORIGINAL = modeling_wavlm.WavLMAttention.torch_multi_head_self_attention
TARGET = r".*encoder\.layers\.\d+\.attention\.(q_proj|k_proj|v_proj|out_proj)"


def tiny():
    torch.manual_seed(0)
    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                      conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 4), num_conv_pos_embeddings=16,
                      num_conv_pos_embedding_groups=4, vocab_size=10, do_stable_layer_norm=True,
                      feat_extract_norm="layer", layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0,
                      activation_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0, mask_time_prob=0.0)
    return WavLMForCTC(cfg).eval()


def lora(model):
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGET))


def lora_grads(m):
    return [p.grad for n, p in m.named_parameters() if "lora_" in n and "attention" in n]


x = torch.randn(2, 4000)


def test_patch_matches_original():
    m = tiny()
    modeling_wavlm.WavLMAttention.torch_multi_head_self_attention = ORIGINAL
    a = m(x).logits
    wavlm_lora.patch()
    b = m(x).logits
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_without_patch_lora_gets_no_gradient():
    modeling_wavlm.WavLMAttention.torch_multi_head_self_attention = ORIGINAL
    m = lora(tiny())
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p)
    out = m(x).logits
    # the output does not depend on any trainable (LoRA) weight at all
    assert not out.requires_grad, "expected the bug: LoRA adapters unused"


def test_with_patch_lora_changes_output_and_gets_gradient():
    wavlm_lora.patch()
    base = tiny()
    ref = base(x).logits.detach()
    m = lora(copy.deepcopy(base))
    assert torch.allclose(m(x).logits, ref, atol=1e-5), "B=0 must equal the base model"
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p)
    out = m(x).logits
    assert not torch.allclose(out, ref, atol=1e-3), "LoRA had no effect"
    out.sum().backward()
    g = lora_grads(m)
    assert g and all(t is not None and t.abs().sum() > 0 for t in g), "LoRA params got no gradient"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
