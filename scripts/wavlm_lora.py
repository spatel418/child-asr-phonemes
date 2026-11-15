"""Make LoRA work on WavLM attention.

WavLM's attention (transformers WavLMAttention.torch_multi_head_self_attention)
passes q_proj.weight, k_proj.weight, v_proj.weight and out_proj.weight straight
to F.multi_head_attention_forward. It never calls the Linear layers, so PEFT's
LoRA wrappers on those layers are skipped: the adapters get no gradient and the
run quietly trains only the CTC head.

patch() replaces that function with one that passes W + (B @ A) * scale for
LoRA-wrapped layers. Gradients then reach the adapters. LoRA dropout is not
applied on this path, so use lora_dropout=0 for attention adapters.
Checked by tests/test_wavlm_lora.py.
"""
import torch.nn.functional as F
from transformers.models.wavlm import modeling_wavlm


def _weight(m):
    base = getattr(m, "base_layer", None)
    if base is None:
        return m.weight
    w = base.weight
    for name in m.active_adapters:
        if name in m.lora_A and not m.disable_adapters and not m.merged:
            w = w + m.get_delta_weight(name)
    return w


def _bias(m):
    return getattr(m, "base_layer", m).bias


def _attn(self, hidden_states, attention_mask, gated_position_bias, output_attentions):
    query = key = value = hidden_states.transpose(0, 1)
    key_padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    attn_output, attn_weights = F.multi_head_attention_forward(
        query, key, value, self.embed_dim, self.num_heads,
        None,
        modeling_wavlm.torch.cat((_bias(self.q_proj), _bias(self.k_proj), _bias(self.v_proj))),
        None, None, False, self.dropout,
        _weight(self.out_proj), _bias(self.out_proj),
        self.training, key_padding_mask, output_attentions, gated_position_bias,
        use_separate_proj_weight=True,
        q_proj_weight=_weight(self.q_proj), k_proj_weight=_weight(self.k_proj), v_proj_weight=_weight(self.v_proj),
    )
    attn_output = attn_output.transpose(0, 1)
    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:])
    return attn_output, attn_weights


def patch():
    modeling_wavlm.WavLMAttention.torch_multi_head_self_attention = _attn
