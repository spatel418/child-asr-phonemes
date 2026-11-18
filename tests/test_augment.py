"""SFT augmentation: speed changes length (and pitch), VTLP shifts pitch but keeps length, off = untouched."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import train_sft as T  # noqa: E402

SR = 16000


def tone(f0=200.0, sec=1.0):
    t = np.arange(int(SR * sec)) / SR
    return np.sin(2 * np.pi * f0 * t).astype(np.float32)


def peak_hz(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.argmax(spec) * SR / len(y)


def test_off_is_identity():
    y = tone()
    assert np.array_equal(T.augment(y), y)
    assert np.array_equal(T.augment(y, (1.0,), 0.0), y)


def test_speed_changes_length_and_pitch():
    y = tone()
    for f in (0.9, 1.1):
        z = T.augment(y, (f,))
        assert abs(len(z) - len(y) / f) <= 2, (f, len(z))
        assert abs(peak_hz(z) - 200 * f) < 5, (f, peak_hz(z))


def test_vtlp_keeps_length_shifts_pitch_in_range():
    torch.manual_seed(0)
    y = tone()
    seen = []
    for _ in range(20):
        z = T.augment(y, (), 0.1)
        assert len(z) == len(y)
        seen.append(peak_hz(z))
    assert min(seen) >= 200 * 0.9 - 5 and max(seen) <= 200 * 1.1 + 5, (min(seen), max(seen))
    assert max(seen) - min(seen) > 10, "warp factor never varied"


def test_specaugment_masks_only_in_train_mode():
    from transformers import WavLMConfig, WavLMForCTC
    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      conv_dim=(32, 32), conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=16,
                      num_conv_pos_embedding_groups=2, vocab_size=10, mask_time_prob=0.5, mask_time_length=2)
    m = WavLMForCTC(cfg)
    x = torch.randn(1, 8000)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x).logits, m(x).logits)
    m.train()
    with torch.no_grad():
        torch.manual_seed(1)
        a = m(x).logits
        torch.manual_seed(2)
        b = m(x).logits
    assert not torch.equal(a, b), "SpecAugment did not change train-mode output"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
