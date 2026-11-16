"""GRPO pieces: path collapse, reward, group stats, loss (KL, dropped groups, gradient direction).
Run: python tests/test_grpo.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402
from test_word_head import mask, model_with_heads, x  # noqa: E402  (tiny WavLM + LoRA + word head)


def test_collapse():
    p = torch.tensor([[[0, 1, 1, 0, 1, 2, 2, 0], [3, 3, 3, 0, 0, 0, 0, 0]]])
    assert G.collapse(p) == [[[1, 1, 2], [3]]]


def test_per_matches_edit_ops():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ref = rng.integers(1, 6, rng.integers(1, 9)).tolist()
        hyp = rng.integers(1, 6, rng.integers(0, 9)).tolist()
        assert abs(G.per(ref, hyp) - sum(T.edit_ops(ref, hyp)) / len(ref)) < 1e-9


def test_sample_paths_respects_lengths():
    logits = torch.randn(2, 7, 5)
    n = torch.tensor([7, 3])
    s = G.sample_paths(logits, n, 4, 1.0)
    assert s.shape == (2, 4, 7) and int(s[1, :, 3:].abs().sum()) == 0


def test_group_stats():
    r = np.array([[-0.2, -0.4, -0.2, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.5, -0.5, -0.5, -0.5]])
    adv, keep, solved = G.group_stats(r)
    assert keep.tolist() == [True, False, False] and solved.tolist() == [False, True, False]
    assert abs(adv[0].mean()) < 1e-9 and (adv[1:] == 0).all()
    assert adv[0, 3] > 0 > adv[0, 1]


def test_frame_logits_match_hf():
    m = model_with_heads()
    lg, n = G.frame_logits(m, x, mask)
    assert torch.allclose(lg, m(input_values=x, attention_mask=mask).logits, atol=1e-5)
    assert n.tolist() == m.get_base_model()._get_feat_extract_output_lengths(mask.sum(-1)).tolist()


def test_kl_zero_against_itself_and_dropped_groups_give_no_grad():
    m = model_with_heads()
    lg, n = G.frame_logits(m, x, mask)
    paths = G.sample_paths(lg, n, 4, 1.0)
    adv = torch.randn(3, 4)
    loss, st = G.grpo_loss(lg, n, paths, adv, torch.tensor([True, True, True]), lg.detach(), 0.03, 1.0, 0.2)
    assert abs(st["kl"]) < 1e-6
    loss0, _ = G.grpo_loss(lg, n, paths, adv, torch.tensor([False, False, False]), lg.detach() + 1.0, 0.03, 1.0, 0.2)
    loss0.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.parameters())


def test_step_raises_odds_of_better_path():
    torch.manual_seed(0)
    m = model_with_heads()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.5)
    lg, n = G.frame_logits(m, x[:1], mask[:1])
    paths = G.sample_paths(lg, n, 2, 1.0)
    adv = torch.tensor([[1.0, -1.0]])   # path 0 good, path 1 bad

    def seq_lp(lg_):
        lp = torch.log_softmax(lg_.float(), -1)
        return lp.gather(-1, paths.permute(0, 2, 1)).permute(0, 2, 1)[..., : int(n[0])].sum(-1)[0]

    before = seq_lp(lg).detach()
    loss, _ = G.grpo_loss(lg, n, paths, adv, torch.tensor([True]), lg.detach(), 0.0, 1.0, 0.2)
    loss.backward()
    opt.step()
    after = seq_lp(G.frame_logits(m, x[:1], mask[:1])[0]).detach()
    assert (after[0] - before[0]) > (after[1] - before[1]), (before, after)


def test_mutual_info_basic():
    # every frame the same phone odds -> MI 0; different confident phones per frame -> MI > 0
    same = torch.zeros(1, 4, 5)
    same[..., 2] = 5.0
    diff = torch.zeros(1, 4, 5)
    for t in range(4):
        diff[0, t, 1 + t] = 8.0
    n = torch.tensor([4])
    assert abs(G.mutual_info(same, n)[0]) < 1e-6 and G.mutual_info(diff, n)[0] > 1.0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
