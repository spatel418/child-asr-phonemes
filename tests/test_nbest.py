"""Prefix beam search: scores equal the exact CTC probability, and the best string matches brute force.
Run: python tests/test_nbest.py"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import nbest_ceiling as N  # noqa: E402


def ctc_logp(lp, y):
    l = torch.nn.functional.ctc_loss(torch.tensor(lp)[:, None, :], torch.tensor([y if y else [0]])[:, : len(y)],
                                     torch.tensor([lp.shape[0]]), torch.tensor([len(y)]), reduction="sum")
    return -float(l)


def collapse(path):
    out, prev = [], -1
    for c in path:
        if c != prev and c != 0:
            out.append(c)
        prev = c
    return tuple(out)


def test_scores_match_ctc_and_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        Tn, V = 5, 3
        lp = torch.log_softmax(torch.tensor(rng.normal(0, 1.5, (Tn, V))), -1).numpy()
        hyps = N.prefix_beam(lp, beam=1000, prune=-1e9, max_tok=V)   # no pruning: exact
        for h, s in hyps:
            assert abs(s - ctc_logp(lp, h)) < 1e-5, (h, s, ctc_logp(lp, h))
        tot = {}
        for path in itertools.product(range(V), repeat=Tn):
            y = collapse(path)
            tot[y] = np.logaddexp(tot.get(y, -1e30), sum(lp[t, c] for t, c in enumerate(path)))
        assert abs(np.logaddexp.reduce([s for _, s in hyps]) - 0.0) < 1e-6   # all mass covered
        assert tuple(hyps[0][0]) == max(tot, key=tot.get)


if __name__ == "__main__":
    test_scores_match_ctc_and_brute_force()
    print("ok test_scores_match_ctc_and_brute_force")
