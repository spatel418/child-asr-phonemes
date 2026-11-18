import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import midtrain as M  # noqa: E402


def test_mix_weights_gives_adult_share_in_seconds():
    rng = np.random.default_rng(0)
    rows = [{"dur": float(d), "kind": "adult"} for d in rng.uniform(0.5, 3, 200)] + \
           [{"dur": float(d), "kind": "child"} for d in rng.uniform(0.5, 15, 5000)]
    p = M.mix_weights(rows, 0.15)
    d = np.array([r["dur"] for r in rows])
    a = np.array([r["kind"] == "adult" for r in rows])
    assert abs(p.sum() - 1) < 1e-9
    assert abs((p * d)[a].sum() / (p * d).sum() - 0.15) < 1e-9


def test_heldout_is_by_speaker_and_about_one_percent():
    who = [f"kid{i}" for i in range(20000)]
    share = np.mean([M.heldout(w) for w in who])
    assert 0.005 < share < 0.015
    assert M.heldout("kid7") == M.heldout("kid7")
