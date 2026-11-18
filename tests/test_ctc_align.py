import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import deletion_direction as DD  # noqa: E402


def collapse(path):
    out, prev = [], None
    for p in path:
        if p != prev and p != 0:
            out.append(p)
        prev = p
    return out


def brute(lp, labels):
    """Best frame path (by log-prob) that collapses to labels."""
    T, V = lp.shape
    best, arg = -1e30, None
    for path in itertools.product(range(V), repeat=T):
        if collapse(path) == list(labels):
            s = sum(lp[t, p] for t, p in enumerate(path))
            if s > best:
                best, arg = s, path
    return best, arg


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(30):
        T, V = rng.integers(3, 7), 4
        lab = list(rng.integers(1, V, size=rng.integers(1, 4)))
        x = rng.normal(size=(T, V))
        lp = x - np.log(np.exp(x).sum(1, keepdims=True))
        b, path = brute(lp, lab)
        al = DD.ctc_align(lp, lab)
        if path is None:
            assert al is None
            continue
        assert al is not None
        # rebuild the frame path from the alignment and check its score equals the brute-force best
        rebuilt = [0 if j == -1 else lab[j] for j in al]
        assert collapse(rebuilt) == lab
        s = sum(lp[t, p] for t, p in enumerate(rebuilt))
        assert abs(s - b) < 1e-9


if __name__ == "__main__":
    test_viterbi_matches_brute_force()
    print("ok")
