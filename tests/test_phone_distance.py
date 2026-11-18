"""Feature-weighted PER: hand-checked distances and DP vs a plain-Python version."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import phone_distance as PD  # noqa: E402
import train_sft as T  # noqa: E402

C = PD.cost_matrix(T.VOCAB)
ID = {p: i for i, p in enumerate(T.VOCAB)}


def test_hand_distances():
    third = 1 / 3
    cases = {("t", "d"): third, ("t", "k"): third, ("t", "s"): third, ("t", "m"): 1.0, ("t", "t"): 0.0,
             ("i", "ɪ"): third, ("i", "ɑ"): 1.0, ("ə", "ɚ"): third, ("t", "i"): 1.0, ("w", "u"): 1.0,
             ("aɪ", "ɪ"): third, ("aɪ", "aʊ"): 0.5, ("ɹ", "w"): third}
    for (p, q), want in cases.items():
        assert abs(PD.dist(p, q) - want) < 1e-9, (p, q, PD.dist(p, q), want)
        assert abs(PD.dist(q, p) - want) < 1e-9, "not symmetric"


def test_every_vocab_phone_covered_and_bounded():
    assert C.shape == (len(T.VOCAB), len(T.VOCAB))
    assert (C[1:, 1:] >= 0).all() and (C[1:, 1:] <= 1).all()
    assert all(C[i, i] == 0 for i in range(1, len(T.VOCAB)))


def test_unknown_phone_stops():
    try:
        PD.dist("t", "ʀ")
    except ValueError:
        return
    raise AssertionError("unknown phone did not stop")


def slow(ref, hyp):
    D = [[0.0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        D[i][0] = i
    for j in range(len(hyp) + 1):
        D[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            D[i][j] = min(D[i - 1][j - 1] + C[ref[i - 1], hyp[j - 1]], D[i - 1][j] + 1, D[i][j - 1] + 1)
    return D[-1][-1]


def test_dp_matches_slow_and_bounds():
    rng = random.Random(0)
    for _ in range(500):
        ref = [rng.randrange(1, len(T.VOCAB)) for _ in range(rng.randrange(0, 15))]
        hyp = [rng.randrange(1, len(T.VOCAB)) for _ in range(rng.randrange(0, 15))]
        w = PD.weighted_distance(ref, hyp, C)
        assert abs(w - slow(ref, hyp)) < 1e-9
        from rapidfuzz.distance import Levenshtein
        assert w <= Levenshtein.distance(ref, hyp) + 1e-9   # never worse than plain edit distance
    t, d, m = ID["t"], ID["d"], ID["m"]
    assert abs(PD.fper([t, t], [t, d], C) - 1 / 6) < 1e-9
    assert PD.fper([t, t], [t, m], C) == 0.5 and PD.fper([t, t], [t], C) == 0.5 and PD.fper([], [], C) == 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
