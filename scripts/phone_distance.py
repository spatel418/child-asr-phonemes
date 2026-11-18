"""Phonetic distance between phones, and a feature-weighted PER.

Feature-weighted PER (fPER): edit distance where a substitution costs the sound distance between the two
phones (0..1) instead of 1. Deleting or adding a phone still costs 1. Divided by the label length, like PER.

Sound distance (one step on any feature = 1/3, capped at 1):
- consonant vs consonant: number of features that differ among voicing, place, manner, divided by 3.
  t -> d = 1/3 (voicing), t -> k = 1/3 (place), t -> s = 1/3 (manner), t -> m = 1.
- vowel vs vowel: (|height steps| + |backness steps| + round differs + rhotic differs) / 3, capped at 1.
  i -> ɪ = 1/3, i -> ɑ = 1.
- diphthong (aɪ aʊ ɔɪ) = start vowel + end vowel. Diphthong vs vowel: 1/3 + the closer of its two parts.
  Diphthong vs diphthong: mean of start-start and end-end distances.
- consonant vs vowel: 1.
An unknown phone stops the script.
"""
import numpy as np
from numba import njit

CONS = {  # phone: (voiced, place, manner); same table as audio_judge_check.py
    "p": (0, "bilabial", "stop"), "b": (1, "bilabial", "stop"),
    "t": (0, "alveolar", "stop"), "d": (1, "alveolar", "stop"),
    "k": (0, "velar", "stop"), "ɡ": (1, "velar", "stop"), "ʔ": (0, "glottal", "stop"),
    "m": (1, "bilabial", "nasal"), "n": (1, "alveolar", "nasal"), "ŋ": (1, "velar", "nasal"),
    "f": (0, "labiodental", "fricative"), "v": (1, "labiodental", "fricative"),
    "θ": (0, "dental", "fricative"), "ð": (1, "dental", "fricative"),
    "s": (0, "alveolar", "fricative"), "z": (1, "alveolar", "fricative"),
    "ʃ": (0, "postalveolar", "fricative"), "ʒ": (1, "postalveolar", "fricative"),
    "h": (0, "glottal", "fricative"), "ç": (0, "palatal", "fricative"),
    "ɬ": (0, "alveolar", "lateral-fricative"),
    "tʃ": (0, "postalveolar", "affricate"), "dʒ": (1, "postalveolar", "affricate"),
    "l": (1, "alveolar", "lateral"), "ɹ": (1, "alveolar", "approximant"),
    "w": (1, "labial-velar", "approximant"), "j": (1, "palatal", "approximant"),
    "ɾ": (1, "alveolar", "tap"),
}
VOW = {  # phone: (height 1=high..5=low, backness 0=front 1=central 2=back, round, rhotic)
    "i": (1, 0, 0, 0), "ɪ": (2, 0, 0, 0), "e": (3, 0, 0, 0), "ɛ": (4, 0, 0, 0), "æ": (5, 0, 0, 0),
    "u": (1, 2, 1, 0), "ʊ": (2, 2, 1, 0), "o": (3, 2, 1, 0), "ɔ": (4, 2, 1, 0), "ɑ": (5, 2, 0, 0),
    "ʌ": (4, 1, 0, 0), "ə": (3, 1, 0, 0), "ɚ": (3, 1, 0, 1),
}
_A = (5, 1, 0, 0)  # open central start of aɪ / aʊ (not a phone in our set)
DIPH = {"aɪ": (_A, VOW["ɪ"]), "aʊ": (_A, VOW["ʊ"]), "ɔɪ": (VOW["ɔ"], VOW["ɪ"])}


def _vv(a, b):
    return min(1.0, (abs(a[0] - b[0]) + abs(a[1] - b[1]) + (a[2] != b[2]) + (a[3] != b[3])) / 3)


def dist(p, q):
    """Sound distance between two phones, 0 (same) .. 1."""
    for x in (p, q):
        if x not in CONS and x not in VOW and x not in DIPH:
            raise ValueError(f"unknown phone {x!r}: add it to phone_distance.py")
    if p == q:
        return 0.0
    if p in CONS and q in CONS:
        return sum(u != v for u, v in zip(CONS[p], CONS[q])) / 3
    if p in CONS or q in CONS:
        return 1.0
    if p in DIPH and q in DIPH:
        return (_vv(DIPH[p][0], DIPH[q][0]) + _vv(DIPH[p][1], DIPH[q][1])) / 2
    if p in DIPH or q in DIPH:
        d, v = (p, q) if p in DIPH else (q, p)
        return min(1.0, 1 / 3 + min(_vv(VOW[v], DIPH[d][0]), _vv(VOW[v], DIPH[d][1])))
    return _vv(VOW[p], VOW[q])


def cost_matrix(vocab):
    """Substitution costs indexed by vocab id. Id 0 is the CTC blank and never appears in a phone string."""
    n = len(vocab)
    C = np.ones((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            C[i, j] = dist(vocab[i], vocab[j])
    return C


@njit(cache=True)
def _wed(ref, hyp, C):
    m, n = len(ref), len(hyp)
    prev = np.arange(n + 1).astype(np.float64)
    cur = np.empty(n + 1, dtype=np.float64)
    for i in range(1, m + 1):
        cur[0] = i
        r = ref[i - 1]
        for j in range(1, n + 1):
            a = prev[j - 1] + C[r, hyp[j - 1]]
            b = prev[j] + 1.0
            c = cur[j - 1] + 1.0
            cur[j] = a if a < b and a < c else (b if b < c else c)
        prev, cur = cur, prev
    return prev[n]


def weighted_distance(ref_ids, hyp_ids, C):
    return float(_wed(np.asarray(ref_ids, dtype=np.int64), np.asarray(hyp_ids, dtype=np.int64), C))


def fper(ref_ids, hyp_ids, C):
    return weighted_distance(ref_ids, hyp_ids, C) / max(len(ref_ids), 1)
