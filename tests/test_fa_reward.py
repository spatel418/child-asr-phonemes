import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import target_bias as TB  # noqa: E402
import train_grpo as G  # noqa: E402
import train_sft as T  # noqa: E402


def ids(s):
    return [T.TOK[p] for p in s.split()]


def item(target, human):
    tgt = target.split()
    h_al = TB.align(tgt, human.split())
    mask = [t in TB.CLASS and h is not None and TB.ok(t, h) for t, h in zip(tgt, h_al)]
    return {"labels": ids(human), "fa": (tgt, mask)}


def test_mask_marks_only_correct_consonants():
    it = item("k æ t", "t æ t")          # child fronted /k/ -> [t]; /t/ correct
    assert it["fa"][1] == [False, False, True]


def test_false_alarm_counts_dropped_correct_consonant():
    it = item("k æ t", "t æ t")
    assert G.false_alarm(it, ids("t æ t")) == 0.0      # matches the human: no false alarm
    assert G.false_alarm(it, ids("k æ t")) == 0.0      # target bias at /k/ is not a false alarm
    assert G.false_alarm(it, ids("t æ")) == 1.0        # dropped the correct final /t/


def test_flap_counts_as_target():
    it = item("w ɔ t ɚ", "w ɔ ɾ ɚ")
    assert it["fa"][1] == [True, False, True, False]
    assert G.false_alarm(it, ids("w ɔ ɾ ɚ")) == 0.0


def test_reward_per_fa_adds_penalty_and_perfect_is_zero():
    it = item("k æ t", "t æ t")
    a = argparse.Namespace(reward="per_fa", fa_weight=1.0)
    assert G.reward(it, ids("t æ t"), a) == 0.0
    assert abs(G.reward(it, ids("t æ"), a) - (-1 / 3 - 1.0)) < 1e-9
    assert G.reward({"labels": ids("t æ t")}, ids("t æ"), a) == -1 / 3   # no target: plain PER
    assert G.reward(it, ids("t æ"), argparse.Namespace(reward="per", fa_weight=1.0)) == -1 / 3


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
