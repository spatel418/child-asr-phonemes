import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from wordset import normalize_chat, normalize_myst, normalize_plain  # noqa: E402


def test_chat_codes():
    assert normalize_chat("*CHI:\t<and he> [/] and he goed [: went] [*] home . \x151200_2400\x15")[0] == \
        "and he and he goed home"


def test_chat_not_said_parts_and_omitted():
    assert normalize_chat("(be)cause 0is goin(g) there .")[0] == "cause goin there"


def test_chat_fillers_actions_fragments():
    t, f = normalize_chat("&-um &=laughs &+fr frog@l is ice+cream_cone no: +...")
    assert t == "um fr frog is ice cream cone no" and not f


def test_chat_unintelligible_and_digits():
    assert "unintelligible" in normalize_chat("I want xxx .")[1]
    assert "digits" in normalize_chat("I am 5 .")[1]
    assert "empty" in normalize_chat("&=laughs .")[1]


def test_myst():
    t, f = normalize_myst("<BREATH> +UM+ IT’S A *MILLITER* D-CELL WHE- <NOISE>")
    assert t == "um it's a militer d cell whe".replace("militer", "milliter") and not f
    assert "unintelligible" in normalize_myst("I THINK (()) YES")[1]
    assert "no_signal" in normalize_myst("<NO_SIGNAL>")[1]


def test_plain():
    assert normalize_plain("the dog's ball")[0] == "the dog's ball"
    assert "empty" in normalize_plain(float("nan"))[1]
