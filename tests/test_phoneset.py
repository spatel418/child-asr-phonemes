"""Tests for scripts/phoneset.py. Run: python -m pytest tests/  (or python tests/test_phoneset.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from phoneset import PHONES, UnknownSymbol, normalize, to_string  # noqa: E402


def norm(label):
    words, flags = normalize(label)
    return to_string(words), flags


def test_same_sound_different_spelling():
    # affricates: tied, digraph, untied
    assert norm("t͡ʃɪp")[0] == norm("ʧɪp")[0] == norm("tʃɪp")[0] == "tʃ ɪ p"
    assert norm("d͡ʒæm")[0] == norm("ʤæm")[0] == "dʒ æ m"
    # /aɪ/ written ɑ͡ɪ or aɪ
    assert norm("mɑ͡ɪ")[0] == norm("maɪ")[0] == "m aɪ"
    # /eɪ oʊ/ merged into e o
    assert norm("ste͡ɪ")[0] == norm("steɪ")[0] == norm("ste")[0] == "s t e"
    assert norm("go͡ʊ")[0] == norm("ɡoʊ")[0] == "ɡ o"
    # r / ɹ, ASCII g / ɡ, dark l
    assert norm("rɛd")[0] == norm("ɹɛd")[0] == "ɹ ɛ d"
    assert norm("gʌm")[0] == "ɡ ʌ m"
    assert norm("hoʊɫ")[0] == "h o l"


def test_marks_removed():
    assert norm("ˈbʌˌtɚ")[0] == "b ʌ t ɚ"           # stress
    assert norm("ˈhiː")[0] == "h i"                 # length
    assert norm("tʰə")[0] == "t ə"                  # aspiration
    assert norm("s̪ʌn")[0] == "s ʌ n"                # dental
    assert norm("pæ~n")[0] == norm("pæ̃n")[0] == "p æ n"  # nasal, ASCII and real
    assert norm("ĩ")[0] == "i"                      # precomposed + diacritic


def test_rhotic_vowels():
    for label in ["ɝ", "ɜ˞", "ə˞", "ɚ", "ɜr", "ɜːr"]:
        assert norm(label)[0] == "ɚ", label


def test_kept_ssd_sounds():
    assert norm("ɬu")[0] == "ɬ u"
    assert norm("ʔʌ")[0] == "ʔ ʌ"
    assert norm("çpəˈɡedi")[0] == "ç p ə ɡ e d i"   # ç must not become c -> k


def test_markers_and_flags():
    s, f = norm("* ˈstʌk (.) ‹ˈfwɑ ki›")
    assert s == "s t ʌ k | f w ɑ | k i" and "untranscribed" in f
    s, f = norm("χχχ")
    assert s == "" and {"unintelligible", "empty_label"} <= f
    assert "unintelligible" in norm("ɑ͡ɪ gɛs xxx ðæt")[1]
    assert norm("ˈbu^diz")[0] == "b u d i z"
    assert "placeholder" in norm("dVl")[1]
    assert "typo" in norm("ʃt͡ʃɚʊB")[1]


def test_k_release_not_doubled():
    assert norm("kχʊkχi")[0] == "k ʊ k i"


def test_split_ts():
    assert norm("ˈræbɪʦ")[0] == "ɹ æ b ɪ t s"


def test_unknown_symbol_stops():
    try:
        normalize("kæ☃t")
    except UnknownSymbol:
        return
    raise AssertionError("unknown symbol was not caught")


def test_output_only_known_phones():
    words, _ = normalize("ˈʧɪcən ɑ͡ɪ gɛs ðə ˈbiːˌhaɪv ˈɡɑt̚ ˈnɑx ˈdãʊ̃n")
    assert all(p in PHONES for w in words for p in w)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
