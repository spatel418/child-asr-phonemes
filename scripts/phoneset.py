"""Shared phone set: turn any corpus's IPA label into the tokens the model learns.

Labs write the same sound in different ways (ʧ / t͡ʃ / tʃ, ɑ͡ɪ / aɪ, r / ɹ), and
add marks that follow the lab, not the child (stress, length, aspiration). This
maps every label to one simplified set. The original label is always kept in
the manifest, so nothing is lost.

    from phoneset import normalize
    words, flags = normalize("ˈʧɪcən *")
    # words = [['tʃ', 'ɪ', 'k', 'ə', 'n']], flags = {'untranscribed'}

Unknown symbols raise UnknownSymbol. Add them to the tables below with a
decision; never drop them silently.

Decisions (2026-11-15, v1, see research log):
  - all diacritics and stress removed; eɪ->e, oʊ->o; one rhotic vowel ɚ
  - affricates tʃ dʒ are one token; ʦ ʣ are split into t s / d z
  - ʔ ɬ ç kept (seen in SSD errors)
  - `*` (untranscribed word) and xxx / χχχ (unintelligible) flag the clip
"""
import re
import unicodedata

VERSION = "v1"

PHONES = [
    # vowels
    "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ʌ", "ə", "ɔ", "o", "ʊ", "u", "ɚ", "aɪ", "aʊ", "ɔɪ",
    # consonants
    "p", "b", "t", "d", "k", "ɡ", "m", "n", "ŋ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ",
    "h", "tʃ", "dʒ", "l", "ɹ", "w", "j", "ɾ", "ʔ", "ɬ", "ç",
]
WORD_SEP = "|"

# Single base letters -> phone(s). Letters already in PHONES map to themselves.
BASE = {
    "g": "ɡ", "r": "ɹ", "ɫ": "l", "c": "k", "x": "k", "χ": "k",
    "a": "ɑ", "ɒ": "ɑ", "ɐ": "ʌ", "ɜ": "ɚ", "ɝ": "ɚ", "ɵ": "o", "ɤ": "o", "ɨ": "ɪ",
    "ɯ": "u", "œ": "ɛ", "ø": "e", "y": "i", "ʏ": "ʊ",
    "ʧ": "tʃ", "ʤ": "dʒ", "ʨ": "tʃ", "ʥ": "dʒ", "ʦ": "t s", "ʣ": "d z",
    "ʍ": "w", "β": "v", "ɸ": "f", "ʋ": "v", "ʂ": "ʃ", "ʝ": "j", "ɟ": "ɡ", "ɣ": "ɡ",
    "ʁ": "ɹ", "ɺ": "ɹ", "ɻ": "ɹ", "ʙ": "b", "ɲ": "n", "ɳ": "n", "ɴ": "n", "ɱ": "m",
    "ʈ": "t", "ɖ": "d", "ɢ": "ɡ", "ʕ": "h", "ħ": "h", "ɦ": "h", "λ": "l",
}
BASE.update({p: p for p in PHONES if len(p) == 1})

# Marks removed (stress, length, aspiration, off-glides, tone/boundary marks).
STRIP_MODIFIERS = set("ˈˌːˑʰʷʲˡᵊˀ˗˕˻˺˸ᶾʼ")
RHOTIC = set("˞ʳʴ")          # make the vowel before them ɚ
TIE = {"͡", "͜"}   # tie bars: join the two letters into one phone
# Punctuation and CHAT/Phon markers removed.
STRIP_PUNCT = set("().^‹›,'~\"")

# Two-letter phones built from tied or untied pairs.
PAIRS = {
    ("a", "ɪ"): "aɪ", ("ɑ", "ɪ"): "aɪ", ("a", "ʊ"): "aʊ", ("ɑ", "ʊ"): "aʊ",
    ("ɔ", "ɪ"): "ɔɪ", ("o", "ɪ"): "ɔɪ", ("e", "ɪ"): "e", ("o", "ʊ"): "o",
    ("t", "ʃ"): "tʃ", ("d", "ʒ"): "dʒ",
}
UNTIED_OK = {("a", "ɪ"), ("a", "ʊ"), ("ɔ", "ɪ"), ("e", "ɪ"), ("o", "ʊ"), ("t", "ʃ"), ("d", "ʒ")}

UNINTELLIGIBLE = re.compile(r"^(x{2,}|χ{2,}|y{3,})$")
PLACEHOLDER = {"V", "C"}  # Phon "some vowel/consonant": flag, don't guess


class UnknownSymbol(ValueError):
    pass


def _segments(word):
    """Split one word into (letters, rhotic) segments; tied letters stay together."""
    s = unicodedata.normalize("NFD", word.replace("ç", "\uE000")).replace("\uE000", "ç")
    segs, i = [], 0
    while i < len(s):
        ch = s[i]
        i += 1
        if ch in STRIP_MODIFIERS:          # e.g. stress before the letter
            continue
        if ch in RHOTIC or unicodedata.category(ch) == "Mn":
            if not segs:  # stray mark (e.g. after a stripped ʰ): nothing to attach to
                continue
            if ch in RHOTIC:
                segs[-1][1] = True
            continue
        letters, rhotic, tied = [ch], False, False
        while i < len(s):
            m = s[i]
            if m in TIE:
                tied = True
            elif m in RHOTIC:
                rhotic = True
            elif unicodedata.category(m) == "Mn" or m in STRIP_MODIFIERS:
                pass
            elif tied:
                letters.append(m)
                tied = False
            else:
                break
            i += 1
        segs.append([letters, rhotic])
    return segs


def _phones(word):
    segs = _segments(word)
    # join untied pairs (e.g. Providence tʃ, eɪ, aɪ)
    joined = []
    for letters, rhotic in segs:
        if (joined and len(joined[-1][0]) == 1 and len(letters) == 1
                and (joined[-1][0][0], letters[0]) in UNTIED_OK and not joined[-1][1]):
            joined[-1] = [joined[-1][0] + letters, rhotic]
        else:
            joined.append([letters, rhotic])
    out, prev_src = [], None
    for letters, rhotic in joined:
        if len(letters) == 2 and tuple(letters) in PAIRS:
            ph = [PAIRS[tuple(letters)]]
        else:
            ph = []
            for ch in letters:
                if ch not in BASE:
                    raise UnknownSymbol(f"{ch!r} (U+{ord(ch):04X} {unicodedata.name(ch, '?')}) in {word!r}")
                ph.extend(BASE[ch].split())
        if rhotic:  # r-colored vowel -> ɚ; a hook on a consonant is dropped
            ph = ["ɚ" if p in VOWELS else p for p in ph]
        # ɜr / ɝr: the r is part of the rhotic vowel
        if prev_src in ("ɜ", "ɝ") and letters in (["r"], ["ɹ"]):
            ph = []
        # kx / kχ: noisy release of the k, not a second k
        if letters in (["x"], ["χ"]) and out and out[-1] == "k":
            ph = []
        out.extend(ph)
        prev_src = letters[-1]
    return out


VOWELS = {"i", "ɪ", "e", "ɛ", "æ", "ɑ", "ʌ", "ə", "ɔ", "o", "ʊ", "u", "ɚ", "aɪ", "aʊ", "ɔɪ"}


def normalize(label):
    """Return (words, flags). words: list of phone lists. flags: set of strings."""
    flags = set()
    text = unicodedata.normalize("NFC", label).replace(" ", " ").replace("\t", " ")
    if "*" in text:
        flags.add("untranscribed")
        text = text.replace("*", " ")
    words = []
    for w in text.split():
        if UNINTELLIGIBLE.match(w):
            flags.add("unintelligible")
            continue
        if w in PLACEHOLDER:
            flags.add("placeholder")
            continue
        if any(ch in PLACEHOLDER for ch in w):  # e.g. dVl: a vowel the labeler left open
            flags.add("placeholder")
            w = "".join(ch for ch in w if ch not in PLACEHOLDER)
        w = "".join(ch for ch in w if ch not in STRIP_PUNCT)
        if not w:
            continue
        ph = _phones(w)
        if ph:
            words.append(ph)
    if not words:
        flags.add("empty_label")
    return words, flags


def to_string(words):
    """Space-separated phones, words joined by ' | '."""
    return f" {WORD_SEP} ".join(" ".join(w) for w in words)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        w, f = normalize(arg)
        print(f"{arg}\t{to_string(w)}\t{sorted(f)}")
