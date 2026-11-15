"""Word labels for the auxiliary word head: what the child said, as lowercase letters.

Alphabet (VERSION w1): a-z, apostrophe, and the word separator. The word head is a CTC
head on letters, used only in training to let word-only clips train the encoder.

Rules aim at what is audible, not the intended word:
  CHAT  [: x] replacements, [/] [*] and other [...] codes dropped; parentheses mark parts
        NOT said, so their contents go ("(be)cause" -> "cause"); &-um fillers and &+fr
        fragments kept as spoken; &=laughs actions dropped; 0word (omitted) dropped;
        @-suffixes, ":" lengthening, "^" dropped; "_" and "+" compounds split.
  MyST  <TAGS> dropped; +UM+ fillers kept; *WORD* (mispronounced) kept; WHE- fragments kept.
  Both  xxx / yyy / (()) / <INDISCERNIBLE> -> flag unintelligible; digits -> flag digits.
normalize_*() returns (text, flags). Text is words joined by single spaces.
"""
import re
import unicodedata

VERSION = "w1"
LETTERS = list("abcdefghijklmnopqrstuvwxyz'")
BLOCKING = {"unintelligible", "digits", "empty", "no_signal"}

_BRACKET = re.compile(r"\[[^\]]*\]")
_BULLET = re.compile(r"\x15\d+_\d+\x15")


def _finish(words, flags):
    out = []
    for w in words:
        w = unicodedata.normalize("NFKD", w.lower().replace("’", "'"))
        w = "".join(ch for ch in w if not unicodedata.combining(ch))
        if any(ch.isdigit() for ch in w):
            flags.add("digits")
        for part in re.split(r"[-_+]", w):
            part = re.sub(r"[^a-z']", "", part).strip("'")
            if part:
                out.append(part)
    if not out:
        flags.add("empty")
    return " ".join(out), flags


def normalize_chat(main):
    """CHAT main tier (with or without the '*CHI:' prefix)."""
    flags = set()
    s = _BULLET.sub(" ", main)
    s = re.sub(r"^\*[A-Z]+:\s*", "", s)
    s = _BRACKET.sub(" ", s).replace("<", " ").replace(">", " ")
    words = []
    for tok in s.split():
        if tok.lower() in ("xxx", "yyy", "www") or tok.lower().startswith(("xxx", "yyy")):
            flags.add("unintelligible")
            continue
        if tok.startswith("&="):          # action / sound, not speech
            continue
        if tok.startswith(("&-", "&+")):  # filler or fragment: spoken
            tok = tok[2:]
        elif tok.startswith("&"):
            tok = tok[1:]
        if tok.startswith("0"):           # omitted word: not said
            continue
        if tok.startswith("+") or re.fullmatch(r"\(\.+\)", tok) or re.fullmatch(r"[.,!?;‡„]+", tok):
            continue                      # terminators, linkers, pauses
        tok = re.sub(r"\([^)]*\)", "", tok)  # not-said parts
        tok = tok.split("@")[0].replace(":", "").replace("^", "")
        words.append(tok)
    return _finish(words, flags)


def normalize_myst(trn):
    flags = set()
    s = trn.strip()
    if "<NO_SIGNAL>" in s:
        flags.add("no_signal")
    if "(())" in s or "<INDISCERNIBLE>" in s:
        flags.add("unintelligible")
    s = re.sub(r"<[^>]*>", " ", s).replace("(())", " ")
    words = [t.strip("+*") for t in s.split()]
    return _finish(words, flags)


def normalize_plain(text):
    """Already-normalized text (SPROUT orthographic_text_normalized)."""
    flags = set()
    if not isinstance(text, str):
        return "", {"empty"}
    return _finish(text.split(), flags)
