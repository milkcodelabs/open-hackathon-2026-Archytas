"""Greek number verbalization for LM training text.

Digits cannot be emitted by the CTC alphabet, so training text must contain the spoken
forms. Policy (documented in README.md): integers and decimals are verbalized as neuter
cardinals ("1980" -> "χίλια εννιακόσια ογδόντα"); gender agreement with the noun is unknown
without parsing, so the neuter default is used (the LM still learns the common spoken
forms; "δύο", "πέντε" etc. are invariant). "%" -> "τοις εκατό". Ordinals are not handled.
"""

from __future__ import annotations

import re

_UNITS = ["", "ένα", "δύο", "τρία", "τέσσερα", "πέντε", "έξι", "επτά", "οκτώ", "εννέα", "δέκα",
          "έντεκα", "δώδεκα", "δεκατρία", "δεκατέσσερα", "δεκαπέντε", "δεκαέξι", "δεκαεπτά", "δεκαοκτώ", "δεκαεννέα"]
_TENS = ["", "", "είκοσι", "τριάντα", "σαράντα", "πενήντα", "εξήντα", "εβδομήντα", "ογδόντα", "ενενήντα"]
_HUNDREDS = ["", "εκατό", "διακόσια", "τριακόσια", "τετρακόσια", "πεντακόσια", "εξακόσια", "επτακόσια", "οκτακόσια", "εννιακόσια"]
_HUNDREDS_F = ["", "εκατό", "διακόσιες", "τριακόσιες", "τετρακόσιες", "πεντακόσιες", "εξακόσιες", "επτακόσιες", "οκτακόσιες", "εννιακόσιες"]


def _below_thousand(n: int, feminine: bool = False) -> list[str]:
    out: list[str] = []
    h, r = divmod(n, 100)
    if h:
        if h == 1 and r:
            out.append("εκατόν")
        else:
            out.append((_HUNDREDS_F if feminine else _HUNDREDS)[h])
    if r:
        if r < 20:
            w = _UNITS[r]
            if feminine and r in (1, 3, 4):
                w = {1: "μία", 3: "τρεις", 4: "τέσσερις"}[r]
            out.append(w)
        else:
            t, u = divmod(r, 10)
            out.append(_TENS[t])
            if u:
                w = _UNITS[u]
                if feminine and u in (1, 3, 4):
                    w = {1: "μία", 3: "τρεις", 4: "τέσσερις"}[u]
                out.append(w)
    return out


def cardinal(n: int) -> str:
    """Neuter cardinal for 0 <= n < 10**12 ("χιλιάδες" is feminine and agrees accordingly)."""
    if n == 0:
        return "μηδέν"
    if n < 0:
        return "μείον " + cardinal(-n)
    if n >= 10**12:  # phone numbers, ids: read digit by digit
        return " ".join(cardinal(int(d)) for d in str(n))
    words: list[str] = []
    billions, n = divmod(n, 10**9)
    millions, n = divmod(n, 10**6)
    thousands, rest = divmod(n, 1000)
    if billions:
        words += (["ένα δισεκατομμύριο"] if billions == 1 else _below_thousand(billions) + ["δισεκατομμύρια"])
    if millions:
        words += (["ένα εκατομμύριο"] if millions == 1 else _below_thousand(millions) + ["εκατομμύρια"])
    if thousands:
        if thousands == 1:
            words.append("χίλια")
        else:
            words += _below_thousand(thousands, feminine=True) + ["χιλιάδες"]
    if rest:
        words += _below_thousand(rest)
    return " ".join(words)


_NUM = re.compile(r"(?<![\w])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(\s?%)?(?![\w])")


def verbalize_digits(text: str) -> str:
    """Replace digit tokens with spoken Greek. Thousands dots and decimal commas are Greek style."""

    def repl(m: re.Match) -> str:
        whole = int(m.group(1).replace(".", ""))
        s = cardinal(whole)
        if m.group(2):
            s += " κόμμα " + " ".join(cardinal(int(d)) for d in m.group(2))
        if m.group(3):
            s += " τοις εκατό"
        return s

    return _NUM.sub(repl, text)


def has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)
