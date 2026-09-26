"""Build el_homophones.bin: every spelling of every Greek sound in the LM vocabulary.

    voicetotext homophones el_homophones.bin --kenlm lm3_300k.bin --vocab lm3_300k.vocab

Read by HomophoneIndex.kt. The released file was built from the same vocabulary and KenLM
binary as el_3gram.gvtlm. ``--arpa`` reads the unigram probabilities straight from an ARPA
file (no kenlm module); near-equal spellings may then be ordered slightly differently,
because the release used the 8-bit quantized KenLM binary's probabilities.

Sound key rules, in order (SoundKey.kt implements the same rules and must stay identical):
  1. lowercase, Unicode NFD; drop the stress mark U+0301; a diaeresis (U+0308) turns its ι/υ
     into a stand-alone /i/ (so "οϊ" is o-i, not the digraph "οι"); other diaereses dropped;
  2. "αυ" -> "av", "ευ" -> "ev", "ηυ" -> "iv";
  3. digraphs: "ει" "οι" "υι" -> "i", "αι" -> "e", "ου" -> "u";
  4. letters: ι η υ -> "i", ο ω -> "o", ε -> "e", ς -> σ;
  5. doubled consonants become single (λλ μμ νν ππ ρρ σσ ττ κκ ββ φφ θθ χχ δδ ζζ ξξ ψψ);
     γγ stays (it is /ng/);
  then NFC. Example: "δήμου" and "δίμου" both give "δiμu" (μ stays; only the listed vowels change).

File layout, all little-endian:

    magic "HOMIDX1\\0"               8 bytes
    n                                int32, number of sound keys
    keyOffsets[n + 1]                int32 each, byte offsets into the key blob
    valueOffsets[n + 1]              int32 each, byte offsets into the value blob
    keyBlob                          UTF-8 keys, concatenated
    valueBlob                        UTF-8 values, concatenated; one value per key: the spellings
                                     joined by single spaces, most frequent first

Keys are sorted by their UTF-8 bytes. They only contain Greek and Latin letters (all below
U+D800), where byte order equals String.compareTo, so the phone binary-searches them.
The released file has 233,383 sound keys, 45,478 of them with more than one spelling.
"""

from __future__ import annotations

import struct
import unicodedata
from pathlib import Path

ACUTE = "́"
DIAERESIS = "̈"
_LONE_I = "\u0001"          # placeholder for a diaeresis vowel, always /i/

_DIGRAPHS = (("αυ", "av"), ("ευ", "ev"), ("ηυ", "iv"),
             ("ει", "i"), ("οι", "i"), ("υι", "i"), ("αι", "e"), ("ου", "u"))
_SINGLES = str.maketrans({"ι": "i", "η": "i", "υ": "i", _LONE_I: "i", "ο": "o", "ω": "o", "ε": "e", "ς": "σ"})
_DOUBLES = "λμνπρστκβφθχδζξψ"
MAGIC = b"HOMIDX1\x00"


def sound_key(word: str) -> str:
    s = unicodedata.normalize("NFD", word.lower()).replace(ACUTE, "")
    s = s.replace("ι" + DIAERESIS, _LONE_I).replace("υ" + DIAERESIS, _LONE_I).replace(DIAERESIS, "")
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = s.translate(_SINGLES)
    for c in _DOUBLES:
        s = s.replace(c + c, c)
    return unicodedata.normalize("NFC", s)


def group(words: list[str], logp: dict[str, float]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for w in words:
        if w and not w.startswith("<"):
            groups.setdefault(sound_key(w), []).append(w)
    for g in groups.values():
        g.sort(key=lambda w: -logp.get(w, -99.0))     # stable: ties keep vocabulary order
    return groups


def unigrams_from_kenlm(model_path: Path, words: list[str]) -> dict[str, float]:
    import kenlm

    m = kenlm.Model(str(model_path))
    return {w: m.score(w, bos=False, eos=False) for w in words}


def unigrams_from_arpa(arpa: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    in_uni = False
    with open(arpa, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(chr(92)):
                if in_uni:
                    break
                in_uni = line == chr(92) + "1-grams:"
                continue
            if in_uni and line:
                p = line.split("\t")
                out[p[1]] = float(p[0])
    return out


def save_binary(groups: dict[str, list[str]], path: Path) -> Path:
    items = sorted((k.encode("utf-8"), " ".join(v).encode("utf-8")) for k, v in groups.items())
    kofs, vofs, ko, vo = [0], [0], 0, 0
    for k, v in items:
        ko += len(k)
        vo += len(v)
        kofs.append(ko)
        vofs.append(vo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", len(items)))
        f.write(struct.pack(f"<{len(kofs)}i", *kofs))
        f.write(struct.pack(f"<{len(vofs)}i", *vofs))
        for k, _ in items:
            f.write(k)
        for _, v in items:
            f.write(v)
    return path


def load_binary(path: Path) -> dict[str, list[str]]:
    """Reader, for checking a file (HomophoneIndex.kt does the same with mmap)."""
    b = Path(path).read_bytes()
    if b[:8] != MAGIC:
        raise ValueError(f"{path} is not a homophone-index file")
    (n,) = struct.unpack_from("<i", b, 8)
    o = 12
    kofs = struct.unpack_from(f"<{n + 1}i", b, o)
    o += 4 * (n + 1)
    vofs = struct.unpack_from(f"<{n + 1}i", b, o)
    o += 4 * (n + 1)
    kb, vb = o, o + kofs[-1]
    return {b[kb + kofs[i]:kb + kofs[i + 1]].decode("utf-8"): b[vb + vofs[i]:vb + vofs[i + 1]].decode("utf-8").split(" ")
            for i in range(n)}


def build(vocab: Path, out: Path, *, kenlm: Path | None = None, arpa: Path | None = None) -> dict[str, int]:
    words = vocab.read_text(encoding="utf-8").split()
    logp = unigrams_from_kenlm(kenlm, words) if kenlm else unigrams_from_arpa(arpa)  # type: ignore[arg-type]
    groups = group(words, logp)
    save_binary(groups, out)
    return {"keys": len(groups), "ambiguous": sum(1 for g in groups.values() if len(g) > 1)}
