"""Convert an ARPA language model into a compact format a phone can mmap from Kotlin.

KenLM's binary is a C++ trie; reading it from Kotlin would mean shipping the NDK. This
writes the same information as sorted arrays instead: simple, mmap-friendly, and searched
with a binary search per lookup (NgramLm.kt in the app). It is larger than KenLM's trie,
which exploits the prefix structure, but it can be verified line by line against the ARPA
it came from.

Layout, all little-endian:

    magic "NGRAM1\0\0"        8 bytes
    order, vocabSize           2 x int32
    counts[order]              int32 each
    vocabBlobLength            int32
    vocabOffsets[vocabSize+1]  int32 each, into the blob
    vocabBlob                  UTF-8 words, sorted, concatenated
    unigrams[vocabSize]        float32 logp, float32 backoff
    bigrams[count2]            int64 key, float32 logp, float32 backoff
    trigrams[count3]           int64 key, float32 logp

Word ids are the index into the sorted vocabulary. An n-gram key packs ids at 21 bits each,
most recent word in the low bits, which keeps every key inside a signed 64-bit integer for
vocabularies up to 2,097,152 words.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

log = logging.getLogger("voicetotext.lm.export")

MAGIC = b"NGRAM1" + bytes(2)
END = chr(92) + "end" + chr(92)
ID_BITS = 21
MAX_VOCAB = 1 << ID_BITS


def pack_key(ids: list[int]) -> int:
    k = 0
    for i in ids:
        k = (k << ID_BITS) | i
    return k


def read_arpa(path: Path) -> tuple[list[int], dict[int, list]]:
    """Returns (counts, {order: [(words, logp, backoff)]}) with logs in base 10, as in ARPA."""
    counts: list[int] = []
    grams: dict[int, list] = {}
    order = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("ngram "):
                counts.append(int(line.split("=")[1]))
            elif line.startswith("\\") and line.endswith("-grams:"):
                order = int(line[1:].split("-")[0])
                grams[order] = []
            elif line == END:
                break
            elif order and line:
                parts = line.split("\t")
                logp = float(parts[0])
                words = parts[1].split(" ")
                backoff = float(parts[2]) if len(parts) > 2 else 0.0
                grams[order].append((words, logp, backoff))
    return counts, grams


def export(arpa: Path, out: Path) -> dict:
    counts, grams = read_arpa(arpa)
    order = max(grams)
    if order > 3:
        raise ValueError(f"this writer handles up to trigrams, the ARPA is order {order}")
    log.info("arpa counts: %s", counts)

    vocab = sorted(w[0] for w, _, _ in grams[1])
    if len(vocab) >= MAX_VOCAB:
        raise ValueError(f"{len(vocab)} words exceeds the {MAX_VOCAB} the key packing allows")
    idx = {w: i for i, w in enumerate(vocab)}

    uni = [(0.0, 0.0)] * len(vocab)
    for words, logp, bo in grams[1]:
        uni[idx[words[0]]] = (logp, bo)

    def rows(n: int, with_backoff: bool):
        out_rows = []
        for words, logp, bo in grams.get(n, []):
            if any(w not in idx for w in words):
                continue
            key = pack_key([idx[w] for w in words])
            out_rows.append((key, logp, bo) if with_backoff else (key, logp))
        out_rows.sort(key=lambda r: r[0])
        return out_rows

    bi = rows(2, True)
    tri = rows(3, False)
    log.info("packed: %d unigrams, %d bigrams, %d trigrams", len(uni), len(bi), len(tri))

    offsets, pos = [0], 0
    for w in vocab:
        pos += len(w.encode("utf-8"))
        offsets.append(pos)
    blob = "".join(vocab).encode("utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", order, len(vocab)))
        f.write(struct.pack("<iii", len(uni), len(bi), len(tri)))
        f.write(struct.pack("<i", len(blob)))
        f.write(struct.pack(f"<{len(offsets)}i", *offsets))
        f.write(blob)
        for logp, bo in uni:
            f.write(struct.pack("<ff", logp, bo))
        for key, logp, bo in bi:
            f.write(struct.pack("<qff", key, logp, bo))
        for key, logp in tri:
            f.write(struct.pack("<qf", key, logp))

    report = {"path": str(out), "order": order, "vocab": len(vocab), "unigrams": len(uni),
              "bigrams": len(bi), "trigrams": len(tri), "size_mb": out.stat().st_size / 1e6}
    log.info("wrote %s", report)
    return report
