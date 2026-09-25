"""The phone LM format: header, sorted vocabulary, packed keys, values straight from the ARPA."""

import struct

import pytest

from voicetotext.decoding.export_lm import ID_BITS, MAGIC, export, pack_key

ARPA = "\n".join([
    "",
    "\\data\\",
    "ngram 1=5",
    "ngram 2=3",
    "ngram 3=2",
    "",
    "\\1-grams:",
    "-1.0\t<unk>\t0",
    "-99\t<s>\t-0.5",
    "-1.2\t</s>\t0",
    "-0.8\tκαλή\t-0.3",
    "-0.9\tμέρα\t-0.2",
    "",
    "\\2-grams:",
    "-0.4\t<s> καλή\t-0.1",
    "-0.2\tκαλή μέρα\t-0.05",
    "-0.6\tμέρα </s>\t0",
    "",
    "\\3-grams:",
    "-0.1\t<s> καλή μέρα",
    "-0.3\tκαλή μέρα </s>",
    "",
    "\\end\\",
    "",
])


def test_export_layout(tmp_path):
    arpa = tmp_path / "t.arpa"
    arpa.write_text(ARPA, encoding="utf-8")
    report = export(arpa, tmp_path / "t.lm")
    assert (report["order"], report["vocab"], report["bigrams"], report["trigrams"]) == (3, 5, 3, 2)

    b = (tmp_path / "t.lm").read_bytes()
    assert b[:8] == MAGIC
    order, n, n1, n2, n3, blob_len = struct.unpack_from("<6i", b, 8)
    assert (order, n, n1, n2, n3) == (3, 5, 5, 3, 2)
    pos = 8 + 24
    offsets = struct.unpack_from(f"<{n + 1}i", b, pos)
    pos += 4 * (n + 1)
    raw = b[pos:pos + blob_len]
    pos += blob_len
    blob = raw.decode("utf-8")
    vocab = [raw[offsets[i]:offsets[i + 1]].decode("utf-8") for i in range(n)]
    assert vocab == sorted(["<unk>", "<s>", "</s>", "καλή", "μέρα"])
    assert blob == "".join(vocab)
    idx = {w: i for i, w in enumerate(vocab)}

    uni = [struct.unpack_from("<ff", b, pos + 8 * i) for i in range(n)]
    pos += 8 * n
    assert uni[idx["καλή"]] == pytest.approx((-0.8, -0.3))

    bi = [struct.unpack_from("<qff", b, pos + 16 * i) for i in range(n2)]
    pos += 16 * n2
    assert [k for k, _, _ in bi] == sorted(k for k, _, _ in bi)
    key = pack_key([idx["καλή"], idx["μέρα"]])
    assert key == (idx["καλή"] << ID_BITS) | idx["μέρα"]
    assert dict((k, (p, bo)) for k, p, bo in bi)[key] == pytest.approx((-0.2, -0.05))

    tri = [struct.unpack_from("<qf", b, pos + 12 * i) for i in range(n3)]
    pos += 12 * n3
    assert pos == len(b)
    tri_key = pack_key([idx["<s>"], idx["καλή"], idx["μέρα"]])
    assert dict(tri)[tri_key] == pytest.approx(-0.1)


def test_rejects_order_above_three(tmp_path):
    arpa = tmp_path / "four.arpa"
    arpa.write_text("\\data\\\nngram 1=1\n\n\\1-grams:\n-1\ta\n\n\\4-grams:\n-1\ta a a a\n\n\\end\\\n", encoding="utf-8")
    with pytest.raises(ValueError):
        export(arpa, tmp_path / "four.lm")
