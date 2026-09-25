from voicetotext.phonetics.homophones import group, load_binary, save_binary, sound_key, unigrams_from_arpa


def test_sound_key_collapses_the_documented_pair():
    assert sound_key("δήμου") == sound_key("δίμου") == "δiμu"   # μ is not rewritten; only the vowels are
    assert sound_key("είσαι") == sound_key("ίσε")
    # a diaeresis splits the digraph: οϊ is o + i, not οι
    assert sound_key("οϊ") == "oi"
    assert sound_key("οι") == "i"
    # γγ is /ng/ and stays; other doubles collapse
    assert "γγ" in sound_key("άγγελος")
    assert "λλ" not in sound_key("άλλο")


def test_binary_roundtrip_orders_by_frequency(tmp_path):
    words = ["δίμου", "δήμου", "καλή", "<unk>"]
    groups = group(words, {"δήμου": -1.0, "δίμου": -2.0, "καλή": -0.5})
    assert "<unk>" not in sum(groups.values(), [])
    path = save_binary(groups, tmp_path / "h.bin")
    back = load_binary(path)
    assert back[sound_key("δήμου")] == ["δήμου", "δίμου"]   # more frequent first
    assert list(back) == sorted(back)                       # keys sorted for the binary search


def test_unigrams_from_arpa(tmp_path):
    arpa = tmp_path / "t.arpa"
    arpa.write_text("\\1-grams:\n-0.8\tδήμου\t-0.1\n-1.5\tδίμου\t0\n\n\\2-grams:\n-0.2\tα β\n", encoding="utf-8")
    assert unigrams_from_arpa(arpa) == {"δήμου": -0.8, "δίμου": -1.5}
