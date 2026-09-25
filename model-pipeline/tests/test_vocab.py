from voicetotext.decoding.vocab import (
    blank_index, delimiter_index, greek_letter_set, inspect_labels, is_special, to_pyctcdecode_labels,
)


def test_phone_label_report(labels):
    rep = inspect_labels(labels)
    assert rep.size == 39
    assert rep.blank == "<pad>" and blank_index(labels) == 0
    assert rep.delimiter == "|" and delimiter_index(labels) == 1
    assert set("άέήίόύώ") <= set(rep.accented)
    assert set("ϊϋΐΰ") <= set(rep.dialytika)
    assert rep.uppercase == [] and rep.digits == [] and rep.latin == []
    assert rep.punctuation == ["'"]
    assert rep.has_final_sigma
    letters = greek_letter_set(labels)
    assert len(letters) == 36 and "'" not in letters
    assert is_special("<pad>") and is_special("<unk>") and not is_special("α")


def test_pyctcdecode_labels_alignment(labels):
    out = to_pyctcdecode_labels(labels)
    assert len(out) == len(labels)
    assert out[labels.index("<pad>")] == "" and out[labels.index("|")] == " "
    assert out.count("") == 1 and out[labels.index("ά")] == "ά"
