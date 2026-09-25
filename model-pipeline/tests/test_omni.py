import numpy as np

from voicetotext.acoustic.omni import greek_subset, read_tokens, transfer_foreign
from voicetotext.decoding.vocab import blank_index, to_pyctcdecode_labels


def test_tokens_and_greek_subset(tmp_path):
    f = tmp_path / "tokens.txt"
    # the space token is written as a line holding only its id
    f.write_text("<s> 0\n<pad> 1\n</s> 2\n<unk> 3\n4\nα 5\nZ 6\nά 7\nἀ 8\n' 9\nς 10\nΑ 11\nж 12\n", encoding="utf-8")
    id2tok = read_tokens(f)
    assert id2tok[4] == " " and id2tok[8] == "ἀ"
    cols, labels = greek_subset(id2tok)
    assert labels == ["<pad>", "|", "α", "ά", "'", "ς"]      # no Latin, Cyrillic, polytonic or capitals
    assert cols == [0, 4, 5, 7, 9, 10]                      # blank is token 0, not <pad>
    assert blank_index(labels) == 0
    assert to_pyctcdecode_labels(labels)[:2] == ["", " "]


def test_foreign_mass_goes_to_letters_not_blank():
    # columns: blank, space, alpha, omicron. 0.5 of the frame's mass sat on a Latin "a".
    p = np.array([[0.2, 0.1, 0.15, 0.05]])
    out = np.exp(transfer_foreign(np.log(p), np.array([False, False, True, True])))
    assert np.allclose(out.sum(), 1.0)
    assert np.allclose(out[0, :2], [0.2, 0.1])                 # blank and space untouched
    assert np.allclose(out[0, 2:], [0.15 * 3.5, 0.05 * 3.5])   # 1 + 0.5 / 0.2
    assert out[0].argmax() == 2                                # the letter now beats blank
