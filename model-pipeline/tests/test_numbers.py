import pytest

from voicetotext.phonetics.numbers import cardinal, verbalize_digits


@pytest.mark.parametrize("n,words", [
    (0, "μηδέν"), (1, "ένα"), (13, "δεκατρία"), (21, "είκοσι ένα"), (100, "εκατό"), (101, "εκατόν ένα"),
    (200, "διακόσια"), (1000, "χίλια"), (1980, "χίλια εννιακόσια ογδόντα"), (2000, "δύο χιλιάδες"),
    (3004, "τρεις χιλιάδες τέσσερα"), (201000, "διακόσιες μία χιλιάδες"), (1000000, "ένα εκατομμύριο"),
    (2500000, "δύο εκατομμύρια πεντακόσιες χιλιάδες"),
])
def test_cardinal(n, words):
    assert cardinal(n) == words


def test_verbalize_in_text():
    assert verbalize_digits("το 1980 ήταν") == "το χίλια εννιακόσια ογδόντα ήταν"
    assert verbalize_digits("1.500 ευρώ") == "χίλια πεντακόσια ευρώ"
    assert verbalize_digits("3,5%") == "τρία κόμμα πέντε τοις εκατό"
    assert verbalize_digits("καμία") == "καμία"
    assert verbalize_digits("6971234567890123") == "έξι εννέα επτά ένα δύο τρία τέσσερα πέντε έξι επτά οκτώ εννέα μηδέν ένα δύο τρία"
    assert cardinal(999_999_999_999).startswith("εννιακόσια ενενήντα εννέα δισεκατομμύρια")
