from voicetotext.decoding.gpt2_rescore import ACOUSTIC_GUARD, GAMMA, Hyp, rerank


def test_rerank_prefers_the_fluent_sentence_within_the_guard():
    hyps = [Hyp("α", acoustic=-10, ngram=-4), Hyp("β", acoustic=-12, ngram=-1)]
    # GPT-2 strongly prefers β, and β is within the acoustic guard, so it wins
    out = rerank(hyps, neural=[-20, -1])
    assert [h.text for h in out] == ["β", "α"]
    assert GAMMA == 0.5


def test_acoustic_guard_blocks_a_fluent_but_unheard_sentence():
    hyps = [Hyp("α", acoustic=-10, ngram=-4), Hyp("β", acoustic=-10 - ACOUSTIC_GUARD - 1, ngram=0)]
    out = rerank(hyps, neural=[0, 100])          # β is what GPT-2 wants, but it was not heard
    assert [h.text for h in out] == ["α", "β"]
