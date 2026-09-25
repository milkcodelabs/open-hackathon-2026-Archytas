package com.openhackathon.voicetotext.decoding


/**
 * Layer 2b: the language model, not the acoustic model, chooses Greek spellings.
 * Mirrors `greek_vt/decoding/spelling.py`.
 *
 * The acoustic model cannot tell ι η υ ει οι apart (or ο/ω, ε/αι, single/double consonants):
 * they sound the same. For every word of every N-best hypothesis this offers the real
 * spellings of the same sound ([HomophoneIndex], [SoundKey]); a small dynamic programme over
 * each hypothesis picks the sequence the LM likes best in context, and the list is re-ranked
 * by acoustic score + the recomputed LM score.
 *
 *  - only same-sound spellings are offered, never other words;
 *  - replacing a word that is already known costs [switchCost] (natural log), so the LM must
 *    clearly prefer the other spelling; an unknown word is replaced for free;
 *  - the acoustic score is kept: the variants sound the same by construction.
 */
class SpellingRescorer(
    private val index: HomophoneIndex,
    private val beam: BeamSearch,
    private val personalWords: Collection<String> = emptyList(),
    private val switchCost: Float = 1.5f,
    private val maxVariants: Int = 6,
    private val stateBeam: Int = 16,
) {
    /** the speaker's own words by sound key, offered before the vocabulary's spellings */
    private val personal: Map<String, List<String>> = personalWords.groupBy { SoundKey.of(it) }

    fun variants(word: String): List<String> {
        val key = SoundKey.of(word)
        val mine = personal[key].orEmpty().filter { it != word }
        val rest = index.group(key).filter { it != word && it !in mine }
        return (listOf(word) + mine + rest).take(maxVariants)
    }

    private class Path(val score: Float, val words: List<String>)

    /** Best spelling of one hypothesis; returns the words and their LM score. */
    private fun respell(words: List<String>, history: List<String>): Path {
        var paths = listOf(Path(0f, emptyList()))
        for (w in words) {
            val known = beam.knows(w)
            val alts = variants(w)
            val best = HashMap<String, Path>()
            for (p in paths) {
                val ctx = history + p.words
                for (a in alts) {
                    var s = beam.wordScore(ctx, a)
                    if (a != w && known) s -= switchCost
                    val seq = p.words + a
                    val key = seq.takeLast(2).joinToString(" ")    // same LM history -> merge
                    val cand = Path(p.score + s, seq)
                    val old = best[key]
                    if (old == null || cand.score > old.score) best[key] = cand
                }
            }
            paths = best.values.sortedByDescending { it.score }.take(stateBeam)
        }
        return paths.first()
    }

    fun rescore(hyps: List<BeamSearch.Hyp>, history: List<String> = emptyList()): List<BeamSearch.Hyp> {
        if (hyps.isEmpty()) return hyps
        val out = LinkedHashMap<String, BeamSearch.Hyp>()
        for (h in hyps) {
            if (h.words.isEmpty()) continue
            val p = respell(h.words, history)
            val n = BeamSearch.Hyp(p.words, h.acoustic, p.score)
            val old = out[n.text]
            if (old == null || n.total > old.total) out[n.text] = n
        }
        return out.values.sortedByDescending { it.total }.ifEmpty { hyps }
    }
}
