package com.openhackathon.voicetotext.decoding

import com.openhackathon.voicetotext.asr.Emissions
import java.util.TreeSet
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow

/**
 * Layer 2a: CTC prefix beam search with a word n-gram language model, the same algorithm and
 * constants as pyctcdecode on the desktop.
 *
 * The language score is applied once per completed word, at the word boundary, exactly as
 * pyctcdecode does; partial words carry no LM score until they are finished, but a partial
 * word that no vocabulary entry starts with is ranked down at once (dead end).
 *
 * Two additions over plain shallow fusion:
 *  - **context**: the words already written in the text field are the LM history of the
 *    first words of this utterance, so "θα πάω στο" is continued, not restarted;
 *  - **the speaker's own words**: names and places outside the 300k vocabulary are neither
 *    cut as dead ends nor scored as unknown. They get a personal probability mixed in
 *    linearly, as the desktop's interpolated personal LM does:
 *    P = (1 - lam) P_general + lam / |personal words|.
 */
class BeamSearch(
    private val labels: List<String>,
    private val lm: NgramLm?,
    private val alpha: Float = 0.7f,
    private val beta: Float = 4.0f,
    private val beamWidth: Int = 128,
    /**
     * Labels below this posterior are not even considered at a frame. e^-5, pyctcdecode's
     * token_min_logp, which is what the desktop numbers were measured with. The old 0.001
     * let 1.9 labels per frame through on Omnilingual (1.2 on wav2vec2) and the beam
     * search took 15 s for a 5 s clip.
     */
    private val pruneProb: Float = 0.0067f,
    /** Drop hypotheses this far (natural log) below the best one; pyctcdecode's beam_prune_logp. */
    private val beamPruneLogp: Float = -10f,
    /** The speaker's own vocabulary, normalized (lowercase Greek). */
    personalWords: Collection<String> = emptyList(),
    /** Interpolation weight of the personal words; the profile default on the desktop. */
    private val personalWeight: Float = 0.1f,
    /** Extra bonus for personal words, natural log; 0 = the interpolation alone. */
    private val hotwordWeight: Float = 0f,
) {
    private val blank = labels.indexOfFirst { it == "[PAD]" || it == "<pad>" }
    private val space = labels.indexOfFirst { it == "|" || it == " " }

    /** Cost of a partial word no vocabulary entry starts with; mirrors unk_score_offset. */
    private val deadEnd = -10f

    private val personal: TreeSet<String> = TreeSet(personalWords.filter { it.isNotBlank() })
    private val personalLog10: Float =
        if (personal.isEmpty()) 0f else log10(personalWeight.toDouble() / personal.size).toFloat()

    /** One decoded sentence with its two scores kept apart (natural log). */
    class Hyp(val words: List<String>, val acoustic: Float, val lm: Float) {
        val text: String get() = words.joinToString(" ")
        val total: Float get() = acoustic + lm
    }

    private class Beam(
        val words: List<String>,
        val partial: String,
        var pBlank: Float,
        var pNonBlank: Float,
        val lmScore: Float,
        /** words joined by spaces, carried along instead of re-joined on every lookup */
        val wordsKey: String,
    ) {
        val key: String = wordsKey + "\u0000" + partial
        fun total(): Float = logAdd(pBlank, pNonBlank) + lmScore
    }

    /**
     * Returns the best hypotheses, best first. [history]: normalized words already written
     * before this utterance; they are LM context only and never part of the output.
     */
    fun decode(em: Emissions, nBest: Int = 1, history: List<String> = emptyList()): List<Hyp> {
        var beams = listOf(Beam(emptyList(), "", 0f, NEG_INF, 0f, ""))
        val v = em.vocabSize

        for (t in 0 until em.numFrames) {
            val next = HashMap<String, Beam>(beamWidth * 4)
            // consider only labels with meaningful mass at this frame
            val candidates = ArrayList<Int>(8)
            for (c in 0 until v) if (exp(em.logProb(t, c)) > pruneProb) candidates.add(c)
            if (candidates.isEmpty()) candidates.add(em.argmaxAt(t))

            for (b in beams) {
                for (c in candidates) {
                    val p = em.logProb(t, c)
                    when {
                        // A blank closes a prefix, so it inherits BOTH the blank and the
                        // non-blank mass. Taking only pBlank here strands every path that
                        // ended on a character, and the only way such a path stays alive is
                        // to repeat its last character: the output comes out doubled.
                        c == blank -> merge(next, Beam(b.words, b.partial,
                            logAdd(b.pBlank, b.pNonBlank) + p, NEG_INF, b.lmScore, b.wordsKey))
                        c == space -> {
                            if (b.partial.isEmpty()) {
                                merge(next, Beam(b.words, "", logAdd(b.pBlank, b.pNonBlank) + p, NEG_INF, b.lmScore, b.wordsKey))
                            } else {
                                val words = b.words + b.partial
                                val wk = if (b.wordsKey.isEmpty()) b.partial else b.wordsKey + " " + b.partial
                                merge(next, Beam(words, "", logAdd(b.pBlank, b.pNonBlank) + p, NEG_INF,
                                    b.lmScore + wordScore(history + b.words, b.partial), wk))
                            }
                        }
                        else -> {
                            val ch = labels[c]
                            if (ch.length != 1) continue          // skip [UNK], <s>, unused
                            val same = b.partial.isNotEmpty() && b.partial.last().toString() == ch
                            if (same) {
                                // repeat of the same character: extends only through a blank
                                merge(next, Beam(b.words, b.partial, NEG_INF, b.pNonBlank + p, b.lmScore, b.wordsKey))
                                merge(next, Beam(b.words, b.partial + ch, NEG_INF, b.pBlank + p, b.lmScore, b.wordsKey))
                            } else {
                                merge(next, Beam(b.words, b.partial + ch, NEG_INF,
                                    logAdd(b.pBlank, b.pNonBlank) + p, b.lmScore, b.wordsKey))
                            }
                        }
                    }
                }
            }
            // Rank with the dead-end penalty so a prefix that can never become a word is
            // dropped now rather than at the next space.
            val ranked = next.values.map { it to it.total() + prefixPenalty(it.partial) }
                .sortedByDescending { it.second }
            val best = ranked.firstOrNull()?.second ?: NEG_INF
            beams = ranked.asSequence().take(beamWidth)
                .filter { it.second >= best + beamPruneLogp }
                .map { it.first }.toList()
        }

        return beams
            .map { b ->
                val words = if (b.partial.isEmpty()) b.words else b.words + b.partial
                val extra = if (b.partial.isEmpty()) 0f else wordScore(history + b.words, b.partial)
                Hyp(words, logAdd(b.pBlank, b.pNonBlank), b.lmScore + extra)
            }
            .sortedByDescending { it.total }
            .distinctBy { it.text }
            .take(nBest)
    }

    private fun prefixPenalty(partial: String): Float {
        val n = lm ?: return 0f
        if (partial.isEmpty()) return 0f
        if (n.hasPrefix(partial)) return 0f
        val p = personal.ceiling(partial)
        return if (p != null && p.startsWith(partial)) 0f else deadEnd
    }

    fun isPersonal(word: String): Boolean = word in personal

    /** Does the general or the personal vocabulary contain this word? */
    fun knows(word: String): Boolean = word in personal || (lm?.contains(word) ?: false)

    /**
     * alpha * log10 P(word | last two words of context) * ln 10 + beta, in natural log like
     * the acoustic scores. Public for the spelling pass, which must score on the same scale.
     */
    fun wordScore(context: List<String>, word: String): Float {
        var s = beta
        val n = lm
        if (n != null) {
            var g = n.score(context.takeLast(2), word)
            if (word in personal) {
                // (1 - lam) * 10^g + lam / |P|, in log10
                g = log10((1.0 - personalWeight) * 10.0.pow(g.toDouble()) + 10.0.pow(personalLog10.toDouble())).toFloat()
                s += hotwordWeight
            }
            s += alpha * LOG10 * g
        }
        return s
    }

    private fun merge(map: HashMap<String, Beam>, b: Beam) {
        val old = map[b.key]
        if (old == null) {
            map[b.key] = b
        } else {
            old.pBlank = logAdd(old.pBlank, b.pBlank)
            old.pNonBlank = logAdd(old.pNonBlank, b.pNonBlank)
        }
    }

    companion object {
        private const val NEG_INF = -1e30f
        /** the LM stores base-10 logs; the acoustic scores are natural logs */
        private const val LOG10 = 2.302585f

        fun logAdd(a: Float, b: Float): Float {
            if (a <= NEG_INF) return b
            if (b <= NEG_INF) return a
            val m = max(a, b)
            return m + ln(exp(a - m) + exp(b - m))
        }
    }
}
