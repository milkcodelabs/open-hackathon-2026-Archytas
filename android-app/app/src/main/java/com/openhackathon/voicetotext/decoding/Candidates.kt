package com.openhackathon.voicetotext.decoding

import kotlin.math.exp

/**
 * One alternative sentence for the user to pick from.
 *
 * [probability] is relative to the other hypotheses the decoder kept: a softmax of their
 * combined scores (acoustic + language model, natural log). It answers "how much more likely
 * is this sentence than the others", not "how likely is it to be right".
 * [changed] holds the positions of the words that differ from the best sentence, so the
 * screen can highlight only what is actually different.
 */
class Candidate(
    val words: List<String>,
    val probability: Float,
    val changed: Set<Int>,
) {
    val text: String get() = words.joinToString(" ")
}

object Candidates {

    /** The [max] best distinct sentences, best first, with probabilities and differences. */
    fun from(hyps: List<BeamSearch.Hyp>, max: Int = 10): List<Candidate> {
        val distinct = hyps.filter { it.words.isNotEmpty() }.distinctBy { it.text }
        if (distinct.isEmpty()) return emptyList()
        // softmax over ALL kept hypotheses, so the shares shown are honest about the tail
        val best = distinct.maxOf { it.total }
        val weights = distinct.map { exp((it.total - best).toDouble()) }
        val z = weights.sum()
        val top = distinct.first().words
        return distinct.zip(weights)
            .sortedByDescending { it.second }
            .take(max)
            .map { (h, w) -> Candidate(h.words, (w / z).toFloat(), differing(h.words, top)) }
    }

    /** Candidate list for a plain sentence (no language model): one entry, certain. */
    fun single(text: String): List<Candidate> {
        val words = text.split(' ').filter { it.isNotBlank() }
        return if (words.isEmpty()) emptyList() else listOf(Candidate(words, 1f, emptySet()))
    }

    /** Positions in [words] that are not part of a longest common word subsequence with [ref]. */
    fun differing(words: List<String>, ref: List<String>): Set<Int> {
        val n = words.size
        val m = ref.size
        val lcs = Array(n + 1) { IntArray(m + 1) }
        for (i in n - 1 downTo 0) for (j in m - 1 downTo 0) {
            lcs[i][j] = if (words[i] == ref[j]) lcs[i + 1][j + 1] + 1 else maxOf(lcs[i + 1][j], lcs[i][j + 1])
        }
        val same = HashSet<Int>()
        var i = 0
        var j = 0
        while (i < n && j < m) {
            when {
                words[i] == ref[j] -> { same.add(i); i++; j++ }
                lcs[i + 1][j] >= lcs[i][j + 1] -> i++
                else -> j++
            }
        }
        return (0 until n).filter { it !in same }.toSet()
    }
}
