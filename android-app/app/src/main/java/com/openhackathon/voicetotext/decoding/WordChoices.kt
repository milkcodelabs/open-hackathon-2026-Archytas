package com.openhackathon.voicetotext.decoding

import kotlin.math.pow

/**
 * The places in a sentence where the recognizer was unsure, each with what it heard there
 * instead, so the person can pick the right words after the sentence was typed.
 *
 * A word's confidence is 1 minus the weight of the candidate sentences that put something
 * else in its place. Words below the threshold become a [Spot]; a spot is widened until every
 * alternative's differing words fit inside it, so replacing the spot with an option always
 * gives exactly that candidate's words there.
 *
 * The weights are the candidates' probabilities flattened by [TEMPERATURE]: a softmax over
 * whole-sentence scores is far too sure of itself (the same reason lattice word posteriors
 * are computed with a scaled-down acoustic score), and at face value almost no word ever
 * looked doubtful.
 *
 * The sentence compared against does not have to be one of the candidates: when a remote
 * model rewrote it, pass the recognizer's own best sentence as [find]'s `forced`, and every
 * word the two disagree on is offered whatever the numbers say.
 */
object WordChoices {

    /** Below this a word is offered for correction. */
    const val THRESHOLD = 0.8f

    /** Sentence scores are divided by this before the softmax (p^(1/T), renormalized). */
    const val TEMPERATURE = 2f

    class Option(val words: List<String>, val probability: Float) {
        val text: String get() = words.joinToString(" ")
    }

    /**
     * Words [start, end) of the sentence. [options] start with what the sentence holds there
     * now, then the forced sentence's words if it differs here, then the rest best first.
     */
    class Spot(val start: Int, val end: Int, val options: List<Option>) {
        val confidence: Float get() = options.first().probability
    }

    /** One run of differences between a candidate and the sentence, in sentence positions. */
    private class Block(val start: Int, val end: Int, val words: List<String>)

    private class Other(val weight: Float, val forced: Boolean, val blocks: List<Block>)

    fun find(
        sentence: List<String>,
        candidates: List<Candidate>,
        threshold: Float = THRESHOLD,
        /** A sentence whose differences from [sentence] are always offered. */
        forced: List<String>? = null,
        temperature: Float = TEMPERATURE,
        maxOptions: Int = 4,
        maxSpan: Int = 3,
    ): List<Spot> {
        if (sentence.isEmpty()) return emptyList()
        val weights = temper(candidates.map { it.probability }, temperature)
        val others = candidates.indices
            .filter { candidates[it].words != sentence && (weights[it] > 0f || candidates[it].words == forced) }
            .map { val w = candidates[it].words; Other(weights[it], w == forced, blocks(w, sentence)) }
            .toMutableList()
        if (forced != null && forced != sentence && others.none { it.forced }) {
            others.add(Other(0f, true, blocks(forced, sentence)))
        }
        if (others.isEmpty()) return emptyList()
        // the recognizer's own sentence may differ from a rewrite over a longer stretch
        fun limit(o: Other) = if (o.forced) maxSpan * 2 else maxSpan

        // a candidate that rewrites a long stretch is a different sentence, not a word choice
        val dispute = FloatArray(sentence.size)
        val mustAsk = BooleanArray(sentence.size)
        for (o in others) for (b in o.blocks) {
            if (b.end - b.start > limit(o)) continue
            for (k in b.start until b.end) {
                dispute[k] += o.weight
                if (o.forced) mustAsk[k] = true
            }
        }

        val spots = ArrayList<Spot>()
        var k = 0
        while (k < sentence.size) {
            if (!mustAsk[k] && 1f - dispute[k] >= threshold) { k++; continue }
            var a = k
            var b = k + 1
            var grew = true
            while (grew) {
                grew = false
                for (o in others) for (bl in o.blocks) {
                    // a faint alternative is dropped rather than allowed to widen the spot
                    if (bl.end == bl.start || (!o.forced && o.weight < MIN_WIDEN)) continue
                    val crosses = bl.start < b && bl.end > a && (bl.start < a || bl.end > b)
                    if (!crosses) continue
                    val na = minOf(a, bl.start)
                    val nb = maxOf(b, bl.end)
                    if (nb - na > limit(o)) continue
                    a = na; b = nb; grew = true
                }
            }
            val here = sentence.subList(a, b)
            val alts = LinkedHashMap<List<String>, Float>()
            var forcedSeg: List<String>? = null
            for (o in others) {
                val seg = segment(sentence, o.blocks, a, b) ?: continue
                if (seg == here) continue
                alts[seg] = (alts[seg] ?: 0f) + o.weight
                if (o.forced) forcedSeg = seg
            }
            val ranked = alts.entries.sortedByDescending { it.value }
            val kept = (listOfNotNull(forcedSeg) +
                ranked.filter { it.value >= MIN_OPTION && it.key != forcedSeg }.map { it.key })
                .take(maxOptions - 1)
            val current = (1f - alts.values.sum()).coerceIn(0f, 1f)
            if (kept.isNotEmpty() && (forcedSeg != null || current < threshold)) {
                spots.add(Spot(a, b, listOf(Option(here, current)) + kept.map { Option(it, alts[it] ?: 0f) }))
            }
            k = b
        }
        return spots
    }

    /** p^(1/T) renormalized: the same as dividing every sentence score by T before the softmax. */
    private fun temper(p: List<Float>, t: Float): List<Float> {
        if (t == 1f) return p
        val w = p.map { if (it > 0f) it.toDouble().pow(1.0 / t) else 0.0 }
        val z = w.sum()
        return if (z > 0) w.map { (it / z).toFloat() } else p
    }

    /** Alternatives this unlikely are noise, not something worth a button. */
    private const val MIN_OPTION = 0.01f
    /** Weight an alternative needs before its differences may make a spot longer. */
    private const val MIN_WIDEN = 0.1f

    /**
     * What the candidate with difference [bs] has in place of sentence words [a, b), or null
     * when one of its differences reaches outside that range.
     */
    private fun segment(sentence: List<String>, bs: List<Block>, a: Int, b: Int): List<String>? {
        if (bs.any { it.start < a && it.end > a }) return null
        val out = ArrayList<String>()
        var pos = a
        while (pos < b) {
            val bl = bs.firstOrNull { it.start == pos }
            when {
                bl == null -> { out.add(sentence[pos]); pos++ }
                bl.end == pos -> {
                    // words the candidate has before this one; at the left edge they belong outside
                    if (pos > a) out.addAll(bl.words)
                    out.add(sentence[pos]); pos++
                }
                bl.end > b -> return null
                else -> { out.addAll(bl.words); pos = bl.end }
            }
        }
        return out
    }

    /** Longest-common-subsequence diff of [words] against [ref], as runs in [ref] positions. */
    private fun blocks(words: List<String>, ref: List<String>): List<Block> {
        val n = words.size
        val m = ref.size
        val lcs = Array(n + 1) { IntArray(m + 1) }
        for (i in n - 1 downTo 0) for (j in m - 1 downTo 0) {
            lcs[i][j] = if (words[i] == ref[j]) lcs[i + 1][j + 1] + 1 else maxOf(lcs[i + 1][j], lcs[i][j + 1])
        }
        val out = ArrayList<Block>()
        var i = 0
        var j = 0
        var bi = 0
        var bj = 0
        while (i < n && j < m) {
            when {
                words[i] == ref[j] -> {
                    if (i > bi || j > bj) out.add(Block(bj, j, words.subList(bi, i)))
                    i++; j++; bi = i; bj = j
                }
                lcs[i + 1][j] >= lcs[i][j + 1] -> i++
                else -> j++
            }
        }
        if (n > bi || m > bj) out.add(Block(bj, m, words.subList(bi, n)))
        return out
    }
}
