package com.openhackathon.voicetotext.decoding

/**
 * The places in a sentence where the recognizer was unsure, each with what it heard there
 * instead, so the person can pick the right words after the sentence was typed.
 *
 * A word's confidence is 1 minus the probability of the candidate sentences that put
 * something else in its place. Words below the threshold become a [Spot]; a spot is widened
 * until every alternative's differing words fit inside it, so replacing the spot with an
 * option always gives exactly that candidate's words there.
 *
 * The sentence compared against does not have to be one of the candidates: when a remote
 * model rewrote it, every word it changed is disputed by the recognizer's own best sentence.
 */
object WordChoices {

    /** Below this a word is offered for correction. */
    const val THRESHOLD = 0.8f

    class Option(val words: List<String>, val probability: Float) {
        val text: String get() = words.joinToString(" ")
    }

    /**
     * Words [start, end) of the sentence. [options] come best first after the first one,
     * which is always what the sentence holds there now.
     */
    class Spot(val start: Int, val end: Int, val options: List<Option>) {
        val confidence: Float get() = options.first().probability
    }

    /** One run of differences between a candidate and the sentence, in sentence positions. */
    private class Block(val start: Int, val end: Int, val words: List<String>)

    fun find(
        sentence: List<String>,
        candidates: List<Candidate>,
        threshold: Float = THRESHOLD,
        maxOptions: Int = 4,
        maxSpan: Int = 3,
    ): List<Spot> {
        if (sentence.isEmpty()) return emptyList()
        val others = candidates.filter { it.words != sentence && it.probability > 0f }
            .map { it to blocks(it.words, sentence) }
        if (others.isEmpty()) return emptyList()

        // a candidate that rewrites a long stretch is a different sentence, not a word choice
        val dispute = FloatArray(sentence.size)
        for ((c, bs) in others) for (b in bs) {
            if (b.end - b.start > maxSpan) continue
            for (k in b.start until b.end) dispute[k] += c.probability
        }

        val spots = ArrayList<Spot>()
        var k = 0
        while (k < sentence.size) {
            if (1f - dispute[k] >= threshold) { k++; continue }
            var a = k
            var b = k + 1
            var grew = true
            while (grew) {
                grew = false
                for ((_, bs) in others) for (bl in bs) {
                    if (bl.end == bl.start) continue
                    val crosses = bl.start < b && bl.end > a && (bl.start < a || bl.end > b)
                    if (!crosses) continue
                    val na = minOf(a, bl.start)
                    val nb = maxOf(b, bl.end)
                    if (nb - na > maxSpan) continue
                    a = na; b = nb; grew = true
                }
            }
            val here = sentence.subList(a, b)
            val alts = LinkedHashMap<List<String>, Float>()
            for ((c, bs) in others) {
                val seg = segment(sentence, bs, a, b) ?: continue
                if (seg != here) alts[seg] = (alts[seg] ?: 0f) + c.probability
            }
            val kept = alts.entries.filter { it.value >= MIN_OPTION }
                .sortedByDescending { it.value }.take(maxOptions - 1)
            val current = (1f - alts.values.sum()).coerceIn(0f, 1f)
            if (kept.isNotEmpty() && current < threshold) {
                spots.add(Spot(a, b, listOf(Option(here, current)) + kept.map { Option(it.key, it.value) }))
            }
            k = b
        }
        return spots
    }

    /** Alternatives this unlikely are noise, not something worth a button. */
    private const val MIN_OPTION = 0.01f

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
