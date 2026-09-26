package com.openhackathon.voicetotext.asr

import kotlin.math.exp
import kotlin.math.ln

/**
 * The (T, V) log-probability matrix produced by the CTC acoustic layer.
 *
 * This is the unit of work. Argmax strings are *views*
 * computed on demand by [greedyDecode]; they are never the stored representation, because
 * everything above the acoustic layer (beam search, speaker adaptation, correction) needs
 * the full distribution, not the winning character.
 *
 * @param logProbs row-major, length T * V.
 * @param labels index-aligned to V.
 */
class Emissions(
    val logProbs: FloatArray,
    val labels: List<String>,
    val frameDurationMs: Float = 20f,
    val audioId: String = "",
    val modelId: String = "",
) {
    val vocabSize: Int = labels.size
    val numFrames: Int = logProbs.size / vocabSize

    init {
        require(logProbs.size % vocabSize == 0) {
            "logProbs length ${logProbs.size} is not a multiple of V=$vocabSize"
        }
    }

    val durationSeconds: Float get() = numFrames * frameDurationMs / 1000f

    fun logProb(frame: Int, label: Int): Float = logProbs[frame * vocabSize + label]

    fun argmaxAt(frame: Int): Int {
        val base = frame * vocabSize
        var best = 0
        var bestVal = logProbs[base]
        for (v in 1 until vocabSize) {
            val x = logProbs[base + v]
            if (x > bestVal) { bestVal = x; best = v }
        }
        return best
    }

    fun maxProbAt(frame: Int): Float = exp(logProbs[frame * vocabSize + argmaxAt(frame)])

    val blankIndex: Int by lazy {
        labels.indexOfFirst { it == "[PAD]" || it == "<pad>" }.also {
            require(it >= 0) { "no blank token in labels" }
        }
    }

    private fun isSpecial(label: String): Boolean =
        label == "[PAD]" || label == "<pad>" || label == "[UNK]" || label == "<unk>" ||
            label == "<s>" || label == "</s>" || label.startsWith("<unused")

    /** One collapsed CTC token on the argmax path. */
    data class TokenSpan(val label: String, val startFrame: Int, val endFrame: Int, val prob: Float)

    /**
     * Argmax per frame, then the CTC collapse rule: merge runs of the same label, drop blanks.
     * This is a deterministic decoding rule, not a correction: it has no vocabulary and no
     * language model, so it will happily emit non-words.
     */
    fun greedyPath(): List<TokenSpan> {
        val out = ArrayList<TokenSpan>()
        var t = 0
        while (t < numFrames) {
            val id = argmaxAt(t)
            val start = t
            var sum = 0f
            while (t < numFrames && argmaxAt(t) == id) { sum += maxProbAt(t); t++ }
            if (id != blankIndex) out.add(TokenSpan(labels[id], start, t, sum / (t - start)))
        }
        return out
    }

    data class WordSpan(val text: String, val startFrame: Int, val endFrame: Int, val confidence: Float)

    fun greedyWords(): List<WordSpan> {
        val words = ArrayList<WordSpan>()
        val chars = ArrayList<TokenSpan>()
        fun flush() {
            if (chars.isNotEmpty()) {
                words.add(
                    WordSpan(
                        chars.joinToString("") { it.label },
                        chars.first().startFrame,
                        chars.last().endFrame,
                        chars.map { it.prob }.average().toFloat(),
                    )
                )
                chars.clear()
            }
        }
        for (tok in greedyPath()) {
            if (tok.label == "|" || tok.label == " ") flush() else if (!isSpecial(tok.label)) chars.add(tok)
        }
        flush()
        return words
    }

    fun greedyDecode(): String = greedyWords().joinToString(" ") { it.text }

    /** Fraction of frames whose argmax is the blank token. High values mean sparse speech. */
    fun blankFraction(): Float {
        var n = 0
        for (t in 0 until numFrames) if (argmaxAt(t) == blankIndex) n++
        return n.toFloat() / numFrames
    }

    companion object {
        /** Row-wise log-softmax of raw logits, in place. */
        fun fromLogits(
            logits: FloatArray,
            labels: List<String>,
            frameDurationMs: Float = 20f,
            audioId: String = "",
            modelId: String = "",
        ): Emissions {
            val v = labels.size
            val frames = logits.size / v
            for (t in 0 until frames) {
                val base = t * v
                var max = logits[base]
                for (i in 1 until v) if (logits[base + i] > max) max = logits[base + i]
                var sum = 0f
                for (i in 0 until v) sum += exp(logits[base + i] - max)
                val logZ = max + ln(sum)
                for (i in 0 until v) logits[base + i] -= logZ
            }
            return Emissions(logits, labels, frameDurationMs, audioId, modelId)
        }
    }
}
