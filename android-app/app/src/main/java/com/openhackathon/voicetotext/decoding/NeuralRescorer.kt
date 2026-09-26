package com.openhackathon.voicetotext.decoding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.LongBuffer

/**
 * Layer 2c: a small Greek GPT-2 (lighteternal/gpt2-finetuned-greek, 124M, int8) re-ranks the
 * best sentences using the whole sentence and the text already written in the field.
 *
 * It never writes text; it only re-orders sentences the acoustic model and the beam search
 * produced, and its score is added to theirs:
 *
 *     total = acoustic + ngram + [gamma] * log P_gpt2(sentence | context)
 *
 * A sentence whose acoustic score is more than [acousticGuard] nats below the best one can
 * never win, so fluency cannot override what was heard.
 *
 * Measured with this int8 graph (weights chosen on FLEURS dev + half of Common Voice):
 * WER 0.129 -> 0.114 on FLEURS test and 0.132 -> 0.123 on the CV half it never saw.
 */
class NeuralRescorer(
    modelFile: File,
    vocabFile: File,
    mergesFile: File,
    private val gamma: Float = 0.5f,
    private val acousticGuard: Float = 8f,
    val topK: Int = 20,
    private val startId: Long = 0L,                 // <|endoftext|>, the model's start token
    private val maxContextTokens: Int = 48,
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer = Gpt2Tokenizer(vocabFile, mergesFile)

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val t0 = System.nanoTime()
        session = env.createSession(modelFile.absolutePath, opts)
        Log.i(TAG, "neural LM ready in ${(System.nanoTime() - t0) / 1_000_000} ms")
        // One fixed sentence: a broken tokenizer or graph shows up in the log right away.
        runCatching {
            Log.i(TAG, "probe ids ${tokenizer.encode(PROBE).joinToString(",")}")
            Log.i(TAG, "probe logp ${score(listOf(PROBE), "")[0]}")
        }.onFailure { Log.e(TAG, "probe failed", it) }
    }

    /** Natural-log P(sentence | start + context) for each sentence, in one batch. */
    fun score(sentences: List<String>, context: String): FloatArray {
        if (sentences.isEmpty()) return FloatArray(0)
        val ctx = context.trim()
        val ctxIds = if (ctx.isEmpty()) IntArray(0) else tokenizer.encode(ctx).takeLast(maxContextTokens).toIntArray()
        val prefix = LongArray(1 + ctxIds.size).also { p -> p[0] = startId; ctxIds.forEachIndexed { i, id -> p[i + 1] = id.toLong() } }
        val lead = if (ctx.isEmpty()) "" else " "
        val seqs = sentences.map { s -> prefix + tokenizer.encode(lead + s).map { it.toLong() } }
        val n = seqs.maxOf { it.size }
        val ids = LongArray(seqs.size * n) { startId }
        val mask = LongArray(seqs.size * n)
        seqs.forEachIndexed { j, x ->
            for (t in x.indices) { ids[j * n + t] = x[t]; mask[j * n + t] = 1L }
        }
        val shape = longArrayOf(seqs.size.toLong(), n.toLong())
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { tIds ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape).use { tMask ->
                session.run(mapOf("input_ids" to tIds, "attention_mask" to tMask)).use { r ->
                    @Suppress("UNCHECKED_CAST")
                    val lp = r[0].value as Array<FloatArray>          // [batch][tokens - 1]
                    return FloatArray(seqs.size) { j ->
                        var s = 0f
                        // token t is predicted at output position t - 1; score the sentence only
                        for (t in prefix.size until seqs[j].size) s += lp[j][t - 1]
                        s
                    }
                }
            }
        }
    }

    /**
     * Re-ranks the first [topK] hypotheses. Their LM part becomes ngram + gamma * neural, so
     * [BeamSearch.Hyp.total] is the final score; hypotheses the acoustic guard excludes keep
     * their place after the eligible ones with a score that cannot compete.
     */
    fun rerank(hyps: List<BeamSearch.Hyp>, context: String): List<BeamSearch.Hyp> {
        val head = hyps.take(topK)
        if (head.size <= 1) return hyps
        val neural = score(head.map { it.text }, context)
        val bestAcoustic = head.first().acoustic
        val eligible = ArrayList<BeamSearch.Hyp>()
        val excluded = ArrayList<BeamSearch.Hyp>()
        head.forEachIndexed { i, h ->
            val n = BeamSearch.Hyp(h.words, h.acoustic, h.lm + gamma * neural[i])
            if (bestAcoustic - h.acoustic > acousticGuard) excluded.add(BeamSearch.Hyp(h.words, h.acoustic, n.lm - EXCLUDED))
            else eligible.add(n)
        }
        return eligible.sortedByDescending { it.total } + excluded
    }

    override fun close() = session.close()

    companion object {
        private const val TAG = "NeuralRescorer"
        /** Pushes an excluded sentence far below every eligible one (and near 0% on screen). */
        private const val EXCLUDED = 1000f
        private const val PROBE = "αναφέρθηκε στις φήμες ως πολιτικό κουτσομπολιό και ανοησίες"

        fun present(dir: File) = files(dir).all { it.exists() }
        fun files(dir: File) = listOf(File(dir, "el_gpt2.int8.onnx"), File(dir, "el_gpt2.vocab.json"), File(dir, "el_gpt2.merges.txt"))
    }
}
