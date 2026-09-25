package com.openhackathon.voicetotext.asr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Whisper on device: encoder once over a 30 s mel window, then a greedy autoregressive
 * decoder loop.
 *
 * The decoder runs without a KV cache: it re-reads the whole prefix each step. That is
 * wasteful in principle, but this checkpoint (large-v3-turbo) has only 4 decoder layers
 * against 32 encoder layers, so the encoder dominates and the simpler code is worth it.
 * If long utterances become slow, the cached decoder graph is the thing to add.
 *
 * Unlike [CtcModel] this produces no emission matrix. It cannot feed speaker adaptation and
 * it cannot verify anything acoustically; it only writes text.
 */
class WhisperModel(dir: File) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession
    private val tokenizer = WhisperTokenizer(File(dir, "tokens.json"))
    private val mel: MelSpectrogram
    private val meta: JSONObject

    private val sot: Int
    private val lang: Int
    private val transcribe: Int
    private val noTimestamps: Int
    private val eot: Int

    val modelId: String

    init {
        meta = JSONObject(File(dir, "whisper_meta.json").readText())
        modelId = meta.optString("model_id", dir.name)
        sot = meta.getInt("sot"); lang = meta.getInt("lang_el")
        transcribe = meta.getInt("transcribe"); noTimestamps = meta.getInt("notimestamps")
        eot = meta.getInt("eot")
        mel = MelSpectrogram(
            nFft = meta.getInt("n_fft"), hop = meta.getInt("hop_length"),
            nMels = meta.getInt("n_mels"), nSamples = meta.getInt("n_samples"),
            melFiltersFile = File(dir, "mel_filters.bin"),
        )
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val t0 = System.nanoTime()
        encoder = env.createSession(File(dir, ENCODER).absolutePath, opts)
        decoder = env.createSession(File(dir, DECODER).absolutePath, opts)
        Log.i(TAG, "sessions ready in ${(System.nanoTime() - t0) / 1_000_000} ms")
    }

    /** Transcribes one clip. Returns the text and the wall time in ms. */
    fun transcribe(waveform: FloatArray, maxNewTokens: Int = 180): Pair<String, Long> {
        val t0 = System.nanoTime()
        val features = mel.compute(waveform)
        val encShape = longArrayOf(1, meta.getInt("n_mels").toLong(), mel.nFrames.toLong())

        OnnxTensor.createTensor(env, FloatBuffer.wrap(features), encShape).use { input ->
            encoder.run(mapOf(ENCODER_IN to input)).use { encOut ->
                val hidden = encOut[0] as OnnxTensor
                val ids = arrayListOf(sot, lang, transcribe, noTimestamps)
                val out = ArrayList<Int>()
                repeat(maxNewTokens) {
                    val next = step(ids, hidden)
                    if (next == eot) return@repeat
                    ids.add(next)
                    out.add(next)
                }
                val text = tokenizer.decode(out)
                return text to (System.nanoTime() - t0) / 1_000_000
            }
        }
    }

    /** One decoder pass over the whole prefix; returns the argmax of the last position. */
    private fun step(ids: List<Int>, hidden: OnnxTensor): Int {
        val buf = LongBuffer.allocate(ids.size)
        for (i in ids) buf.put(i.toLong())
        buf.rewind()
        OnnxTensor.createTensor(env, buf, longArrayOf(1, ids.size.toLong())).use { inputIds ->
            decoder.run(mapOf(DECODER_IDS to inputIds, DECODER_ENC to hidden)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val logits = (r[0].value as Array<Array<FloatArray>>)[0]
                val last = logits[logits.size - 1]
                var best = 0
                for (v in 1 until last.size) if (last[v] > last[best]) best = v
                return best
            }
        }
    }

    override fun close() {
        runCatching { decoder.close() }
        runCatching { encoder.close() }
    }

    companion object {
        private const val TAG = "WhisperModel"
        const val ENCODER = "encoder_model.onnx"
        const val DECODER = "decoder_model.onnx"
        private const val ENCODER_IN = "input_features"
        private const val DECODER_IDS = "input_ids"
        private const val DECODER_ENC = "encoder_hidden_states"

        /** Everything the on-device Whisper needs, relative to the app files directory. */
        fun filesPresent(dir: File): Boolean =
            File(dir, ENCODER).exists() && File(dir, DECODER).exists() &&
                File(dir, "tokens.json").exists() && File(dir, "mel_filters.bin").exists() &&
                File(dir, "whisper_meta.json").exists()
    }
}
