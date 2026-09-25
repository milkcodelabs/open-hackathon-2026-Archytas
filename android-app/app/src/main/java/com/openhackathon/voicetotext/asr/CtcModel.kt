package com.openhackathon.voicetotext.asr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * The acoustic layer: 16 kHz mono waveform in, [Emissions] out.
 *
 * Mirrors `greek_vt/acoustic/ctc_model.py`. The exported graph starts *after* the
 * Wav2Vec2 feature extractor's normalization, so the same zero-mean/unit-variance step is
 * applied here; getting this wrong silently degrades every number downstream.
 */
class CtcModel(private val modelFile: File, labelsFile: File) : AutoCloseable {

    val labels: List<String> = parseLabels(labelsFile.readText())
    val modelId: String = modelFile.name

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    var frameDurationMs: Float = 20f

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val t0 = System.nanoTime()
        session = env.createSession(modelFile.absolutePath, opts)
        Log.i(TAG, "session ready in ${(System.nanoTime() - t0) / 1_000_000} ms, V=${labels.size}")
    }

    /** Runs the encoder on one clip. Returns the emission matrix and the wall time in ms. */
    fun emit(waveform: FloatArray, audioId: String = ""): Pair<Emissions, Long> {
        val x = normalize(waveform)
        val shape = longArrayOf(1, x.size.toLong())
        val t0 = System.nanoTime()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), shape).use { input ->
            session.run(mapOf(INPUT to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result[0].value as Array<Array<FloatArray>>)[0]
                val frames = logits.size
                val v = labels.size
                val flat = FloatArray(frames * v)
                for (t in 0 until frames) System.arraycopy(logits[t], 0, flat, t * v, v)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val em = Emissions.fromLogits(flat, labels, frameDurationMs, audioId, modelId)
                return em to ms
            }
        }
    }

    /** Zero mean, unit variance over the whole clip, matching Wav2Vec2FeatureExtractor. */
    private fun normalize(wav: FloatArray): FloatArray {
        var mean = 0.0
        for (s in wav) mean += s
        mean /= wav.size
        var varSum = 0.0
        for (s in wav) { val d = s - mean; varSum += d * d }
        val std = sqrt(varSum / wav.size + 1e-7)
        return FloatArray(wav.size) { ((wav[it] - mean) / std).toFloat() }
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "CtcModel"
        private const val INPUT = "input_values"

        /**
         * The labels file is the JSON array written by `gvt export onnx`. Uses the platform
         * JSON parser so that escapes (the vocabulary contains a bare combining acute, U+0301)
         * are handled correctly.
         */
        fun parseLabels(json: String): List<String> {
            val arr = org.json.JSONArray(json)
            return List(arr.length()) { arr.getString(it) }
        }
    }
}
