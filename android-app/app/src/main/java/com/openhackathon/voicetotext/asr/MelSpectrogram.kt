package com.openhackathon.voicetotext.asr

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * Log-mel spectrogram, bit-for-bit the same recipe as HuggingFace's WhisperFeatureExtractor.
 * Getting this wrong does not crash, it just degrades the transcription, so the constants are
 * taken from the exported `whisper_meta.json` rather than hardcoded.
 *
 * Steps: pad or trim to exactly 30 s, reflect-pad by nFft/2, framed STFT with a periodic Hann
 * window, power spectrum, mel projection, log10, dynamic-range clamp to 8 decades, scale.
 */
class MelSpectrogram(
    private val nFft: Int,
    private val hop: Int,
    private val nMels: Int,
    private val nSamples: Int,
    melFiltersFile: File,
) {
    private val nFreq = nFft / 2 + 1
    val nFrames = nSamples / hop           // 3000 for a 30 s window

    /** (nFreq, nMels), row-major, as written by the exporter. */
    private val melFilters: FloatArray

    private val window = FloatArray(nFft) { 0.5f - 0.5f * cos(2.0 * PI * it / nFft).toFloat() }

    // Precomputed DFT twiddles: nFft is 400, not a power of two, so a direct real DFT with a
    // lookup table is simpler than mixed-radix and is not the bottleneck next to the encoder.
    private val cosT = FloatArray(nFreq * nFft)
    private val sinT = FloatArray(nFreq * nFft)

    init {
        val bytes = melFiltersFile.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val rows = bb.int
        val cols = bb.int
        require(rows == nFreq && cols == nMels) { "mel filters are ${rows}x$cols, expected ${nFreq}x$nMels" }
        melFilters = FloatArray(rows * cols) { bb.float }
        for (k in 0 until nFreq) {
            for (n in 0 until nFft) {
                val a = -2.0 * PI * k * n / nFft
                cosT[k * nFft + n] = cos(a).toFloat()
                sinT[k * nFft + n] = sin(a).toFloat()
            }
        }
    }

    /** Returns (nMels * nFrames) row-major, ready for the ONNX encoder input. */
    fun compute(waveform: FloatArray): FloatArray {
        val x = FloatArray(nSamples)
        System.arraycopy(waveform, 0, x, 0, minOf(waveform.size, nSamples))

        val pad = nFft / 2
        val padded = FloatArray(nSamples + 2 * pad)
        for (i in 0 until pad) padded[i] = x[pad - i]                       // reflect
        System.arraycopy(x, 0, padded, pad, nSamples)
        for (i in 0 until pad) padded[nSamples + pad + i] = x[nSamples - 2 - i]

        val power = FloatArray(nFreq)
        val mel = FloatArray(nMels * nFrames)
        val frame = FloatArray(nFft)
        var globalMax = -Float.MAX_VALUE

        for (t in 0 until nFrames) {
            val off = t * hop
            for (n in 0 until nFft) frame[n] = padded[off + n] * window[n]
            for (k in 0 until nFreq) {
                var re = 0f
                var im = 0f
                val base = k * nFft
                for (n in 0 until nFft) {
                    val v = frame[n]
                    re += v * cosT[base + n]
                    im += v * sinT[base + n]
                }
                power[k] = re * re + im * im
            }
            for (m in 0 until nMels) {
                var acc = 0f
                for (k in 0 until nFreq) acc += power[k] * melFilters[k * nMels + m]
                val v = log10(max(acc, 1e-10f).toDouble()).toFloat()
                mel[m * nFrames + t] = v
                if (v > globalMax) globalMax = v
            }
        }

        val floor = globalMax - 8f
        for (i in mel.indices) mel[i] = (max(mel[i], floor) + 4f) / 4f
        return mel
    }
}
