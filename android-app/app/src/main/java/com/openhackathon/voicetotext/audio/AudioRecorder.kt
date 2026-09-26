package com.openhackathon.voicetotext.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Microphone capture at 16 kHz mono float, the rate the CTC model expects. */
class AudioRecorder {

    private var record: AudioRecord? = null
    @Volatile private var recording = false
    private val chunks = ArrayList<FloatArray>()

    @SuppressLint("MissingPermission")
    fun start() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        val bufSize = maxOf(minBuf, SAMPLE_RATE * 2)
        val r = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, CHANNEL, ENCODING, bufSize)
        check(r.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }
        chunks.clear()
        record = r
        recording = true
        r.startRecording()
        Thread {
            val buf = FloatArray(bufSize / 4)
            while (recording) {
                val n = r.read(buf, 0, buf.size, AudioRecord.READ_BLOCKING)
                if (n > 0) synchronized(chunks) { chunks.add(buf.copyOf(n)) }
            }
        }.start()
    }

    /** Stops and returns the whole clip as one float array in [-1, 1]. */
    fun stop(): FloatArray {
        recording = false
        record?.run { stop(); release() }
        record = null
        synchronized(chunks) {
            val total = chunks.sumOf { it.size }
            val out = FloatArray(total)
            var at = 0
            for (c in chunks) { c.copyInto(out, at); at += c.size }
            Log.i(TAG, "captured $total samples (${total / SAMPLE_RATE.toFloat()} s)")
            return out
        }
    }

    companion object {
        private const val TAG = "AudioRecorder"
        const val SAMPLE_RATE = 16000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_FLOAT

        /**
         * Minimal WAV reader for files pushed with `adb push`: 16-bit or 32-bit float PCM,
         * mono or stereo, any rate. Resampling is linear, which is adequate for test clips.
         */
        fun readWav(file: File, targetRate: Int = SAMPLE_RATE): FloatArray {
            RandomAccessFile(file, "r").use { f ->
                val header = ByteArray(12)
                f.readFully(header)
                var channels = 1
                var rate = targetRate
                var bits = 16
                var format = 1
                var data: FloatArray? = null
                while (f.filePointer < f.length() - 8) {
                    val idBytes = ByteArray(4)
                    f.readFully(idBytes)
                    val id = String(idBytes)
                    val sizeBytes = ByteArray(4)
                    f.readFully(sizeBytes)
                    val size = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).int
                    when (id) {
                        "fmt " -> {
                            val fmt = ByteArray(size)
                            f.readFully(fmt)
                            val bb = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
                            format = bb.short.toInt()
                            channels = bb.short.toInt()
                            rate = bb.int
                            bb.int; bb.short
                            bits = bb.short.toInt()
                        }
                        "data" -> {
                            val raw = ByteArray(size)
                            f.readFully(raw)
                            val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
                            val n = size / (bits / 8)
                            val mono = FloatArray(n / channels)
                            var k = 0
                            while (k < mono.size) {
                                var acc = 0f
                                for (c in 0 until channels) {
                                    acc += when {
                                        format == 3 && bits == 32 -> bb.float
                                        bits == 16 -> bb.short / 32768f
                                        bits == 32 -> bb.int / 2147483648f
                                        else -> throw IllegalArgumentException("unsupported wav: $bits bit, format $format")
                                    }
                                }
                                mono[k++] = acc / channels
                            }
                            data = mono
                        }
                        else -> f.skipBytes(size)
                    }
                    if (size % 2 == 1) f.skipBytes(1)
                }
                val wav = data ?: throw IllegalArgumentException("no data chunk in ${file.name}")
                if (rate == targetRate) return wav
                val ratio = targetRate.toDouble() / rate
                val out = FloatArray((wav.size * ratio).toInt())
                for (i in out.indices) {
                    val src = i / ratio
                    val a = src.toInt().coerceAtMost(wav.size - 1)
                    val b = (a + 1).coerceAtMost(wav.size - 1)
                    val frac = (src - a).toFloat()
                    out[i] = wav[a] * (1 - frac) + wav[b] * frac
                }
                return out
            }
        }
    }
}
