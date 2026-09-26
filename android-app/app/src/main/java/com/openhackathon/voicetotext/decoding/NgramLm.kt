package com.openhackathon.voicetotext.decoding

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Word n-gram language model, memory mapped.
 *
 * KenLM's own binary is a C++ trie and reading it here would mean shipping the NDK, so the
 * same information is stored as sorted arrays and looked up with a binary search. Scores are
 * base-10 logs, exactly as in the ARPA, and backoff follows the standard recipe:
 * try the longest context, otherwise fall back a level and pay its backoff weight.
 */
class NgramLm private constructor(
    private val buf: ByteBuffer,
    val order: Int,
    val vocabSize: Int,
    private val nUni: Int,
    private val nBi: Int,
    private val nTri: Int,
    private val offVocabIdx: Int,
    private val offVocabBlob: Int,
    private val offUni: Int,
    private val offBi: Int,
    private val offTri: Int,
) {
    /** Unknown word. Every out-of-vocabulary token maps here and is heavily penalised. */
    val unk = -1

    private fun wordAt(i: Int): String {
        val s = buf.getInt(offVocabIdx + i * 4)
        val e = buf.getInt(offVocabIdx + (i + 1) * 4)
        val bytes = ByteArray(e - s)
        val dup = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        dup.position(offVocabBlob + s)
        dup.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    /** Word id, or [unk]. Binary search over the sorted vocabulary blob. */
    fun id(word: String): Int {
        var lo = 0
        var hi = vocabSize - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = wordAt(mid).compareTo(word)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return mid
            }
        }
        return unk
    }

    fun contains(word: String): Boolean = id(word) != unk

    /**
     * Can this half-typed word still become a real one?
     *
     * The beam search asks at every frame. Without it a prefix that no word starts with
     * survives until the next space, wasting beam width on a dead end. The vocabulary blob
     * is sorted, so this is a lower-bound binary search and costs the same as any lookup.
     */
    fun hasPrefix(partial: String): Boolean {
        if (partial.isEmpty()) return true
        var lo = 0
        var hi = vocabSize
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (wordAt(mid) < partial) lo = mid + 1 else hi = mid
        }
        return lo < vocabSize && wordAt(lo).startsWith(partial)
    }

    private fun key(ids: IntArray): Long {
        var k = 0L
        for (i in ids) k = (k shl ID_BITS) or i.toLong()
        return k
    }

    /** Binary search in a record array; returns the record index or -1. */
    private fun find(base: Int, count: Int, stride: Int, k: Long): Int {
        var lo = 0
        var hi = count - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val cur = buf.getLong(base + mid * stride)
            when {
                cur < k -> lo = mid + 1
                cur > k -> hi = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    private fun uniLogp(w: Int) = buf.getFloat(offUni + w * 8)
    private fun uniBackoff(w: Int) = buf.getFloat(offUni + w * 8 + 4)

    /**
     * log10 P(word | context), context most-recent-last, with ARPA backoff.
     *
     * The recipe is: take the longest n-gram that exists; otherwise pay the backoff weight
     * *of the context that was dropped* and try one order shorter. Forgetting that weight
     * silently biases every score, which is what the check against KenLM caught.
     */
    fun score(context: IntArray, word: Int): Float {
        if (word == unk) return OOV_LOGP
        if (order >= 3 && context.size >= 2) {
            val w1 = context[context.size - 2]
            val w2 = context.last()
            if (w1 != unk && w2 != unk) {
                val tri = find(offTri, nTri, 12, key(intArrayOf(w1, w2, word)))
                if (tri >= 0) return buf.getFloat(offTri + tri * 12 + 8)
                // the trigram is absent: back off, paying bo(w1 w2) when that context exists
                var bo = 0f
                val ctx = find(offBi, nBi, 16, key(intArrayOf(w1, w2)))
                if (ctx >= 0) bo += buf.getFloat(offBi + ctx * 16 + 12)
                return bo + bigramScore(w2, word)
            }
        }
        if (context.isNotEmpty()) return bigramScore(context.last(), word)
        return uniLogp(word)
    }

    private fun bigramScore(ctx: Int, word: Int): Float {
        if (ctx == unk) return uniLogp(word)
        val i = find(offBi, nBi, 16, key(intArrayOf(ctx, word)))
        if (i >= 0) return buf.getFloat(offBi + i * 16 + 8)
        return uniBackoff(ctx) + uniLogp(word)
    }

    fun score(context: List<String>, word: String): Float =
        score(IntArray(context.size) { id(context[it]) }, id(word))

    companion object {
        private const val TAG = "NgramLm"
        private val MAGIC = "NGRAM1\u0000\u0000".toByteArray(Charsets.US_ASCII)
        private const val ID_BITS = 21
        /** Roughly the cost of an unseen word; mirrors pyctcdecode's unk_score_offset. */
        const val OOV_LOGP = -10f

        fun load(file: File): NgramLm {
            val ch = RandomAccessFile(file, "r").channel
            val buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(8).also { buf.get(it) }
            require(magic.contentEquals(MAGIC)) { "not an n-gram LM file: ${file.name}" }
            val order = buf.int
            val vocabSize = buf.int
            val nUni = buf.int
            val nBi = buf.int
            val nTri = buf.int
            val blobLen = buf.int
            val offVocabIdx = buf.position()
            val offVocabBlob = offVocabIdx + (vocabSize + 1) * 4
            val offUni = offVocabBlob + blobLen
            val offBi = offUni + nUni * 8
            val offTri = offBi + nBi * 16
            Log.i(TAG, "loaded ${file.name}: order $order, $vocabSize words, $nBi bigrams, $nTri trigrams")
            return NgramLm(buf, order, vocabSize, nUni, nBi, nTri, offVocabIdx, offVocabBlob, offUni, offBi, offTri)
        }
    }
}
