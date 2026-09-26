package com.openhackathon.voicetotext.decoding

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Sound key -> real spellings of that sound, most frequent first. Memory mapped from
 * `el_homophones.bin`:
 *
 *     8-byte magic, int32 n, (n+1) int32 key offsets, (n+1) int32 value offsets,
 *     key blob (UTF-8), value blob (UTF-8, spellings separated by one space)
 *
 * Keys are sorted by their UTF-8 bytes. They hold only Greek and Latin letters, all below
 * U+D800, where that order is the same as String.compareTo, so a plain binary search works.
 */
class HomophoneIndex private constructor(
    private val buf: ByteBuffer,
    val size: Int,
    private val offKeyIdx: Int,
    private val offValIdx: Int,
    private val offKeyBlob: Int,
    private val offValBlob: Int,
) {
    private fun slice(idx: Int, blob: Int, i: Int): String {
        val s = buf.getInt(idx + i * 4)
        val e = buf.getInt(idx + (i + 1) * 4)
        val bytes = ByteArray(e - s)
        val dup = buf.duplicate()
        dup.position(blob + s)
        dup.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    /** Spellings of this sound key (empty when the vocabulary has none). */
    fun group(key: String): List<String> {
        var lo = 0
        var hi = size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = slice(offKeyIdx, offKeyBlob, mid).compareTo(key)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return slice(offValIdx, offValBlob, mid).split(' ')
            }
        }
        return emptyList()
    }

    companion object {
        private val MAGIC = "HOMIDX1\u0000".toByteArray(Charsets.US_ASCII)

        fun load(file: File): HomophoneIndex {
            val ch = RandomAccessFile(file, "r").channel
            val buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(8)
            buf.get(magic)
            require(magic.contentEquals(MAGIC)) { "${file.name} is not a homophone index" }
            val n = buf.int
            val offKeyIdx = 12
            val offValIdx = offKeyIdx + (n + 1) * 4
            val offKeyBlob = offValIdx + (n + 1) * 4
            val keyBlobLen = buf.getInt(offKeyIdx + n * 4)
            return HomophoneIndex(buf, n, offKeyIdx, offValIdx, offKeyBlob, offKeyBlob + keyBlobLen)
        }
    }
}
