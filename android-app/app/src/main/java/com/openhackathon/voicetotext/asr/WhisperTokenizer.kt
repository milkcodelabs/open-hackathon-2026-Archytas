package com.openhackathon.voicetotext.asr

import org.json.JSONArray
import java.io.File

/**
 * Decode-only side of Whisper's byte-level BPE. We never encode text on device, so all that
 * is needed is id -> token string, concatenation, and the GPT-2 byte-level unmapping before
 * a UTF-8 decode. Without that last step Greek comes out as mojibake, because each token
 * string holds bytes disguised as printable code points.
 */
class WhisperTokenizer(tokensFile: File) {

    private val tokens: Array<String>

    init {
        val arr = JSONArray(tokensFile.readText())
        tokens = Array(arr.length()) { arr.getString(it) }
    }

    val size: Int get() = tokens.size

    fun token(id: Int): String = if (id in tokens.indices) tokens[id] else ""

    /** True for <|...|> control tokens, which must never reach the user. */
    fun isSpecial(id: Int): Boolean {
        val t = token(id)
        return t.startsWith("<|") && t.endsWith("|>")
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (isSpecial(id)) continue
            sb.append(token(id))
        }
        val bytes = ArrayList<Byte>(sb.length)
        for (ch in sb) {
            val b = byteOf(ch)
            if (b >= 0) bytes.add(b.toByte())
        }
        return String(bytes.toByteArray(), Charsets.UTF_8).trim()
    }

    companion object {
        /** Inverse of GPT-2's bytes_to_unicode table, built once. */
        private val unicodeToByte = HashMap<Char, Int>(256).apply {
            val bs = ArrayList<Int>()
            for (b in '!'.code..'~'.code) bs.add(b)
            for (b in 0xA1..0xAC) bs.add(b)
            for (b in 0xAE..0xFF) bs.add(b)
            val cs = ArrayList<Int>(bs)
            var n = 0
            for (b in 0 until 256) {
                if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
            }
            for (i in bs.indices) put(cs[i].toChar(), bs[i])
        }

        private fun byteOf(ch: Char): Int = unicodeToByte[ch] ?: -1
    }
}
