package com.openhackathon.voicetotext.decoding

import org.json.JSONObject
import java.io.File
import java.util.regex.Pattern

/**
 * GPT-2 byte-level BPE encoder, the tokenizer of the layer-2c neural LM
 * (`el_gpt2.vocab.json` + `el_gpt2.merges.txt`).
 *
 * Same three steps as Hugging Face's GPT2Tokenizer:
 *  1. split the text with GPT-2's pattern (a leading space stays with its word);
 *  2. map every UTF-8 byte of a piece to a printable character (bytes_to_unicode);
 *  3. merge adjacent symbols by the rank of the pair in merges.txt until none applies.
 */
class Gpt2Tokenizer(vocabFile: File, mergesFile: File) {

    private val vocab: Map<String, Int>
    private val ranks: Map<String, Int>
    private val byteChar = CharArray(256)
    private val cache = HashMap<String, IntArray>()

    init {
        val json = JSONObject(vocabFile.readText())
        val v = HashMap<String, Int>(json.length() * 2)
        for (k in json.keys()) v[k] = json.getInt(k)
        vocab = v

        val r = HashMap<String, Int>()
        var rank = 0
        mergesFile.forEachLine { line ->
            if (line.isNotBlank() && !line.startsWith("#version")) r[line] = rank++
        }
        ranks = r

        // GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to 256+n
        val printable = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toSet()
        var n = 0
        for (b in 0 until 256) {
            byteChar[b] = if (b in printable) b.toChar() else (256 + n++).toChar()
        }
    }

    /** Token ids of [text], no special tokens added. */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        val m = PIECES.matcher(text)
        while (m.find()) {
            val piece = m.group()
            val ids = cache.getOrPut(piece) { bpe(piece) }
            for (id in ids) out.add(id)
        }
        return out.toIntArray()
    }

    private fun bpe(piece: String): IntArray {
        val bytes = piece.toByteArray(Charsets.UTF_8)
        var symbols = MutableList(bytes.size) { byteChar[bytes[it].toInt() and 0xff].toString() }
        while (symbols.size > 1) {
            var best = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until symbols.size - 1) {
                val rk = ranks[symbols[i] + " " + symbols[i + 1]] ?: continue
                if (rk < bestRank) { bestRank = rk; best = i }
            }
            if (best < 0) break
            val a = symbols[best]
            val b = symbols[best + 1]
            // merge every occurrence of the pair, left to right, as the reference does
            val merged = ArrayList<String>(symbols.size)
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == a && symbols[i + 1] == b) {
                    merged.add(a + b); i += 2
                } else {
                    merged.add(symbols[i]); i += 1
                }
            }
            symbols = merged
        }
        return IntArray(symbols.size) { vocab[symbols[it]] ?: UNKNOWN }
    }

    companion object {
        /** GPT-2's pre-tokenization pattern. */
        private val PIECES: Pattern = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )
        /** Every byte has a token in a byte-level vocabulary, so this should not happen. */
        private const val UNKNOWN = 0
    }
}
