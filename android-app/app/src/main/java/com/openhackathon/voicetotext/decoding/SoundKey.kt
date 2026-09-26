package com.openhackathon.voicetotext.decoding

import java.text.Normalizer
import java.util.Locale

/**
 * The pronunciation class of a Greek word. Words are looked up in `el_homophones.bin` by
 * this key, so it must follow exactly the rules the index was built with.
 *
 *  1. lowercase, NFD, drop the stress mark; a diaeresis makes its ι/υ a stand-alone /i/;
 *  2. αυ ευ ηυ keep their consonant value (av ev iv);
 *  3. ει οι υι -> i, αι -> e, ου -> u;
 *  4. ι η υ -> i, ο ω -> o, ε -> e, ς -> σ;
 *  5. doubled consonants -> single (γγ stays, it is /ng/).
 */
object SoundKey {
    private const val ACUTE = "́"
    private const val DIAERESIS = "̈"
    private const val LONE_I = "\u0001"

    private val DIGRAPHS = listOf(
        "αυ" to "av", "ευ" to "ev", "ηυ" to "iv",
        "ει" to "i", "οι" to "i", "υι" to "i", "αι" to "e", "ου" to "u",
    )
    private val SINGLES = mapOf(
        'ι' to 'i', 'η' to 'i', 'υ' to 'i', '\u0001' to 'i',
        'ο' to 'o', 'ω' to 'o', 'ε' to 'e', 'ς' to 'σ',
    )
    private const val DOUBLES = "λμνπρστκβφθχδζξψ"

    fun of(word: String): String {
        var s = Normalizer.normalize(word.lowercase(Locale.ROOT), Normalizer.Form.NFD).replace(ACUTE, "")
        s = s.replace("ι$DIAERESIS", LONE_I).replace("υ$DIAERESIS", LONE_I).replace(DIAERESIS, "")
        for ((a, b) in DIGRAPHS) s = s.replace(a, b)
        val sb = StringBuilder(s.length)
        for (c in s) sb.append(SINGLES[c] ?: c)
        s = sb.toString()
        for (c in DOUBLES) s = s.replace("$c$c", "$c")
        return Normalizer.normalize(s, Normalizer.Form.NFC)
    }
}
