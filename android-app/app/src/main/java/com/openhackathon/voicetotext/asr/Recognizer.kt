package com.openhackathon.voicetotext.asr

import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.decoding.BeamSearch
import com.openhackathon.voicetotext.decoding.Candidate
import com.openhackathon.voicetotext.decoding.Candidates
import com.openhackathon.voicetotext.decoding.HomophoneIndex
import com.openhackathon.voicetotext.decoding.NgramLm
import com.openhackathon.voicetotext.decoding.SpellingRescorer
import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import java.io.File
import java.text.Normalizer
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Process-wide owner of the acoustic models, shared by the bubble and the control panel.
 *
 * Two acoustic engines, both CTC (same emission matrix, same layer 2):
 *
 *  - [Engine.OMNI] Meta Omnilingual CTC 300M, restricted to the Greek columns inside the
 *    graph. The default layer 1: WER 0.139 on FLEURS with the LM instead of 0.430.
 *  - [Engine.CTC] wav2vec2 (lighteternal). Still the better one on Common Voice style
 *    speech (0.073 against 0.132), so it stays selectable.
 */
object Recognizer {
    private const val TAG = "Recognizer"
    private const val PREFS = "gvt"
    private const val KEY_ENGINE = "engine"
    private const val KEY_LM = "use_lm"

    enum class Engine { OMNI, CTC }

    private const val KEY_OMNI_DEFAULT = "omni_default_applied"

    @Volatile private var ctc: CtcModel? = null
    /** Which CTC engine [ctc] holds: both CTC engines share the slot, never both loaded. */
    @Volatile private var ctcEngine: Engine? = null
    @Volatile private var lm: NgramLm? = null
    @Volatile private var beam: BeamSearch? = null
    /** Layer 2b; null when el_homophones.bin is missing (then 2a alone runs). */
    @Volatile private var homophones: HomophoneIndex? = null
    @Volatile private var speller: SpellingRescorer? = null
    /** The speaker's own words (my_words.txt), normalized like the model's labels. */
    @Volatile private var personalWords: List<String> = emptyList()

    /** How many hypotheses layer 2a hands to layer 2b. */
    private const val N_BEST = 50
    /** How many alternative sentences the screen offers. */
    const val MAX_CANDIDATES = 10

    private val _last = MutableStateFlow<Result?>(null)
    /** The latest recognition, from the panel or the bubble, for the candidates list. */
    val lastResult: StateFlow<Result?> = _last.asStateFlow()
    /** Words already in the text field that seed the LM (the 3-gram uses two). */
    private const val CONTEXT_WORDS = 2

    /** Layer 2. Off means the phone runs bare greedy decoding, which is what it did before. */
    @Volatile var useLanguageModel: Boolean = true
        private set
    @Volatile var speakerId: String = "default"   // TODO(M5): set by enrollment
    @Volatile var lastError: String? = null

    @Volatile var engine: Engine = Engine.OMNI
        private set

    private fun prefs(ctx: Context): SharedPreferences =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun restoreEngine(ctx: Context) {
        val p = prefs(ctx)
        engine = runCatching { Engine.valueOf(p.getString(KEY_ENGINE, Engine.OMNI.name)!!) }
            .getOrDefault(Engine.OMNI)
        // Omnilingual became layer 1 after the saved choice was made: switch once, when its
        // files are there. After that the user's own choice is kept.
        if (!p.getBoolean(KEY_OMNI_DEFAULT, false) && omniPresent(ctx)) {
            engine = Engine.OMNI
            p.edit().putString(KEY_ENGINE, engine.name).putBoolean(KEY_OMNI_DEFAULT, true).apply()
        }
        useLanguageModel = p.getBoolean(KEY_LM, true)
    }

    fun setUseLanguageModel(ctx: Context, on: Boolean) {
        useLanguageModel = on
        prefs(ctx).edit().putBoolean(KEY_LM, on).apply()
        if (!on) { beam = null; speller = null }
        val m = ctc
        val e = ctcEngine
        if (on && m != null && e != null && lmPresent(ctx)) {
            val n = lm ?: NgramLm.load(lmFile(ctx)).also { lm = it }
            buildLayer2(ctx, e, m.labels, n)
        }
        Log.i(TAG, "language model = $on")
    }

    fun lmFile(ctx: Context): File = File(filesRoot(ctx), "el_3gram.gvtlm")
    fun lmPresent(ctx: Context): Boolean = lmFile(ctx).exists()
    fun lmSizeMb(ctx: Context): Long = lmFile(ctx).length() / 1_000_000
    fun homophonesFile(ctx: Context): File = File(filesRoot(ctx), "el_homophones.bin")
    /** One word or name per line; imported like the models. */
    fun myWordsFile(ctx: Context): File = File(filesRoot(ctx), "my_words.txt")

    fun setEngine(ctx: Context, e: Engine) {
        engine = e
        prefs(ctx).edit().putString(KEY_ENGINE, e.name).apply()
        Log.i(TAG, "engine = $e")
    }

    // ------------------------------------------------------------------ files

    fun filesRoot(ctx: Context): File = ctx.getExternalFilesDir(null) ?: ctx.filesDir
    fun modelFile(ctx: Context): File = File(filesRoot(ctx), "model.onnx")
    fun labelsFile(ctx: Context): File = File(filesRoot(ctx), "labels.json")
    fun omniFile(ctx: Context): File = File(filesRoot(ctx), "omni.onnx")
    fun omniLabelsFile(ctx: Context): File = File(filesRoot(ctx), "omni.labels.json")
    fun testWav(ctx: Context): File = File(filesRoot(ctx), "test.wav")

    fun ctcPresent(ctx: Context): Boolean = modelFile(ctx).exists() && labelsFile(ctx).exists()
    fun omniPresent(ctx: Context): Boolean = omniFile(ctx).exists() && omniLabelsFile(ctx).exists()

    /** Whether the engine the user picked can actually run. */
    fun filesPresent(ctx: Context): Boolean = when (engine) {
        Engine.OMNI -> omniPresent(ctx)
        Engine.CTC -> ctcPresent(ctx)
    }

    fun sizeMb(ctx: Context): Long = when (engine) {
        Engine.OMNI -> omniFile(ctx).length() / 1_000_000
        Engine.CTC -> modelFile(ctx).length() / 1_000_000
    }

    val ready: Boolean get() = ctc != null && ctcEngine == engine

    /** alpha/beta tuned on FLEURS dev for each acoustic model (RESULTS.md). */
    private fun beamFor(e: Engine, labels: List<String>, n: NgramLm, words: List<String>): BeamSearch =
        if (e == Engine.OMNI) BeamSearch(labels, n, alpha = 0.7f, beta = 3.0f, personalWords = words)
        else BeamSearch(labels, n, alpha = 0.7f, beta = 4.0f, personalWords = words)

    /** Layer 2a (beam + LM + context + the speaker's words) and, if its index is there, 2b. */
    private fun buildLayer2(ctx: Context, e: Engine, labels: List<String>, n: NgramLm) {
        personalWords = readMyWords(ctx, labels)
        val b = beamFor(e, labels, n, personalWords)
        beam = b
        val hf = homophonesFile(ctx)
        val idx = homophones ?: if (hf.exists()) runCatching { HomophoneIndex.load(hf) }
            .onFailure { Log.e(TAG, "homophone index unreadable", it) }.getOrNull()?.also { homophones = it } else null
        speller = idx?.let { SpellingRescorer(it, b, personalWords) }
        Log.i(TAG, "layer 2: ${personalWords.size} personal words, spelling pass ${if (speller != null) "on" else "off (no el_homophones.bin)"}")
    }

    /** After the LM, the homophone index or my_words.txt was imported: rebuild layer 2 only. */
    @Synchronized
    fun reloadLayer2(ctx: Context) {
        homophones = null
        val m = ctc ?: return
        val e = ctcEngine ?: return
        if (!useLanguageModel || !lmPresent(ctx)) return
        lm = null
        buildLayer2(ctx, e, m.labels, NgramLm.load(lmFile(ctx)).also { lm = it })
    }

    private fun readMyWords(ctx: Context, labels: List<String>): List<String> {
        val f = myWordsFile(ctx)
        if (!f.exists()) return emptyList()
        return runCatching { f.readLines() }.getOrDefault(emptyList())
            .flatMap { normalizeWords(it, labels) }.distinct()
    }

    /**
     * Text -> the model's words: lowercase, NFC, every character outside the label set
     * becomes a space. The same rule the desktop normalizer applies to references.
     */
    fun normalizeWords(text: String, labels: List<String>): List<String> {
        val chars = labels.filter { it.length == 1 && it != "|" }.map { it[0] }.toSet()
        val s = Normalizer.normalize(text.lowercase(Locale.ROOT), Normalizer.Form.NFC)
        val sb = StringBuilder(s.length)
        for (c in s) sb.append(if (c in chars) c else ' ')
        return sb.toString().split(' ').filter { it.isNotBlank() }
    }

    /** The last words before the cursor, from the last sentence end on. */
    private fun contextWords(context: String?, labels: List<String>): List<String> {
        if (context.isNullOrBlank()) return emptyList()
        val tail = context.split('.', '!', '?', ';', '\u037e', '\u2026', '\n').last()
        return normalizeWords(tail, labels).takeLast(CONTEXT_WORDS)
    }

    // ------------------------------------------------------------------ loading

    @Synchronized
    fun load(ctx: Context): Boolean {
        if (ready) return true
        if (!filesPresent(ctx)) {
            lastError = when (engine) {
                Engine.OMNI -> "Λείπει το omni.onnx στο ${filesRoot(ctx).absolutePath}"
                Engine.CTC -> "Λείπει το model.onnx στο ${filesRoot(ctx).absolutePath}"
            }
            return false
        }
        val t0 = System.currentTimeMillis()
        return try {
            ctc?.close(); ctc = null; beam = null; speller = null
            val m = if (engine == Engine.OMNI) CtcModel(omniFile(ctx), omniLabelsFile(ctx))
                    else CtcModel(modelFile(ctx), labelsFile(ctx))
            ctc = m
            ctcEngine = engine
            if (useLanguageModel && lmPresent(ctx)) {
                val n = lm ?: NgramLm.load(lmFile(ctx)).also { lm = it }
                buildLayer2(ctx, engine, m.labels, n)
            }
            lastError = null
            Log.i(TAG, "$engine ready in ${System.currentTimeMillis() - t0} ms")
            true
        } catch (e: Exception) {
            lastError = e.toString()
            Log.e(TAG, "load failed for $engine", e)
            false
        }
    }

    /** Frees the engine that is no longer selected, so both never sit in memory at once. */
    @Synchronized
    fun releaseUnused() {
        if (ctcEngine != engine) { ctc?.close(); ctc = null; ctcEngine = null; beam = null; speller = null }
    }

    // ------------------------------------------------------------------ recognition

    class Result(
        val text: String,
        val inferenceMs: Long,
        val audioSeconds: Float,
        val engine: Engine,
        val emissions: Emissions,
        val usedLanguageModel: Boolean = false,
        val beamMs: Long = 0,
        /** The best sentences, best first; [text] is the first. One entry without the LM. */
        val candidates: List<Candidate> = emptyList(),
    ) {
        val rtf: Float get() = inferenceMs / 1000f / audioSeconds
    }

    /**
     * [context]: the text already in the field the result will be typed into (text before
     * the cursor), or null. Layer 2 continues it instead of starting a new sentence.
     */
    fun recognize(waveform: FloatArray, audioId: String = "", context: String? = null): Result {
        val seconds = waveform.size / AudioRecorder.SAMPLE_RATE.toFloat()
        val m = ctc ?: throw IllegalStateException("το μοντέλο δεν φορτώθηκε")
        val (em, acousticMs) = m.emit(waveform, audioId)
        val b = beam
        val result = if (b == null) {
            val text = em.greedyDecode()
            Result(text, acousticMs, seconds, engine, em, false, 0, Candidates.single(text))
        } else {
            val t0 = System.nanoTime()
            val history = contextWords(context, m.labels)
            var hyps = b.decode(em, N_BEST, history)
            speller?.let { hyps = it.rescore(hyps, history) }
            val candidates = Candidates.from(hyps, MAX_CANDIDATES)
            val text = candidates.firstOrNull()?.text ?: em.greedyDecode()
            val beamMs = (System.nanoTime() - t0) / 1_000_000
            Result(text, acousticMs + beamMs, seconds, engine, em, true, beamMs,
                candidates.ifEmpty { Candidates.single(text) })
        }
        _last.value = result
        return result
    }

    /** The engine's name for the screen. */
    @Suppress("UNUSED_PARAMETER")
    fun label(ctx: Context): String = when (engine) {
        Engine.OMNI -> "Omnilingual"
        Engine.CTC -> "wav2vec2"
    }
}
