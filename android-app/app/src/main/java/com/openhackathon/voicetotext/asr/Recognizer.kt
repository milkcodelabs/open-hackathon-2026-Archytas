package com.openhackathon.voicetotext.asr

import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.decoding.BeamSearch
import com.openhackathon.voicetotext.decoding.Candidate
import com.openhackathon.voicetotext.decoding.Candidates
import com.openhackathon.voicetotext.decoding.HomophoneIndex
import com.openhackathon.voicetotext.decoding.NeuralRescorer
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
 * Process-wide owner of the acoustic model, shared by the bubble and the control panel.
 *
 * Layer 1 is Meta Omnilingual CTC 300M, restricted to the Greek columns inside the graph
 * (WER 0.139 on FLEURS with the LM), or a speaker's own fine-tune of it (omni.personal[.<name>].*;
 * several can sit on the phone, one is chosen).
 */
object Recognizer {
    private const val TAG = "Recognizer"
    private const val PREFS = "settings"
    private const val KEY_LM = "use_lm"
    private const val KEY_NEURAL = "use_neural_lm"
    /** Before several personal models: a switch for omni.personal.*. Read once, to migrate. */
    private const val KEY_PERSONAL_OLD = "use_personal_model"
    private const val KEY_PERSONAL = "personal_model"
    private const val DEFAULT_PERSONAL = "omni.personal"

    @Volatile private var ctc: CtcModel? = null
    /** Which personal model [ctc] holds ("" = the general one) and its name for the screen. */
    @Volatile private var loadedKey: String = ""
    @Volatile private var loadedTitle: String? = null
    @Volatile private var lm: NgramLm? = null
    @Volatile private var beam: BeamSearch? = null
    /** Layer 2b; null when el_homophones.bin is missing (then 2a alone runs). */
    @Volatile private var homophones: HomophoneIndex? = null
    @Volatile private var speller: SpellingRescorer? = null
    /** Layer 2c; null when switched off or its files are missing. */
    @Volatile private var neural: NeuralRescorer? = null
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

    /** Layer 2. Off means bare greedy decoding of layer 1. */
    @Volatile var useLanguageModel: Boolean = true
        private set
    /** Layer 2c, the neural LM that re-ranks the best sentences with context. */
    @Volatile var useNeuralLm: Boolean = true
        private set
    /** The chosen personal model's key (its file stem, e.g. omni.personal.nikol); "" = general. */
    @Volatile var personalKey: String = DEFAULT_PERSONAL
        private set
    @Volatile var speakerId: String = "default"
    @Volatile var lastError: String? = null

    private fun prefs(ctx: Context): SharedPreferences =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun restoreSettings(ctx: Context) {
        val p = prefs(ctx)
        renameOldFiles(ctx)
        useLanguageModel = p.getBoolean(KEY_LM, true)
        useNeuralLm = p.getBoolean(KEY_NEURAL, true)
        personalKey = p.getString(KEY_PERSONAL, null)
            ?: if (p.getBoolean(KEY_PERSONAL_OLD, true)) DEFAULT_PERSONAL else ""
    }

    // ------------------------------------------------------------------ personal models
    //
    // A speaker's fine-tuned Omnilingual lives next to the downloaded one under its own names:
    // omni.personal.onnx (the one ModelDownloader can fetch) and any omni.personal.<name>.onnx
    // copied onto the phone, each with <stem>.labels.json and optionally <stem>.json. A model
    // counts only when BOTH its graph and its labels are present (its labels may list the
    // letters in a different order than the base model's). The screen offers every one found.

    /** One personal model on the phone. [key] is its file stem. */
    class PersonalModel(val key: String, val model: File, val labels: File, val meta: File) {
        private val json by lazy { runCatching { org.json.JSONObject(meta.readText()) }.getOrNull() }
        /** Short name for a button: the "speaker" of <stem>.json up to " (", else the file name. */
        val title: String get() = json?.optString("speaker")?.substringBefore(" (")?.takeIf { it.isNotBlank() }
            ?: key.removePrefix(DEFAULT_PERSONAL).removePrefix(".").ifEmpty { "προσωπικό" }
        val sizeMb: Long get() = model.length() / 1_000_000
        /** One line for the screen from <stem>.json: speaker, base, trained, note. */
        val info: String get() = json?.let { j ->
            listOf("speaker", "base", "trained", "note").mapNotNull { k -> j.optString(k).takeIf { it.isNotBlank() } }
                .joinToString("  ·  ")
        } ?: ""
    }

    private val PERSONAL_FILE = Regex("""omni\.personal(\.[A-Za-z0-9_-]+)?\.onnx""")

    /** Every personal model on the phone, the downloadable omni.personal.* first. */
    fun personalModels(ctx: Context): List<PersonalModel> {
        val root = filesRoot(ctx)
        return (root.listFiles() ?: emptyArray()).mapNotNull { f ->
            val m = PERSONAL_FILE.matchEntire(f.name) ?: return@mapNotNull null
            val stem = DEFAULT_PERSONAL + m.groupValues[1]
            val labels = File(root, "$stem.labels.json")
            if (labels.exists()) PersonalModel(stem, f, labels, File(root, "$stem.json")) else null
        }.sortedBy { if (it.key == DEFAULT_PERSONAL) "" else it.key }
    }

    /** The chosen personal model, or null for the general one (also when the chosen one is gone). */
    fun selectedPersonal(ctx: Context): PersonalModel? =
        if (personalKey.isEmpty()) null else personalModels(ctx).find { it.key == personalKey }

    /** [key]: a [PersonalModel.key], or "" for the general model. */
    fun setPersonal(ctx: Context, key: String) {
        personalKey = key
        prefs(ctx).edit().putString(KEY_PERSONAL, key).apply()
        dropAcoustic()
        Log.i(TAG, "personal model = ${key.ifEmpty { "none" }}")
    }

    /** Forget the loaded acoustic model; the next recognition loads the right one. */
    @Synchronized
    fun dropAcoustic() {
        ctc?.close(); ctc = null; loadedKey = ""; loadedTitle = null; beam = null; speller = null
    }

    fun neuralPresent(ctx: Context): Boolean = NeuralRescorer.present(filesRoot(ctx))
    fun neuralSizeMb(ctx: Context): Long = NeuralRescorer.files(filesRoot(ctx)).sumOf { it.length() } / 1_000_000

    fun setUseNeuralLm(ctx: Context, on: Boolean) {
        useNeuralLm = on
        prefs(ctx).edit().putBoolean(KEY_NEURAL, on).apply()
        if (!on) { neural?.close(); neural = null } else loadNeural(ctx)
        Log.i(TAG, "neural LM = $on")
    }

    /** Loads layer 2c once; it does not depend on the acoustic engine. */
    @Synchronized
    private fun loadNeural(ctx: Context) {
        if (neural != null || !useNeuralLm || !neuralPresent(ctx)) return
        val (m, v, g) = NeuralRescorer.files(filesRoot(ctx))
        neural = runCatching { NeuralRescorer(m, v, g) }
            .onFailure { Log.e(TAG, "neural LM failed to load", it) }.getOrNull()
    }

    fun setUseLanguageModel(ctx: Context, on: Boolean) {
        useLanguageModel = on
        prefs(ctx).edit().putBoolean(KEY_LM, on).apply()
        if (!on) { beam = null; speller = null }
        val m = ctc
        if (on && m != null && lmPresent(ctx)) {
            val n = lm ?: NgramLm.load(lmFile(ctx)).also { lm = it }
            buildLayer2(ctx, m.labels, n)
        }
        Log.i(TAG, "language model = $on")
    }

    fun lmFile(ctx: Context): File = File(filesRoot(ctx), "el_3gram.ngram")
    fun lmPresent(ctx: Context): Boolean = lmFile(ctx).exists()
    fun lmSizeMb(ctx: Context): Long = lmFile(ctx).length() / 1_000_000
    fun homophonesFile(ctx: Context): File = File(filesRoot(ctx), "el_homophones.bin")
    /** One word or name per line; imported like the models. */
    fun myWordsFile(ctx: Context): File = File(filesRoot(ctx), "my_words.txt")

    /** The language model used to be called el_3gram.gvtlm; same bytes, so keep the copy. */
    private fun renameOldFiles(ctx: Context) {
        val old = File(filesRoot(ctx), "el_3gram.gvtlm")
        val now = lmFile(ctx)
        if (old.exists() && !now.exists() && old.renameTo(now)) Log.i(TAG, "renamed ${old.name} -> ${now.name}")
    }

    // ------------------------------------------------------------------ files

    fun filesRoot(ctx: Context): File = ctx.getExternalFilesDir(null) ?: ctx.filesDir
    fun omniFile(ctx: Context): File = File(filesRoot(ctx), "omni.onnx")
    fun omniLabelsFile(ctx: Context): File = File(filesRoot(ctx), "omni.labels.json")
    fun testWav(ctx: Context): File = File(filesRoot(ctx), "test.wav")

    /**
     * Dev-mode test recordings: <files>/samples/<name>.wav, and optionally <name>.txt with the
     * sentence that was meant, e.g. a speaker's sentence their personal model never heard.
     */
    fun samples(ctx: Context): List<File> =
        File(filesRoot(ctx), "samples").listFiles { f -> f.name.endsWith(".wav") }?.sortedBy { it.name } ?: emptyList()

    fun omniPresent(ctx: Context): Boolean = omniFile(ctx).exists() && omniLabelsFile(ctx).exists()

    /** Whether layer 1 can run: the downloaded model or the chosen personal one is there. */
    fun filesPresent(ctx: Context): Boolean = omniPresent(ctx) || selectedPersonal(ctx) != null

    fun sizeMb(ctx: Context): Long = (selectedPersonal(ctx)?.model ?: omniFile(ctx)).length() / 1_000_000

    val ready: Boolean get() = ctc != null

    /** Layer 2a (beam + LM + context + the speaker's words) and, if its index is there, 2b. */
    private fun buildLayer2(ctx: Context, labels: List<String>, n: NgramLm) {
        personalWords = readMyWords(ctx, labels)
        // alpha/beta tuned on FLEURS dev for Omnilingual
        val b = BeamSearch(labels, n, alpha = 0.7f, beta = 3.0f, personalWords = personalWords)
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
        if (!useLanguageModel || !lmPresent(ctx)) return
        lm = null
        buildLayer2(ctx, m.labels, NgramLm.load(lmFile(ctx)).also { lm = it })
    }

    private fun readMyWords(ctx: Context, labels: List<String>): List<String> {
        val f = myWordsFile(ctx)
        if (!f.exists()) return emptyList()
        return runCatching { f.readLines() }.getOrDefault(emptyList())
            .flatMap { normalizeWords(it, labels) }.distinct()
    }

    /**
     * Text -> the model's words: lowercase, NFC, every character outside the label set
     * becomes a space.
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
        val personal = selectedPersonal(ctx)
        if (ready && loadedKey == (personal?.key ?: "")) return true
        if (!filesPresent(ctx)) {
            lastError = "Λείπει το omni.onnx στο ${filesRoot(ctx).absolutePath}"
            return false
        }
        val t0 = System.currentTimeMillis()
        return try {
            ctc?.close(); ctc = null; beam = null; speller = null
            val m = if (personal != null) CtcModel(personal.model, personal.labels)
                    else CtcModel(omniFile(ctx), omniLabelsFile(ctx))
            ctc = m
            loadedKey = personal?.key ?: ""
            loadedTitle = personal?.title
            if (useLanguageModel && lmPresent(ctx)) {
                val n = lm ?: NgramLm.load(lmFile(ctx)).also { lm = it }
                buildLayer2(ctx, m.labels, n)
            }
            loadNeural(ctx)
            lastError = null
            Log.i(TAG, "${label(null)} ready in ${System.currentTimeMillis() - t0} ms")
            true
        } catch (e: Exception) {
            lastError = e.toString()
            Log.e(TAG, "load failed", e)
            false
        }
    }

    // ------------------------------------------------------------------ recognition

    class Result(
        val text: String,
        val inferenceMs: Long,
        val audioSeconds: Float,
        val emissions: Emissions,
        val usedLanguageModel: Boolean = false,
        val beamMs: Long = 0,
        /** Layer 2c's share of [beamMs]; 0 when it did not run. */
        val neuralMs: Long = 0,
        /** The best sentences, best first; [text] is the first. One entry without the LM. */
        val candidates: List<Candidate> = emptyList(),
        /** What each layer produced, in order, for the analysis on the screen. */
        val stages: List<Stage> = emptyList(),
    ) {
        val rtf: Float get() = inferenceMs / 1000f / audioSeconds
    }

    /** One step of the pipeline: its best sentence and the time it took. */
    class Stage(val layer: String, val what: String, val text: String, val ms: Long)

    /**
     * [context]: the text already in the field the result will be typed into (text before
     * the cursor), or null. Layer 2 continues it instead of starting a new sentence.
     */
    fun recognize(waveform: FloatArray, audioId: String = "", context: String? = null): Result {
        val seconds = waveform.size / AudioRecorder.SAMPLE_RATE.toFloat()
        val m = ctc ?: throw IllegalStateException("το μοντέλο δεν φορτώθηκε")
        val (em, acousticMs) = m.emit(waveform, audioId)
        val b = beam
        val layer1 = Stage(
            "Layer 1", "${label(null)}: το πιο πιθανό γράμμα κάθε 20 ms, χωρίς γλωσσικό μοντέλο",
            em.greedyDecode(), acousticMs,
        )
        val result = if (b == null) {
            val text = em.greedyDecode()
            Result(text, acousticMs, seconds, em, false, 0, 0, Candidates.single(text), listOf(layer1))
        } else {
            val stages = arrayListOf(layer1)
            val t0 = System.nanoTime()
            val history = contextWords(context, m.labels)
            var hyps = b.decode(em, N_BEST, history)
            val beamOnlyMs = (System.nanoTime() - t0) / 1_000_000
            stages.add(Stage("Layer 2α", "αναζήτηση + ελληνικό 3-gram" +
                (if (history.isEmpty()) "" else ", συνέχεια του «${history.joinToString(" ")}»") +
                (if (personalWords.isEmpty()) "" else ", ${personalWords.size} δικές σου λέξεις"),
                hyps.firstOrNull()?.text ?: "", beamOnlyMs))
            speller?.let { sp ->
                val ts = System.nanoTime()
                hyps = sp.rescore(hyps, history)
                stages.add(Stage("Layer 2β", "ορθογραφία: ομόηχες γραφές (ι/η/υ/ει/οι, ο/ω, ε/αι), διαλέγει το γλωσσικό μοντέλο",
                    hyps.firstOrNull()?.text ?: "", (System.nanoTime() - ts) / 1_000_000))
            }
            var neuralMs = 0L
            val nr = neural
            if (nr != null && hyps.size > 1) {
                val tn = System.nanoTime()
                // the sentence so far, in the model's words, is the neural LM's context
                val ctxText = normalizeWords(context ?: "", m.labels).takeLast(30).joinToString(" ")
                hyps = runCatching { nr.rerank(hyps, ctxText).take(nr.topK) }
                    .onFailure { Log.e(TAG, "neural rescoring failed, keeping layer 2b's order", it) }
                    .getOrDefault(hyps)
                neuralMs = (System.nanoTime() - tn) / 1_000_000
                stages.add(Stage("Layer 2γ", "νευρωνικό GPT-2: ξαναδιαλέγει ανάμεσα στις ${nr.topK} καλύτερες με βάση όλη τη φράση",
                    hyps.firstOrNull()?.text ?: "", neuralMs))
            }
            val candidates = Candidates.from(hyps, MAX_CANDIDATES)
            val text = candidates.firstOrNull()?.text ?: em.greedyDecode()
            val beamMs = (System.nanoTime() - t0) / 1_000_000
            Result(text, acousticMs + beamMs, seconds, em, true, beamMs, neuralMs,
                candidates.ifEmpty { Candidates.single(text) }, stages)
        }
        _last.value = result
        return result
    }

    /** The model's name for the screen; with no context, it describes the loaded model. */
    fun label(ctx: Context?): String =
        (if (ctx != null) selectedPersonal(ctx)?.title else loadedTitle)
            ?.let { "Omnilingual (προσωπικό: $it)" } ?: "Omnilingual"
}
