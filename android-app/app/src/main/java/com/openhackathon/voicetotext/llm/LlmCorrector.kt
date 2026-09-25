package com.openhackathon.voicetotext.llm

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import com.openhackathon.voicetotext.BuildConfig
import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.decoding.WordChoices
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.abs
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject

/**
 * Optional layer 3: a fast hosted LLM reads what the phone recognized and returns the
 * sentence that makes sense.
 *
 * Only text leaves the phone, and only when the switch is on and there is a network: the
 * phonetic output of layer 1, the best sentence and, when the recognizer was unsure, the
 * runner-up. Never the audio, never the text already in the field. Any failure (no key,
 * timeout, quota, a reply that is not a plausible sentence) falls back to the phone's own
 * result, so the switch can never make dictation stop working.
 *
 * Both providers speak the OpenAI chat-completions protocol, so one client serves both.
 */
object LlmCorrector {
    private const val TAG = "LlmCorrector"
    private const val PREFS = "gvt"
    private const val KEY_ON = "use_llm"
    private const val KEY_PROVIDER = "llm_provider"

    private const val CONNECT_TIMEOUT_MS = 2_500
    private const val READ_TIMEOUT_MS = 4_500

    enum class Provider(
        val label: String,
        val url: String,
        val model: String,
        /** OpenAI `reasoning_effort`; the lowest each model accepts, for latency. */
        val reasoning: String?,
    ) {
        GEMINI(
            "Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "gemini-3.5-flash-lite", "minimal",
        ),
        GROQ(
            "Groq", "https://api.groq.com/openai/v1/chat/completions",
            "openai/gpt-oss-120b", "low",
        ),
    }

    @Volatile var enabled: Boolean = false
        private set
    @Volatile var provider: Provider = Provider.GEMINI
        private set

    /** One call's outcome, for the panel. [text] is null when the phone's result was kept. */
    class Outcome(
        val result: Recognizer.Result,
        val provider: Provider,
        val sent: String,
        val text: String?,
        val words: List<String>,
        val ms: Long,
        val error: String?,
    )

    private val _last = MutableStateFlow<Outcome?>(null)
    val last: StateFlow<Outcome?> = _last.asStateFlow()

    fun key(p: Provider): String = when (p) {
        Provider.GEMINI -> BuildConfig.GEMINI_API_KEY
        Provider.GROQ -> BuildConfig.GROQ_API_KEY
    }

    fun hasKey(p: Provider): Boolean = key(p).isNotBlank()

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun restore(ctx: Context) {
        val p = prefs(ctx)
        enabled = p.getBoolean(KEY_ON, false)
        provider = runCatching { Provider.valueOf(p.getString(KEY_PROVIDER, null)!!) }
            .getOrElse { Provider.entries.firstOrNull { hasKey(it) } ?: Provider.GEMINI }
    }

    fun setEnabled(ctx: Context, on: Boolean) {
        enabled = on
        prefs(ctx).edit().putBoolean(KEY_ON, on).apply()
    }

    fun setProvider(ctx: Context, p: Provider) {
        provider = p
        prefs(ctx).edit().putString(KEY_PROVIDER, p.name).apply()
    }

    /** A network that has actually reached the internet, not just a Wi-Fi with a login page. */
    fun online(ctx: Context): Boolean {
        val cm = ctx.getSystemService(ConnectivityManager::class.java) ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    fun shouldRun(ctx: Context): Boolean = enabled && hasKey(provider) && online(ctx)

    /**
     * Blocking; call it off the main thread. The outcome's [Outcome.words] is the sentence to
     * type, in the recognizer's normalized words; empty means keep the phone's own result.
     */
    fun refine(res: Recognizer.Result): Outcome {
        val p = provider
        val best = res.candidates.firstOrNull()
        val t0 = System.nanoTime()
        fun done(sent: String, text: String?, words: List<String>, error: String?) =
            Outcome(res, p, sent, text, words, (System.nanoTime() - t0) / 1_000_000, error)
                .also { _last.value = it }

        if (best == null) return done("", null, emptyList(), "κενό αποτέλεσμα")
        val phonetic = res.stages.firstOrNull()?.text ?: res.emissions.greedyDecode()
        val unsure = WordChoices.find(best.words, res.candidates).isNotEmpty()
        val second = if (unsure) res.candidates.getOrNull(1)?.text else null
        val prompt = buildString {
            append("PHONETIC: ").append(phonetic).append('\n')
            append("BEST: ").append(best.text)
            if (second != null) append('\n').append("SECOND: ").append(second)
        }

        val out = try {
            val reply = complete(p, prompt)
            val line = reply.lines().map { it.trim() }.firstOrNull { it.isNotEmpty() }.orEmpty()
                .removePrefix("BEST:").trim().trim('"', '«', '»', '“', '”')
            val words = Recognizer.normalizeWords(line, res.emissions.labels)
            val size = best.words.size
            when {
                words.isEmpty() -> done(prompt, null, emptyList(), "κενή απάντηση")
                abs(words.size - size) > maxOf(3, size / 2) -> done(prompt, null, emptyList(), "άσχετη απάντηση: $line")
                else -> done(prompt, words.joinToString(" "), words, null)
            }
        } catch (e: Exception) {
            Log.w(TAG, "${p.label} failed, keeping the phone's result", e)
            done(prompt, null, emptyList(), e.message ?: e.toString())
        }
        Log.i(TAG, "${p.label} ${out.ms} ms: ${best.text} -> ${out.text ?: "(kept) ${out.error}"}")
        return out
    }

    private fun complete(p: Provider, prompt: String): String {
        return try {
            post(p, prompt, p.reasoning)
        } catch (e: HttpError) {
            // a model that does not take this reasoning level rejects the whole request
            if (e.code == 400 && p.reasoning != null) post(p, prompt, null) else throw e
        }
    }

    private class HttpError(val code: Int, message: String) : IOException("HTTP $code: $message")

    private fun post(p: Provider, prompt: String, reasoning: String?): String {
        val body = JSONObject().apply {
            put("model", p.model)
            put("max_tokens", 1024)          // includes reasoning tokens on some providers
            if (reasoning != null) put("reasoning_effort", reasoning)
            put("messages", JSONArray()
                .put(JSONObject().put("role", "system").put("content", SYSTEM_PROMPT))
                .put(JSONObject().put("role", "user").put("content", prompt)))
        }
        val conn = (URL(p.url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Authorization", "Bearer ${key(p)}")
        }
        try {
            conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) throw HttpError(code, errorMessage(text))
            return JSONObject(text).getJSONArray("choices").getJSONObject(0)
                .getJSONObject("message").optString("content")
        } finally {
            conn.disconnect()
        }
    }

    /** Gemini wraps its error in a list, OpenAI-style APIs do not. */
    private fun errorMessage(body: String): String {
        val obj = runCatching { JSONObject(body) }.getOrNull()
            ?: runCatching { JSONArray(body).getJSONObject(0) }.getOrNull()
        return obj?.optJSONObject("error")?.optString("message")?.takeIf { it.isNotBlank() }
            ?: body.take(200)
    }

    private val SYSTEM_PROMPT = """
        You correct the output of an offline Greek speech recognizer. People with motor
        disabilities use it to dictate text, and their speech can be slow or unclear.

        You receive up to three lines:
        PHONETIC: the letters the acoustic model heard, with no dictionary. The sounds are
        mostly right; the spelling, accents and word boundaries often are not.
        BEST: the recognizer's best sentence after its Greek language model.
        SECOND: its runner-up, sent only when it was unsure.

        Write the Greek sentence the speaker most likely said.
        - Start from BEST. Change a word only when the sentence does not make sense or is
          ungrammatical, and prefer words from SECOND or words that sound like PHONETIC.
        - Letters that sound the same in Greek are interchangeable in PHONETIC: ι η υ ει οι,
          ο ω, ε αι, single and double consonants.
        - Keep the meaning and about the same number of words. Never add content, answer,
          translate or explain.
        - Correct accents (τόνοι), lowercase, no punctuation.
        Reply with the sentence only.
    """.trimIndent()
}
