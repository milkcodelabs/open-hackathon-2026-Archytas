package com.openhackathon.voicetotext

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.models.ModelDownloader
import com.openhackathon.voicetotext.service.OverlayService
import com.openhackathon.voicetotext.service.TypingAccessibilityService
import com.openhackathon.voicetotext.ui.MainActions
import com.openhackathon.voicetotext.ui.MainScreen
import com.openhackathon.voicetotext.ui.ScreenState
import com.openhackathon.voicetotext.ui.theme.VoicetotextTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Setup screen. The interface people actually use is the floating bubble
 * ([OverlayService]); this walks through the permissions, gets the models onto the phone,
 * shows or hides the bubble, and lets the microphone be tried inside the app.
 */
class MainActivity : ComponentActivity() {

    /** In-app recording, so the phone can be tested without the floating bubble. */
    private val recorder = AudioRecorder()

    private var recording by mutableStateOf(false)
    private var busy by mutableStateOf(false)
    private var hint by mutableStateOf<String?>(null)
    private var result by mutableStateOf("")
    private var detail by mutableStateOf("")
    /** Bumped whenever something outside Compose may have changed (permissions, files). */
    private var tick by mutableIntStateOf(0)

    private val askPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { tick++ }

    /**
     * Copies model files the user picks into the app's own directory: the fallback when the
     * phone cannot download them, and the way to add wav2vec2 or my_words.txt.
     */
    private val importFiles =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            lifecycleScope.launch {
                busy = true
                val done = withContext(Dispatchers.IO) { uris.mapNotNull { copyIn(it) } }
                busy = false
                Recognizer.releaseUnused()
                if (done.any { it in LAYER2_FILES }) withContext(Dispatchers.Default) { Recognizer.reloadLayer2(this@MainActivity) }
                result = if (done.isEmpty()) "Δεν αντιγράφηκε τίποτα" else "Εισήχθησαν: " + done.joinToString(", ")
                tick++
            }
        }

    /** Files whose import changes layer 2 (language model, spellings, the speaker's words). */
    private val LAYER2_FILES = setOf("el_3gram.gvtlm", "el_homophones.bin", "my_words.txt")

    /** Every model file sits at the top level of the app's files folder. */
    private fun copyIn(uri: Uri): String? {
        val name = contentResolver.query(uri, null, null, null, null)?.use { c ->
            val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (c.moveToFirst() && i >= 0) c.getString(i) else null
        } ?: return null
        val dest = File(Recognizer.filesRoot(this), name)
        return runCatching {
            contentResolver.openInputStream(uri)!!.use { input ->
                dest.outputStream().use { out -> input.copyTo(out, 1 shl 20) }
            }
            name
        }.getOrNull()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        Recognizer.restoreEngine(this)

        // Models missing and on Wi-Fi: fetch them straight away. On mobile data, wait for
        // the button, since it is almost half a gigabyte.
        if (!ModelDownloader.complete(this) && ModelDownloader.onUnmeteredNetwork(this)) startDownload()

        setContent {
            VoicetotextTheme {
                MainScreen(state = screenState(), actions = actions)
            }
        }
    }

    override fun onResume() { super.onResume(); tick++ }

    override fun onPause() {
        if (recording) { runCatching { recorder.stop() }; recording = false }
        super.onPause()
    }

    // ------------------------------------------------------------------ state

    private fun micGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    private fun screenState(): ScreenState {
        if (tick < 0) error("unreachable")     // reading tick makes a bump recompose the screen
        val files = Recognizer.filesPresent(this)
        return ScreenState(
            mic = micGranted(),
            overlay = Settings.canDrawOverlays(this),
            accessibility = TypingAccessibilityService.enabled,
            filesPresent = files,
            modelLabel = Recognizer.label(this),
            modelMb = if (files) Recognizer.sizeMb(this) else 0,
            engine = Recognizer.engine,
            lmPresent = Recognizer.lmPresent(this),
            lmOn = Recognizer.useLanguageModel,
            lmMb = Recognizer.lmSizeMb(this),
            neuralPresent = Recognizer.neuralPresent(this),
            neuralOn = Recognizer.useNeuralLm,
            neuralMb = Recognizer.neuralSizeMb(this),
            bubbleOn = OverlayService.running,
            recording = recording,
            busy = busy,
            hint = hint,
            testWav = Recognizer.testWav(this).exists(),
            speaker = Recognizer.speakerId,
            result = result,
            detail = detail,
            modelsComplete = ModelDownloader.complete(this),
            missingMb = ModelDownloader.missingBytes(this) / 1_000_000,
            totalMb = ModelDownloader.totalBytes / 1_000_000,
            metered = !ModelDownloader.onUnmeteredNetwork(this),
        )
    }

    private val actions = object : MainActions {
        override fun askMic() {
            val perms = mutableListOf(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33) perms += Manifest.permission.POST_NOTIFICATIONS
            askPermissions.launch(perms.toTypedArray())
        }

        override fun openOverlaySettings() {
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
        }

        override fun openAccessibilitySettings() {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }

        override fun toggleBubble() {
            val i = Intent(this@MainActivity, OverlayService::class.java)
            if (OverlayService.running) startService(i.setAction(OverlayService.ACTION_STOP))
            else ContextCompat.startForegroundService(this@MainActivity, i)
            lifecycleScope.launch { delay(600); tick++ }
        }

        override fun toggleRecording() = this@MainActivity.toggleRecording()

        override fun importFiles() = importFiles.launch(arrayOf("*/*"))

        override fun selectEngine(e: Recognizer.Engine) {
            if (e == Recognizer.engine) return
            Recognizer.setEngine(this@MainActivity, e)
            Recognizer.releaseUnused()
            reloadBubble()
            tick++
        }

        override fun setLanguageModel(on: Boolean) {
            if (on == Recognizer.useLanguageModel) return
            Recognizer.setUseLanguageModel(this@MainActivity, on)
            Recognizer.releaseUnused()
            reloadBubble()
            tick++
        }

        override fun setNeuralLm(on: Boolean) {
            if (on == Recognizer.useNeuralLm) return
            lifecycleScope.launch {
                withContext(Dispatchers.Default) { Recognizer.setUseNeuralLm(this@MainActivity, on) }
                tick++
            }
        }

        override fun runCheck() = this@MainActivity.runCheck()

        override fun download() = startDownload()

        override fun cancelDownload() { ModelDownloader.cancel(); tick++ }
    }

    private fun reloadBubble() {
        if (OverlayService.running) {
            startService(Intent(this, OverlayService::class.java).setAction(OverlayService.ACTION_RELOAD))
        }
    }

    private fun startDownload() {
        ModelDownloader.start(this) { ok ->
            lifecycleScope.launch {
                if (ok) {
                    Recognizer.releaseUnused()
                    withContext(Dispatchers.Default) { Recognizer.reloadLayer2(this@MainActivity) }
                    reloadBubble()
                }
                tick++
            }
        }
    }

    // ------------------------------------------------------------------ record in the app

    private fun toggleRecording() {
        if (busy) return
        if (!recording) {
            if (OverlayService.running) {
                hint = "Το αιωρούμενο εικονίδιο κρατάει το μικρόφωνο. Κρύψ' το πρώτα."
                return
            }
            val started = runCatching { recorder.start() }
                .onFailure { hint = "Το μικρόφωνο δεν άνοιξε: $it" }
                .isSuccess
            if (!started) return
            hint = null
            recording = true
            result = ""
            detail = ""
        } else {
            recording = false
            val wav = runCatching { recorder.stop() }.getOrNull()
            if (wav == null || wav.size < AudioRecorder.SAMPLE_RATE / 5) {
                result = "Πολύ σύντομο"
                return
            }
            transcribe(wav)
        }
    }

    private fun transcribe(wav: FloatArray) = lifecycleScope.launch {
        busy = true
        val ok = withContext(Dispatchers.Default) { Recognizer.load(this@MainActivity) }
        if (!ok) {
            busy = false
            result = Recognizer.lastError ?: "Αποτυχία φόρτωσης"
            return@launch
        }
        val res = runCatching { withContext(Dispatchers.Default) { Recognizer.recognize(wav, "panel") } }
        busy = false
        res.onSuccess { show(it) }.onFailure { result = "Σφάλμα: $it" }
    }

    /**
     * One place that renders a result, used by both the microphone and the fixture check.
     * The sentences themselves are shown by the candidates list, which follows
     * [Recognizer.lastResult], the stages by the analysis card; here only a note if nothing
     * was heard.
     */
    private fun show(res: Recognizer.Result) {
        result = if (res.text.isBlank()) "(δεν αναγνωρίστηκε τίποτα)" else ""
        detail = ""       // the analysis card at the bottom shows every stage and its timing
    }

    // ------------------------------------------------------------------ on-device check

    private fun runCheck() {
        lifecycleScope.launch {
            result = "..."
            detail = ""
            val ok = withContext(Dispatchers.Default) { Recognizer.load(this@MainActivity) }
            if (!ok) { result = Recognizer.lastError ?: "Αποτυχία φόρτωσης"; return@launch }
            val wav = AudioRecorder.readWav(Recognizer.testWav(this@MainActivity))
            val res = withContext(Dispatchers.Default) { Recognizer.recognize(wav, "test.wav") }
            show(res)
        }
    }
}
