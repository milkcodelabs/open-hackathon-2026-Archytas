package com.openhackathon.voicetotext.models

import android.content.Context
import android.net.ConnectivityManager
import android.util.Log
import com.openhackathon.voicetotext.asr.Recognizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Fetches the model files from the project's GitHub Release when they are not on the phone.
 *
 * The models are far too large for the APK (the acoustic model alone is 365 MB), so the app
 * ships without them and downloads them once into the same folder the recognizer reads from
 * ([Recognizer.filesRoot]). Each file is written to `<name>.part`, resumed with an HTTP Range
 * request if the connection drops, checked against its size and SHA-256, and only then
 * renamed into place, so a half-downloaded file is never loaded as a model.
 *
 * Runs in a process-wide scope: turning the screen or leaving the app does not cancel it.
 */
object ModelDownloader {
    private const val TAG = "ModelDownloader"

    /** The release that holds the models. The repository must be public for this to work. */
    const val RELEASE_TAG = "models-v1"
    const val BASE_URL =
        "https://github.com/milkcodelabs/open-hackathon-2026-Archytas/releases/download/$RELEASE_TAG/"

    /**
     * [group]: files that only work together (a model and its labels). If any file of a group
     * is wrong, the whole group is fetched again, so a model is never paired with the labels of
     * another one. Files in neither list are never touched.
     */
    class ModelFile(val name: String, val bytes: Long, val sha256: String, val what: String, val group: String = name)

    /** What the default engine (Omnilingual + layer 2) needs. wav2vec2 stays optional (import). */
    val FILES = listOf(
        ModelFile("omni.onnx", 365_353_615L,
            "6ae7949b48ecc0658970a473e4c1526544b206464aa8682e8a5e24f04ac66798", "ακουστικό μοντέλο", "omni"),
        ModelFile("omni.labels.json", 235L,
            "97cc3eb93df00bc5c144c09d29432d441b6b575e6cf865e5625cefbccdb823e3", "ετικέτες", "omni"),
        ModelFile("el_3gram.gvtlm", 94_892_390L,
            "b9c45410d1a2215d0997862fab7a2c1aef5dd01e6dc1f1bb66a14e6c719c1426", "γλωσσικό μοντέλο"),
        ModelFile("el_homophones.bin", 10_498_299L,
            "9d6b6da6c8b86c671853a96f97b68e7c8795e1bec6e976825ce08c27404c2159", "ορθογραφία"),
        ModelFile("el_gpt2.int8.onnx", 163_794_538L,
            "02d09b774689a94792e6e8771fd013ce47cf8cf19b45873541686f10f6f47019", "νευρωνικό γλωσσικό μοντέλο", "gpt2"),
        ModelFile("el_gpt2.vocab.json", 1_719_026L,
            "6aac9ba2ea2bec874ed39cbf1c511466cb3e8b1ed2e1105ed100c99bf5242a2b", "λεξιλόγιο νευρωνικού", "gpt2"),
        ModelFile("el_gpt2.merges.txt", 1_377_379L,
            "79b67fa9426a6d70838a23304cea1b5acb0d700e4acf4af9190015a1ca9b2568", "λεξιλόγιο νευρωνικού", "gpt2"),
        ModelFile("test.wav", 311_098L,
            "f8c48eadf3625ddf46fedc0521ce688902dcc0f3bd3b821c8581070cc0688afc", "δοκιμαστικός ήχος"),
    )

    val totalBytes: Long = FILES.sumOf { it.bytes }

    /**
     * The speaker's own fine-tuned acoustic model (Omnilingual CTC 300M v2 adapted on their
     * recordings, M7 run r5) with its labels (a different letter order from the base model's)
     * and a short description for the screen. Fetched only when the user asks for it, never at
     * start-up; Recognizer uses it instead of omni.onnx while the "Προσωπικό μοντέλο" switch is on.
     */
    val PERSONAL = listOf(
        ModelFile("omni.personal.onnx", 356_199_242L,
            "83acf5aa777f83b5f78b535980f58a4185cfde2c1690799173ce3d32c12e3757", "προσωπικό ακουστικό μοντέλο", "personal"),
        ModelFile("omni.personal.labels.json", 235L,
            "0098d2e62f383f6cbb9a7450669ce22831b7c32204ab98b96059e52f4dc5d07c", "ετικέτες προσωπικού", "personal"),
        ModelFile("omni.personal.json", 176L,
            "8ace0f8c832942e28ccefd455b80c529bce75382824f5cedd846ac48ff37968f", "περιγραφή προσωπικού", "personal"),
    )

    val personalBytes: Long = PERSONAL.sumOf { it.bytes }

    /** A snapshot for the progress bar. [bytesDone] / [bytesTotal] cover only what was missing. */
    data class Progress(
        val running: Boolean = false,
        val file: String? = null,
        val bytesDone: Long = 0,
        val bytesTotal: Long = 0,
        val bytesPerSecond: Double = 0.0,
        val verifying: Boolean = false,
        val finished: Boolean = false,
        val error: String? = null,
    ) {
        val fraction: Float get() = if (bytesTotal > 0) (bytesDone.toDouble() / bytesTotal).toFloat().coerceIn(0f, 1f) else 0f
        /** Seconds left at the current speed, or null while the speed is not known yet. */
        val secondsLeft: Long? get() =
            if (bytesPerSecond > 1.0) ((bytesTotal - bytesDone) / bytesPerSecond).toLong() else null
    }

    private val _progress = MutableStateFlow(Progress())
    val progress: StateFlow<Progress> = _progress.asStateFlow()

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile private var job: Job? = null

    private fun target(ctx: Context, f: ModelFile) = File(Recognizer.filesRoot(ctx), f.name)
    private fun partial(ctx: Context, f: ModelFile) = File(Recognizer.filesRoot(ctx), f.name + ".part")

    /**
     * A file counts as present when its size matches (the hash was checked when it arrived).
     * A wrong file makes its whole group missing, so a model and its labels are replaced together.
     */
    fun missing(ctx: Context): List<ModelFile> = missingOf(ctx, FILES)

    /** The personal model's files that are absent or wrong (the whole group if any is). */
    fun missingPersonal(ctx: Context): List<ModelFile> = missingOf(ctx, PERSONAL)

    private fun missingOf(ctx: Context, files: List<ModelFile>): List<ModelFile> {
        val bad = files.filter { target(ctx, it).length() != it.bytes }.map { it.group }.toSet()
        return files.filter { it.group in bad }
    }

    fun complete(ctx: Context): Boolean = missing(ctx).isEmpty()

    fun missingBytes(ctx: Context): Long = missing(ctx).sumOf { it.bytes }

    /** Wi-Fi or another unmetered network: safe to start ~470 MB without asking. */
    fun onUnmeteredNetwork(ctx: Context): Boolean {
        val cm = ctx.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        return cm.activeNetwork != null && !cm.isActiveNetworkMetered
    }

    /** Starts downloading whatever is missing. Does nothing if a download is already running. */
    fun start(ctx: Context, onDone: (Boolean) -> Unit = {}) = run(ctx, { missing(it) }, onDone)

    /** Downloads the personal model (on request only). */
    fun startPersonal(ctx: Context, onDone: (Boolean) -> Unit = {}) = run(ctx, { missingPersonal(it) }, onDone)

    private fun run(ctx: Context, which: (Context) -> List<ModelFile>, onDone: (Boolean) -> Unit) {
        if (job?.isActive == true) return
        val app = ctx.applicationContext
        job = scope.launch {
            val ok = runCatching { downloadMissing(app, which(app)) }
                .onFailure { e ->
                    Log.e(TAG, "download failed", e)
                    _progress.value = _progress.value.copy(running = false, verifying = false,
                        error = e.message ?: e.toString())
                }.isSuccess
            onDone(ok && which(app).isEmpty())
        }
    }

    fun cancel() {
        job?.cancel()
        _progress.value = _progress.value.copy(running = false, verifying = false, error = "Ακυρώθηκε")
    }

    private suspend fun downloadMissing(ctx: Context, todo: List<ModelFile>) {
        val alreadyHave = todo.sumOf { partial(ctx, it).length().coerceAtMost(it.bytes) }
        val total = todo.sumOf { it.bytes }
        var done = alreadyHave
        _progress.value = Progress(running = true, bytesDone = done, bytesTotal = total)
        Recognizer.filesRoot(ctx).mkdirs()

        var speed = 0.0
        for (f in todo) {
            val part = partial(ctx, f)
            if (part.length() > f.bytes) part.delete()
            var have = part.length()
            if (have < f.bytes) {
                val conn = (URL(BASE_URL + f.name).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 20_000
                    readTimeout = 30_000
                    instanceFollowRedirects = true        // GitHub redirects to its file CDN
                    if (have > 0) setRequestProperty("Range", "bytes=$have-")
                }
                try {
                    val code = conn.responseCode
                    if (code == 200 && have > 0) {        // the server ignored Range: start over
                        done -= have
                        have = 0
                        part.delete()
                    } else if (code != 200 && code != 206) {
                        throw IllegalStateException("${f.name}: HTTP $code από ${BASE_URL.substringBefore("/releases")}")
                    }
                    conn.inputStream.use { input ->
                        FileOutputStream(part, have > 0).use { out ->
                            val buf = ByteArray(256 * 1024)
                            var lastT = System.nanoTime()
                            var lastBytes = done
                            while (true) {
                                currentCoroutineContext().ensureActive()   // cancel() stops here
                                val n = input.read(buf)
                                if (n < 0) break
                                out.write(buf, 0, n)
                                done += n
                                val now = System.nanoTime()
                                if (now - lastT > 250_000_000L) {
                                    val inst = (done - lastBytes) / ((now - lastT) / 1e9)
                                    speed = if (speed == 0.0) inst else 0.8 * speed + 0.2 * inst
                                    lastT = now
                                    lastBytes = done
                                    _progress.value = Progress(running = true, file = f.name, bytesDone = done,
                                        bytesTotal = total, bytesPerSecond = speed)
                                }
                            }
                        }
                    }
                } finally {
                    conn.disconnect()
                }
            }
            _progress.value = _progress.value.copy(file = f.name, verifying = true)
            if (part.length() != f.bytes || sha256(part) != f.sha256) {
                part.delete()
                throw IllegalStateException("${f.name}: το αρχείο δεν ταιριάζει με το αναμενόμενο, ξαναδοκίμασε")
            }
            val dst = target(ctx, f)
            dst.delete()
            if (!part.renameTo(dst)) throw IllegalStateException("${f.name}: δεν αποθηκεύτηκε")
            Log.i(TAG, "${f.name} ready (${f.bytes} bytes)")
        }
        _progress.value = Progress(running = false, bytesDone = total, bytesTotal = total, finished = true)
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
