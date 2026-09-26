package com.openhackathon.voicetotext.ui

import android.content.Context
import android.media.MediaPlayer
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.openhackathon.voicetotext.R
import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.ui.theme.Bg
import com.openhackathon.voicetotext.ui.theme.Brand
import com.openhackathon.voicetotext.ui.theme.Listening
import com.openhackathon.voicetotext.ui.theme.Muted
import com.openhackathon.voicetotext.ui.theme.Ok
import com.openhackathon.voicetotext.ui.theme.OnBg
import com.openhackathon.voicetotext.ui.theme.OnMuted
import com.openhackathon.voicetotext.ui.theme.Surface
import com.openhackathon.voicetotext.ui.theme.SurfaceSoft
import java.io.File
import java.io.RandomAccessFile
import java.time.Instant
import java.util.Locale

/** The sentences of the initial training: every-day phrases and the sounds that are hard to say. */
val TRAINING_SENTENCES = listOf(
    "Καλημέρα, πώς μπορώ να σας βοηθήσω σήμερα;",
    "Θέλω να φάω χορτόπιτα με φρέσκια ντομάτα.",
    "Ο Βασίλης διαβάζει γρήγορα για τις γιορτές.",
    "Το τζάκι καπνίζει και ο σκύλος ψάχνει ξύλα.",
    "Στο κέντρο της πόλης μπήκαν τεράστιες ταμπέλες.",
    "Το μεγάλο ρολόι χτυπάει στις δώδεκα το μεσημέρι.",
    "Αυτός ο δρόμος είναι πολύ εύκολος και ευθύς.",
    "Ξαφνικά, η μικρή γάτα πήδηξε στο ψηλό δέντρο.",
    "Χθες το βράδυ είδαμε ένα ψάρι στα βαθιά νερά.",
    "Ναι, ίσως πάω αύριο το πρωί στη δουλειά μου.",
    "Η γρήγορη αλεπού ξέφυγε από τον άγριο λύκο.",
    "Προγραμματίζω τις διακοπές μου για τον επόμενο μήνα.",
    "Ο Γιώργος και η Γιάννα γελάνε δυνατά στον κήπο.",
    "Πότε ακριβώς περιμένεις το πακέτο από το ταχυδρομείο;",
    "Η Ζωή σχεδιάζει να ταξιδέψει με το πλοίο αύριο.",
)

private val Violet = Color(0xFF8B5CF6)

/**
 * The takes of the initial training, on the phone: <files>/training/sNN.wav (16 kHz mono
 * 16-bit) and metadata.csv in the recording page's format (model-pipeline/personalization),
 * so the folder can go straight into pack_bundle.py. Nothing is uploaded.
 */
object TrainingStore {
    fun dir(ctx: Context): File = File(Recognizer.filesRoot(ctx), "training").apply { mkdirs() }
    fun wav(ctx: Context, i: Int): File = File(dir(ctx), String.format(Locale.US, "s%02d.wav", i + 1))

    /** Seconds of each recorded take, by sentence index. */
    fun durations(ctx: Context): Map<Int, Float> = TRAINING_SENTENCES.indices.mapNotNull { i ->
        val f = wav(ctx, i)
        if (f.length() > 44) i to (f.length() - 44) / 2f / AudioRecorder.SAMPLE_RATE else null
    }.toMap()

    fun save(ctx: Context, i: Int, pcm: FloatArray) {
        writeWav(wav(ctx, i), pcm)
        writeMetadata(ctx)
    }

    private fun writeMetadata(ctx: Context) {
        val rows = durations(ctx).toSortedMap().map { (i, d) ->
            val f = wav(ctx, i)
            listOf("wav/${f.name}", TRAINING_SENTENCES[i], "", "train_pool", "initial", "1",
                String.format(Locale.US, "%.2f", d), "${AudioRecorder.SAMPLE_RATE}",
                Instant.ofEpochMilli(f.lastModified()).toString()).joinToString(",") { csv(it) }
        }
        File(dir(ctx), "metadata.csv").writeText(
            "﻿file,text,pronounced_as,set,category,session,duration_s,sample_rate,recorded_at\n" +
                rows.joinToString("\n") + "\n")
    }

    private fun csv(s: String) = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

    private fun writeWav(file: File, pcm: FloatArray) {
        val rate = AudioRecorder.SAMPLE_RATE
        val data = ByteArray(pcm.size * 2)
        for (k in pcm.indices) {
            val v = (pcm[k].coerceIn(-1f, 1f) * 32767).toInt()
            data[2 * k] = (v and 0xFF).toByte(); data[2 * k + 1] = (v shr 8 and 0xFF).toByte()
        }
        RandomAccessFile(file, "rw").use { f ->
            f.setLength(0)
            fun i32(x: Int) = f.write(byteArrayOf(x.toByte(), (x shr 8).toByte(), (x shr 16).toByte(), (x shr 24).toByte()))
            fun i16(x: Int) = f.write(byteArrayOf(x.toByte(), (x shr 8).toByte()))
            f.write("RIFF".toByteArray()); i32(36 + data.size); f.write("WAVEfmt ".toByteArray())
            i32(16); i16(1); i16(1); i32(rate); i32(rate * 2); i16(2); i16(16)
            f.write("data".toByteArray()); i32(data.size); f.write(data)
        }
    }
}

/**
 * Initial training: the speaker reads [TRAINING_SENTENCES] one by one. Each take is saved on
 * the phone at once; at the end the screen shows how the personal model would be made. In
 * this version nothing is sent anywhere: the takes stay on the phone.
 */
@Composable
fun TrainingScreen(micGranted: Boolean, micBusy: Boolean, askMic: () -> Unit, onBack: () -> Unit) {
    val ctx = LocalContext.current
    val recorder = remember { AudioRecorder() }
    val done = remember { mutableStateMapOf<Int, Float>().apply { putAll(TrainingStore.durations(ctx)) } }
    var idx by remember { mutableIntStateOf(TRAINING_SENTENCES.indices.firstOrNull { it !in done } ?: 0) }
    var recording by remember { mutableStateOf(false) }
    var note by remember { mutableStateOf<String?>(null) }
    var showPlan by remember { mutableStateOf(false) }
    var player by remember { mutableStateOf<MediaPlayer?>(null) }
    DisposableEffect(Unit) {
        onDispose { if (recording) runCatching { recorder.stop() }; player?.release() }
    }

    fun toggle() {
        if (!micGranted) { askMic(); return }
        if (!recording) {
            if (micBusy) { note = "Το αιωρούμενο κουμπί κρατάει το μικρόφωνο. Απενεργοποίησέ το πρώτα."; return }
            if (runCatching { recorder.start() }.isFailure) { note = "Το μικρόφωνο δεν άνοιξε."; return }
            player?.release(); player = null
            recording = true; note = null
        } else {
            recording = false
            val pcm = runCatching { recorder.stop() }.getOrNull()
            if (pcm == null || pcm.size < AudioRecorder.SAMPLE_RATE / 2) { note = "Πολύ σύντομο, ξαναδοκίμασε."; return }
            TrainingStore.save(ctx, idx, pcm)
            done[idx] = pcm.size / AudioRecorder.SAMPLE_RATE.toFloat()
            note = String.format(Locale("el"), "Αποθηκεύτηκε, %.1f δευτ.", done[idx])
            TRAINING_SENTENCES.indices.map { (idx + 1 + it) % TRAINING_SENTENCES.size }.firstOrNull { it !in done }?.let { idx = it }
        }
    }

    val total = TRAINING_SENTENCES.size
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(0f to Color(0xFF1A2146), 0.5f to Bg))
            .safeDrawingPadding()
            .padding(start = 18.dp, end = 18.dp, top = 8.dp, bottom = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            TextButton(onClick = onBack) { Text("‹ Πίσω", fontSize = 16.sp, color = OnMuted) }
            Spacer(Modifier.weight(1f))
            Text("${done.size} / $total", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = OnBg)
        }
        Text("Αρχική εκπαίδευση", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = OnBg, modifier = Modifier.padding(top = 4.dp))
        Text("Διάβασε κάθε πρόταση με τον δικό σου τρόπο.", fontSize = 15.sp, color = OnMuted, textAlign = TextAlign.Center)
        LinearProgressIndicator(
            progress = { done.size / total.toFloat() },
            modifier = Modifier.fillMaxWidth().padding(top = 14.dp).height(8.dp).clip(RoundedCornerShape(4.dp)),
            color = Ok, trackColor = SurfaceSoft,
        )

        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(22.dp))
                    .background(Surface)
                    .border(1.dp, Color.White.copy(alpha = 0.06f), RoundedCornerShape(22.dp))
                    .padding(horizontal = 20.dp, vertical = 24.dp),
            ) {
                Text("Πρόταση ${idx + 1}", fontSize = 14.sp, color = OnMuted)
                Text(TRAINING_SENTENCES[idx], fontSize = 25.sp, fontWeight = FontWeight.SemiBold, color = OnBg,
                    textAlign = TextAlign.Center, lineHeight = 32.sp, modifier = Modifier.padding(top = 10.dp))
                Text(
                    when {
                        recording -> "Ακούω…"
                        note != null -> note!!
                        idx in done -> String.format(Locale("el"), "Υπάρχει λήψη, %.1f δευτ. Ηχογράφησε ξανά για να την αλλάξεις.", done[idx])
                        else -> "Πάτα το κουμπί, διάβασε, πάτα ξανά."
                    },
                    fontSize = 14.sp, color = if (idx in done && !recording) Ok else OnMuted,
                    textAlign = TextAlign.Center, modifier = Modifier.padding(top = 14.dp),
                )
            }
        }

        RecordButton(recording) { toggle() }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth().padding(top = 14.dp)) {
            OutlinedButton(onClick = { idx--; note = null }, enabled = idx > 0 && !recording, modifier = Modifier.weight(1f).height(48.dp)) { Text("Προηγ.", fontSize = 15.sp) }
            OutlinedButton(
                onClick = {
                    player?.release()
                    player = MediaPlayer().apply { setDataSource(TrainingStore.wav(ctx, idx).path); prepare(); start() }
                },
                enabled = idx in done && !recording, modifier = Modifier.weight(1f).height(48.dp),
            ) { Text("Άκου", fontSize = 15.sp) }
            OutlinedButton(onClick = { idx++; note = null }, enabled = idx < total - 1 && !recording, modifier = Modifier.weight(1f).height(48.dp)) { Text("Επόμ.", fontSize = 15.sp) }
        }
        Spacer(Modifier.height(12.dp))
        val all = done.size == total
        Row(
            horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(if (all) Brush.horizontalGradient(listOf(Brand, Violet)) else Brush.horizontalGradient(listOf(Surface, Surface)))
                .clickable(enabled = all && !recording, role = Role.Button) { showPlan = true },
        ) {
            Text(if (all) "Έναρξη εκπαίδευσης" else "Έναρξη εκπαίδευσης (${total - done.size} ακόμα)",
                fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = if (all) OnBg else Muted)
        }
    }

    if (showPlan) {
        val minutes = done.values.sum() / 60f
        AlertDialog(
            onDismissRequest = { showPlan = false },
            confirmButton = { TextButton(onClick = { showPlan = false }) { Text("Εντάξει", fontSize = 16.sp) } },
            title = { Text("Πώς γίνεται η εκπαίδευση") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    PlanRow(true, String.format(Locale("el"), "$total ηχογραφήσεις στο κινητό (%.1f λεπτά)", minutes))
                    PlanRow(false, "Αποστολή σε GPU στο cloud")
                    PlanRow(false, "Προσαρμογή του μοντέλου στη φωνή σου (~10 λεπτά)")
                    PlanRow(false, "Το προσωπικό μοντέλο γυρίζει στο κινητό και δουλεύει χωρίς internet")
                    Text("Σε αυτή την έκδοση οι ηχογραφήσεις μένουν μόνο στο κινητό. Δεν στέλνονται πουθενά.",
                        fontSize = 14.sp, color = OnMuted)
                }
            },
            containerColor = Surface, titleContentColor = OnBg, textContentColor = OnBg,
        )
    }
}

@Composable
private fun PlanRow(ok: Boolean, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(24.dp).clip(CircleShape).background(if (ok) Ok else Color.Transparent)
                .border(2.dp, if (ok) Ok else Muted, CircleShape),
        ) { if (ok) Icon(painterResource(R.drawable.ic_check), contentDescription = null, tint = OnBg, modifier = Modifier.size(14.dp)) }
        Spacer(Modifier.width(10.dp))
        Text(text, fontSize = 15.sp, color = if (ok) OnBg else OnMuted)
    }
}

@Composable
private fun RecordButton(recording: Boolean, onClick: () -> Unit) {
    val pulse = rememberInfiniteTransition(label = "rec")
    val glow by pulse.animateFloat(1f, if (recording) 1.2f else 1f,
        infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "glow")
    Box(contentAlignment = Alignment.Center, modifier = Modifier.size(128.dp)) {
        Box(Modifier.size(112.dp).scale(glow).alpha(0.22f).clip(CircleShape).background(if (recording) Listening else Brand))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(92.dp)
                .clip(CircleShape)
                .background(if (recording) Brush.linearGradient(listOf(Listening, Listening)) else Brush.linearGradient(listOf(Brand, Violet)))
                .clickable(role = Role.Button, onClick = onClick),
        ) {
            Icon(painterResource(if (recording) R.drawable.ic_stop else R.drawable.ic_mic),
                contentDescription = if (recording) "σταμάτημα" else "ηχογράφηση", tint = OnBg, modifier = Modifier.size(42.dp))
        }
    }
}
