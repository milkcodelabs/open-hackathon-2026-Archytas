package com.openhackathon.voicetotext.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.openhackathon.voicetotext.R
import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.decoding.Candidate
import com.openhackathon.voicetotext.decoding.Candidates
import com.openhackathon.voicetotext.decoding.WordChoices
import com.openhackathon.voicetotext.llm.LlmCorrector
import com.openhackathon.voicetotext.models.ModelDownloader
import com.openhackathon.voicetotext.ui.theme.Bg
import com.openhackathon.voicetotext.ui.theme.Brand
import com.openhackathon.voicetotext.ui.theme.Listening
import com.openhackathon.voicetotext.ui.theme.Muted
import com.openhackathon.voicetotext.ui.theme.Ok
import com.openhackathon.voicetotext.ui.theme.OnBg
import com.openhackathon.voicetotext.ui.theme.OnMuted
import com.openhackathon.voicetotext.ui.theme.Surface
import com.openhackathon.voicetotext.ui.theme.SurfaceSoft
import com.openhackathon.voicetotext.ui.theme.Thinking
import java.util.Locale

/** A dev-mode test recording and the sentence that was meant (empty when unknown). */
data class SampleChoice(val path: String, val name: String, val reference: String)

/** One personal acoustic model on the phone, as the screen shows it. */
data class PersonalChoice(val key: String, val title: String, val mb: Long, val info: String)

/** Everything the screen shows, read once per recomposition by MainActivity. */
data class ScreenState(
    val mic: Boolean,
    val overlay: Boolean,
    val accessibility: Boolean,
    val filesPresent: Boolean,
    val modelLabel: String,
    val modelMb: Long,
    val lmPresent: Boolean,
    val lmOn: Boolean,
    val lmMb: Long,
    val neuralPresent: Boolean,
    val neuralOn: Boolean,
    val neuralMb: Long,
    /** The personal models on the phone; [personalKey] is the one in use ("" = the general model). */
    val personals: List<PersonalChoice>,
    val personalKey: String,
    /** The acoustic model in memory now, or null before the first recognition loads one. */
    val loadedModel: String?,
    val samples: List<SampleChoice>,
    /** MB of the personal model still to download (0 when it is complete on the phone). */
    val personalMissingMb: Long,
    val llmOn: Boolean,
    val llmProvider: LlmCorrector.Provider,
    /** Providers whose API key was built into the app. */
    val llmKeys: Set<LlmCorrector.Provider>,
    val online: Boolean,
    val bubbleOn: Boolean,
    val recording: Boolean,
    val busy: Boolean,
    val hint: String?,
    val testWav: Boolean,
    val speaker: String,
    val result: String,
    val detail: String,
    val modelsComplete: Boolean,
    val missingMb: Long,
    val totalMb: Long,
    val metered: Boolean,
)

interface MainActions {
    fun askMic()
    fun openOverlaySettings()
    fun openAccessibilitySettings()
    fun toggleBubble()
    fun toggleRecording()
    fun importFiles()
    fun setLanguageModel(on: Boolean)
    fun setNeuralLm(on: Boolean)
    fun selectPersonal(key: String)
    fun setLlm(on: Boolean)
    fun selectLlmProvider(p: LlmCorrector.Provider)
    fun runCheck()
    fun runSample(path: String)
    fun download()
    fun cancelDownload()
    fun downloadPersonal()
}

/** Dev mode: every setting, the in-app microphone and the analysis of the last recognition. */
@Composable
fun MainScreen(state: ScreenState, actions: MainActions, onExit: () -> Unit) {
    Column(
        modifier = Modifier
            .background(Bg)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("DEV MODE", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Thinking,
                letterSpacing = 1.sp, modifier = Modifier.weight(1f))
            TextButton(onClick = onExit) { Text("Έξοδος", fontSize = 15.sp) }
        }
        VoiceModelCard(state, actions)
        Hero(state, actions)
        CandidatesCard(state)
        ModelsCard(state, actions)
        EngineCard(state, actions)
        LlmCard(state, actions)
        PermissionsCard(state, actions)
        SpeakerCard(state)
        CheckCard(state, actions)
        AnalysisCard()
        Spacer(Modifier.height(24.dp))
    }
}

// ---------------------------------------------------------------------- hero

@Composable
private fun Hero(state: ScreenState, actions: MainActions) {
    val ready = state.mic && state.overlay && state.filesPresent
    val color = when {
        state.recording -> Listening
        state.busy -> Thinking
        state.bubbleOn -> Ok
        ready -> Brand
        else -> Muted
    }
    val pulse = rememberInfiniteTransition(label = "pulse")
    val glow by pulse.animateFloat(
        initialValue = 1f, targetValue = if (state.recording) 1.18f else 1f,
        animationSpec = infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "glow",
    )
    val enabled = state.mic && state.filesPresent && !state.busy

    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Φωνή σε κείμενο", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = OnBg)
        Text("Μίλα και γράφεται στα ελληνικά, σε όποια εφαρμογή θέλεις.",
            fontSize = 16.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp))
        Spacer(Modifier.height(20.dp))
        Box(contentAlignment = Alignment.Center, modifier = Modifier.size(190.dp)) {
            Box(Modifier.size(170.dp).scale(glow).alpha(0.22f).clip(CircleShape).background(color))
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(132.dp)
                    .clip(CircleShape)
                    .background(color)
                    .clickable(enabled = enabled) { actions.toggleRecording() }
                    .semantics { contentDescription = if (state.recording) "σταμάτημα" else "μικρόφωνο" },
            ) {
                Icon(
                    painter = painterResource(if (state.recording) R.drawable.ic_stop else R.drawable.ic_mic),
                    contentDescription = null, tint = OnBg, modifier = Modifier.size(60.dp),
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        Text(
            when {
                state.recording -> "Ακούω..."
                state.busy -> "Επεξεργασία..."
                !state.filesPresent -> "Λείπει το μοντέλο"
                !state.mic -> "Χρειάζεται άδεια μικροφώνου"
                else -> "Έτοιμο"
            },
            fontSize = 22.sp, fontWeight = FontWeight.SemiBold, color = OnBg,
        )
        Text(
            state.hint ?: when {
                state.recording -> "Πάτησε ξανά το μικρόφωνο για να σταματήσεις."
                state.busy -> "Μια στιγμή."
                state.mic && state.filesPresent -> "Πάτησε το μικρόφωνο και μίλα."
                !state.filesPresent -> "Κατέβασε τα μοντέλα παρακάτω."
                else -> "Δώσε τις άδειες παρακάτω."
            },
            fontSize = 16.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp),
        )
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = { actions.toggleBubble() },
            enabled = ready || state.bubbleOn,
            modifier = Modifier.fillMaxWidth().height(60.dp),
            shape = RoundedCornerShape(18.dp),
        ) {
            Text(if (state.bubbleOn) "Απόκρυψη εικονιδίου" else "Εμφάνιση εικονιδίου", fontSize = 18.sp)
        }
        Text("Το εικονίδιο αιωρείται πάνω από κάθε εφαρμογή.",
            fontSize = 14.sp, color = OnMuted, modifier = Modifier.padding(top = 6.dp))
    }
}

// ---------------------------------------------------------------------- candidates

/**
 * The best sentences the decoder found for the last recognition (from the panel or the
 * bubble), best first. The chosen one is shown large; tapping another chooses it. Words that
 * differ from the best sentence are highlighted so the alternatives can be read at a glance.
 */
@Composable
private fun CandidatesCard(state: ScreenState) {
    val last by Recognizer.lastResult.collectAsState()
    val res = last ?: return
    val candidates = res.candidates
    if (candidates.isEmpty()) return
    var chosen by remember(res) { mutableIntStateOf(0) }
    var copied by remember(res) { mutableStateOf(false) }
    val context = LocalContext.current
    val pick = candidates[chosen.coerceIn(0, candidates.lastIndex)]

    Section("Πιθανές προτάσεις") {
        Text(pick.text, fontSize = 24.sp, fontWeight = FontWeight.SemiBold, color = OnBg, lineHeight = 30.sp)
        Spacer(Modifier.height(10.dp))
        Button(
            onClick = {
                val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("voice-to-text", pick.text))
                copied = true
            },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(16.dp),
        ) { Text(if (copied) "Αντιγράφηκε" else "Αντιγραφή", fontSize = 16.sp) }

        if (candidates.size > 1) {
            Spacer(Modifier.height(14.dp))
            Text("Πάτα μια πρόταση για να την επιλέξεις:", fontSize = 14.sp, color = OnMuted)
            Spacer(Modifier.height(8.dp))
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                candidates.forEachIndexed { i, c ->
                    CandidateRow(rank = i + 1, candidate = c, selected = i == chosen) {
                        chosen = i
                        copied = false
                    }
                }
            }
            // what the bubble above the keyboard would ask about for this result
            val spots = remember(res) { WordChoices.find(candidates.first().words, candidates) }
            Spacer(Modifier.height(10.dp))
            Text(
                if (spots.isEmpty()) {
                    "Το εικονίδιο δεν θα ρωτούσε: όλες οι λέξεις πάνω από ${(WordChoices.THRESHOLD * 100).toInt()}%."
                } else {
                    "Το εικονίδιο θα ρωτούσε: " + spots.joinToString("  ·  ") { s ->
                        s.options.joinToString(" / ") { o -> "${o.text.ifEmpty { "(τίποτα)" }} ${(o.probability * 100).toInt()}%" }
                    }
                },
                fontSize = 13.sp, color = OnMuted, lineHeight = 18.sp,
            )
        } else if (!state.lmOn) {
            Spacer(Modifier.height(10.dp))
            Text("Άνοιξε το γλωσσικό μοντέλο για να βλέπεις εναλλακτικές προτάσεις.", fontSize = 14.sp, color = OnMuted)
        }
    }
}

@Composable
private fun CandidateRow(rank: Int, candidate: Candidate, selected: Boolean, onClick: () -> Unit) {
    val text = buildAnnotatedString {
        candidate.words.forEachIndexed { i, w ->
            if (i > 0) append(" ")
            if (i in candidate.changed) {
                withStyle(SpanStyle(color = Thinking, fontWeight = FontWeight.Bold)) { append(w) }
            } else {
                append(w)
            }
        }
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(if (selected) Brand.copy(alpha = 0.22f) else SurfaceSoft)
            .border(
                width = if (selected) 2.dp else 0.dp,
                color = if (selected) Brand else SurfaceSoft,
                shape = RoundedCornerShape(14.dp),
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 10.dp)
            .semantics { contentDescription = "πρόταση $rank: ${candidate.text}" },
    ) {
        Text("$rank", fontSize = 16.sp, fontWeight = FontWeight.Bold,
            color = if (selected) Brand else OnMuted, modifier = Modifier.width(26.dp))
        Text(text, fontSize = 17.sp, color = OnBg, modifier = Modifier.weight(1f))
        Spacer(Modifier.width(10.dp))
        Text(percent(candidate.probability), fontSize = 14.sp, color = OnMuted)
    }
}

private fun percent(p: Float): String = when {
    p >= 0.995f -> ">99%"
    p < 0.01f -> "<1%"
    else -> "${(p * 100).toInt()}%"
}

// ---------------------------------------------------------------------- models

@Composable
private fun ModelsCard(state: ScreenState, actions: MainActions) {
    val p by ModelDownloader.progress.collectAsState()
    Section("Μοντέλα") {
        StatusRow(
            ok = state.modelsComplete,
            text = if (state.modelsComplete) "Έτοιμα στο κινητό (${state.totalMb} MB)"
            else "Λείπουν ${state.missingMb} MB",
        )
        when {
            p.running -> {
                Spacer(Modifier.height(12.dp))
                LinearProgressIndicator(
                    progress = { p.fraction },
                    modifier = Modifier.fillMaxWidth().height(12.dp).clip(RoundedCornerShape(6.dp)),
                    color = Brand,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    String.format(Locale.US, "%d%%   %d / %d MB", (p.fraction * 100).toInt(),
                        p.bytesDone / 1_000_000, p.bytesTotal / 1_000_000),
                    fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = OnBg,
                )
                Text(
                    if (p.verifying) "Έλεγχος: ${p.file}"
                    else listOfNotNull(
                        p.file,
                        if (p.bytesPerSecond > 0) String.format(Locale.US, "%.1f MB/s", p.bytesPerSecond / 1e6) else null,
                        p.secondsLeft?.let { "απομένουν ${remaining(it)}" } ?: "υπολογίζεται ο χρόνος...",
                    ).joinToString("  ·  "),
                    fontSize = 15.sp, color = OnMuted,
                )
                TextButton(onClick = { actions.cancelDownload() }) { Text("Ακύρωση", fontSize = 16.sp) }
            }
            !state.modelsComplete -> {
                Spacer(Modifier.height(12.dp))
                Button(
                    onClick = { actions.download() },
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(16.dp),
                ) { Text("Λήψη μοντέλων (${state.missingMb} MB)", fontSize = 17.sp) }
                if (state.metered) {
                    Text("Είσαι σε δίκτυο κινητής: η λήψη χρεώνεται στα δεδομένα σου. Με Wi-Fi ξεκινά μόνη της.",
                        fontSize = 14.sp, color = OnMuted, modifier = Modifier.padding(top = 6.dp))
                }
                p.error?.let {
                    Text("Η λήψη σταμάτησε: $it", fontSize = 14.sp, color = Listening, modifier = Modifier.padding(top = 6.dp))
                }
            }
        }
        Spacer(Modifier.height(14.dp))
        StatusRow(
            ok = state.personalMissingMb == 0L,
            text = if (state.personalMissingMb == 0L) "Προσωπικό μοντέλο στο κινητό (${ModelDownloader.personalBytes / 1_000_000} MB)"
            else "Προσωπικό μοντέλο: προαιρετικό",
        )
        if (state.personalMissingMb > 0 && !p.running) {
            Spacer(Modifier.height(8.dp))
            OutlinedButton(
                onClick = { actions.downloadPersonal() },
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text("Λήψη προσωπικού μοντέλου (${state.personalMissingMb} MB)", fontSize = 16.sp) }
            Text("Εκπαιδευμένο στη φωνή του ομιλητή. Μετά τη λήψη διαλέγεται από το «Προσωπικό μοντέλο».",
                fontSize = 13.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp))
        }
        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = { actions.importFiles() }, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text("Εισαγωγή αρχείων από το κινητό", fontSize = 16.sp)
        }
    }
}

private fun remaining(seconds: Long): String = when {
    seconds < 60 -> "~$seconds δευτ."
    seconds < 3600 -> "~${(seconds + 59) / 60} λεπτά"
    else -> String.format(Locale.US, "~%d ώρ. %d λεπ.", seconds / 3600, (seconds % 3600) / 60)
}

// ---------------------------------------------------------------------- voice model

/**
 * Which acoustic model listens: the general one or a speaker's own, one row each, with the
 * model in memory right now; and the test recordings on the phone, run through the chosen
 * model with the meant sentence beside the result. A personal model is trained on one
 * person's voice, so it is only fair to judge it on that person's recordings.
 */
@Composable
private fun VoiceModelCard(state: ScreenState, actions: MainActions) {
    Section("Μοντέλο φωνής") {
        Text(
            "Φορτωμένο τώρα: " + (state.loadedModel ?: "κανένα ακόμα, φορτώνει με την πρώτη χρήση"),
            fontSize = 14.sp, color = OnMuted,
        )
        Spacer(Modifier.height(8.dp))
        val options = listOf(PersonalChoice("", "Γενικό", state.modelMb, "Omnilingual, για κάθε ομιλητή")) + state.personals
        options.forEach { o ->
            val selected = state.personalKey == o.key
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(if (selected) SurfaceSoft else Color.Transparent)
                    .clickable { actions.selectPersonal(o.key) }
                    .padding(vertical = 8.dp, horizontal = 4.dp),
            ) {
                RadioButton(selected = selected, onClick = { actions.selectPersonal(o.key) })
                Column(Modifier.weight(1f)) {
                    Text(o.title, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = OnBg)
                    if (o.info.isNotBlank()) Text(o.info, fontSize = 13.sp, color = OnMuted)
                }
            }
        }
        if (state.samples.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            Text("Δοκιμή με ηχογράφηση", fontSize = 16.sp, color = OnBg)
            state.samples.forEach { s ->
                Spacer(Modifier.height(8.dp))
                OutlinedButton(
                    onClick = { actions.runSample(s.path) },
                    enabled = state.filesPresent && !state.busy,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                ) {
                    Text("▶  ${s.name}" + (if (s.reference.isNotBlank()) "\n${s.reference}" else ""),
                        fontSize = 15.sp, textAlign = TextAlign.Center)
                }
            }
            if (state.result.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                Text(state.result, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = OnBg)
            }
            if (state.detail.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Text(state.detail, fontSize = 13.sp, color = OnMuted)
            }
        }
    }
}

// ---------------------------------------------------------------------- engine

@Composable
private fun EngineCard(state: ScreenState, actions: MainActions) {
    Section("Μοντέλο αναγνώρισης") {
        StatusRow(
            ok = state.filesPresent,
            text = if (state.filesPresent) "${state.modelLabel}: ${state.modelMb} MB" else "Λείπει το ${state.modelLabel}",
        )
        Text(
            "Meta Omnilingual CTC 300M, μόνο με τα ελληνικά γράμματα. Τρέχει όλο στο κινητό.",
            fontSize = 14.sp, color = OnMuted, modifier = Modifier.padding(top = 8.dp),
        )
        run {
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Γλωσσικό μοντέλο", fontSize = 16.sp, color = OnBg)
                    Text(
                        when {
                            !state.lmPresent -> "Λείπει το el_3gram.ngram"
                            state.lmOn -> "Ενεργό, ${state.lmMb} MB. Διορθώνει λέξεις που ακούστηκαν σωστά."
                            else -> "Ανενεργό. Καθαρή ακουστική έξοδος."
                        },
                        fontSize = 14.sp, color = OnMuted,
                    )
                }
                Switch(
                    checked = state.lmOn && state.lmPresent,
                    onCheckedChange = { actions.setLanguageModel(it) },
                    enabled = state.lmPresent,
                )
            }
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Νευρωνικό μοντέλο συμφραζομένων", fontSize = 16.sp, color = OnBg)
                    Text(
                        when {
                            !state.neuralPresent -> "Λείπει το el_gpt2 (κατέβασέ το από τα Μοντέλα)."
                            !state.lmOn -> "Χρειάζεται το γλωσσικό μοντέλο ανοιχτό."
                            state.neuralOn -> "Ενεργό, ${state.neuralMb} MB. Διαλέγει ανάμεσα στις καλύτερες " +
                                "προτάσεις με βάση όλη τη φράση. Δεν αλλάζει ό,τι ακούστηκε."
                            else -> "Ανενεργό."
                        },
                        fontSize = 14.sp, color = OnMuted,
                    )
                }
                Switch(
                    checked = state.neuralOn && state.neuralPresent && state.lmOn,
                    onCheckedChange = { actions.setNeuralLm(it) },
                    enabled = state.neuralPresent && state.lmOn,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------- online correction

/**
 * The optional online check: a switch, which service answers, and the last exchange (what
 * was sent, what came back, how long it took) so the effect can be judged.
 */
@Composable
private fun LlmCard(state: ScreenState, actions: MainActions) {
    val last by LlmCorrector.last.collectAsState()
    val hasKey = state.llmProvider in state.llmKeys
    Section("Διόρθωση στο διαδίκτυο") {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Έλεγχος από γλωσσικό μοντέλο (LLM)", fontSize = 16.sp, color = OnBg)
                Text(
                    when {
                        !hasKey -> "Λείπει το κλειδί API για το ${state.llmProvider.label}."
                        !state.llmOn -> "Ανενεργό. Όλα γίνονται στο κινητό."
                        !state.online -> "Ενεργό, αλλά χωρίς σύνδεση: γράφεται ό,τι βρει το κινητό."
                        else -> "Ενεργό. Στέλνει μόνο το κείμενο, ποτέ τον ήχο, και γράφει την πρόταση που βγάζει νόημα."
                    },
                    fontSize = 14.sp, color = OnMuted,
                )
            }
            Switch(
                checked = state.llmOn && hasKey,
                onCheckedChange = { actions.setLlm(it) },
                enabled = hasKey,
            )
        }
        Spacer(Modifier.height(12.dp))
        val providers = LlmCorrector.Provider.entries
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            providers.forEachIndexed { i, p ->
                SegmentedButton(
                    selected = state.llmProvider == p,
                    onClick = { actions.selectLlmProvider(p) },
                    shape = SegmentedButtonDefaults.itemShape(i, providers.size),
                    modifier = Modifier.height(54.dp),
                ) {
                    Text(if (p in state.llmKeys) p.label else "${p.label} (χωρίς κλειδί)", fontSize = 14.sp, maxLines = 1)
                }
            }
        }
        Text(
            "${state.llmProvider.model}. Το κλειδί μπαίνει στο local.properties ως " +
                (if (state.llmProvider == LlmCorrector.Provider.GEMINI) "GEMINI_API_KEY" else "GROQ_API_KEY") +
                " και χρειάζεται νέο build.",
            fontSize = 13.sp, color = OnMuted, modifier = Modifier.padding(top = 8.dp),
        )
        last?.let { o ->
            Spacer(Modifier.height(12.dp))
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(SurfaceSoft)
                    .padding(14.dp),
            ) {
                Text("${o.provider.label}  ·  ${o.ms} ms", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = OnBg)
                if (o.sent.isNotEmpty()) {
                    Text(o.sent, fontSize = 13.sp, fontFamily = FontFamily.Monospace, color = OnMuted,
                        modifier = Modifier.padding(top = 6.dp))
                }
                Text(
                    o.text?.let { "→ $it" } ?: "Κράτησε το αποτέλεσμα του κινητού: ${o.error}",
                    fontSize = 17.sp, fontWeight = FontWeight.SemiBold,
                    color = if (o.text != null) Ok else Listening,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }
    }
}

// ---------------------------------------------------------------------- permissions

@Composable
private fun PermissionsCard(state: ScreenState, actions: MainActions) {
    Section("Ρύθμιση") {
        PermissionRow(state.mic, "Μικρόφωνο", "Άδεια") { actions.askMic() }
        PermissionRow(state.overlay, "Εμφάνιση πάνω από εφαρμογές", "Άδεια") { actions.openOverlaySettings() }
        PermissionRow(state.accessibility, "Γράψιμο στο πεδίο κειμένου", "Άνοιγμα") { actions.openAccessibilitySettings() }
    }
}

@Composable
private fun PermissionRow(ok: Boolean, text: String, button: String, onClick: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Box(Modifier.weight(1f)) { StatusRow(ok, text) }
        if (!ok) {
            Button(onClick = onClick, modifier = Modifier.height(48.dp)) { Text(button, fontSize = 15.sp) }
        }
    }
}

// ---------------------------------------------------------------------- speaker

@Composable
private fun SpeakerCard(state: ScreenState) {
    Section("Ο ομιλητής") {
        Text(if (state.speaker == "default") "Προεπιλεγμένο προφίλ" else "Προφίλ: ${state.speaker}",
            fontSize = 16.sp, color = OnBg)
        Text("Διαβάζοντας μια σύντομη λίστα λέξεων, η εφαρμογή μαθαίνει πώς μιλάς εσύ και γίνεται πιο ακριβής.",
            fontSize = 14.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp))
        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = {}, enabled = false, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text("Έναρξη εκμάθησης φωνής", fontSize = 16.sp)
        }
        Text("Έρχεται στο επόμενο στάδιο.", fontSize = 13.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp))
    }
}

// ---------------------------------------------------------------------- check

@Composable
private fun CheckCard(state: ScreenState, actions: MainActions) {
    var tryText by remember { mutableStateOf("") }
    Section("Έλεγχος") {
        OutlinedButton(
            onClick = { actions.runCheck() },
            enabled = state.filesPresent && state.testWav,
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) { Text("Δοκιμή με το test.wav", fontSize = 16.sp) }
        Spacer(Modifier.height(10.dp))
        OutlinedTextField(
            value = tryText,
            onValueChange = { tryText = it },
            placeholder = { Text("Δοκίμασε εδώ: πάτα μέσα, μετά πάτα το αιωρούμενο εικονίδιο") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2,
        )
        if (state.result.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            Text(state.result, fontSize = 22.sp, fontWeight = FontWeight.SemiBold, color = OnBg)
        }
        if (state.detail.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(state.detail, fontSize = 13.sp, fontFamily = FontFamily.Monospace, color = OnMuted)
        }
    }
}

// ---------------------------------------------------------------------- analysis

/**
 * How the last recognition (panel or bubble) went from sound to text: what layer 1 heard,
 * what each later layer changed (changed words highlighted), how long each step took, and
 * the final sentence.
 */
@Composable
private fun AnalysisCard() {
    val last by Recognizer.lastResult.collectAsState()
    val res = last ?: return
    if (res.stages.isEmpty()) return
    val em = res.emissions
    Section("Ανάλυση της τελευταίας αναγνώρισης") {
        Text(
            String.format(Locale.US, "ήχος %.2f s  ·  %d καρέ των 20 ms  ·  κενά %.0f%%  ·  σύνολο %d ms (RTF %.2f)",
                res.audioSeconds, em.numFrames, em.blankFraction() * 100, res.inferenceMs, res.rtf),
            fontSize = 14.sp, color = OnMuted,
        )
        Spacer(Modifier.height(14.dp))
        var previous: List<String>? = null
        res.stages.forEachIndexed { i, st ->
            val words = st.text.split(' ').filter { it.isNotBlank() }
            StageBlock(st, words, previous)
            if (i == 0) {
                val conf = em.greedyWords()
                if (conf.isNotEmpty()) {
                    Text(
                        "βεβαιότητα ανά λέξη: " + conf.joinToString("  ") {
                            String.format(Locale.US, "%s %.2f", it.text, it.confidence)
                        },
                        fontSize = 13.sp, color = OnMuted, modifier = Modifier.padding(top = 4.dp),
                    )
                }
            }
            Text("↓", fontSize = 20.sp, color = OnMuted,
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), textAlign = TextAlign.Center)
            previous = words
        }
        Column(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(Ok.copy(alpha = 0.16f))
                .padding(14.dp),
        ) {
            Text("ΤΕΛΙΚΟ", fontSize = 13.sp, fontWeight = FontWeight.Bold, color = Ok, letterSpacing = 1.sp)
            Text(res.text.ifBlank { "(τίποτα)" }, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, color = OnBg,
                modifier = Modifier.padding(top = 4.dp))
        }
    }
}

@Composable
private fun StageBlock(st: Recognizer.Stage, words: List<String>, previous: List<String>?) {
    val changed = if (previous == null) emptySet() else Candidates.differing(words, previous)
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(SurfaceSoft)
            .padding(14.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(st.layer, fontSize = 13.sp, fontWeight = FontWeight.Bold, color = OnBg,
                modifier = Modifier.clip(RoundedCornerShape(8.dp)).background(Brand).padding(horizontal = 8.dp, vertical = 3.dp))
            Spacer(Modifier.width(10.dp))
            Text("${st.ms} ms", fontSize = 13.sp, color = OnMuted)
        }
        Text(st.what, fontSize = 13.sp, color = OnMuted, modifier = Modifier.padding(top = 6.dp))
        Text(
            buildAnnotatedString {
                if (words.isEmpty()) append("(τίποτα)")
                words.forEachIndexed { i, w ->
                    if (i > 0) append(" ")
                    if (i in changed) withStyle(SpanStyle(color = Thinking, fontWeight = FontWeight.Bold)) { append(w) }
                    else append(w)
                }
            },
            fontSize = 18.sp, color = OnBg, modifier = Modifier.padding(top = 6.dp),
        )
        if (previous != null) {
            Text(
                if (changed.isEmpty() && words.size == previous.size) "χωρίς αλλαγή"
                else "άλλαξαν ${changed.size} λέξεις",
                fontSize = 12.sp, color = if (changed.isEmpty()) OnMuted else Thinking,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
    }
}

// ---------------------------------------------------------------------- building blocks

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Surface),
        shape = RoundedCornerShape(22.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(20.dp)) {
            Text(greekCaps(title), fontSize = 13.sp, fontWeight = FontWeight.Bold, color = OnMuted,
                letterSpacing = 1.sp)
            Spacer(Modifier.height(12.dp))
            content()
        }
    }
}

/** A requirement: green tick when satisfied, grey mark when not. */
@Composable
private fun StatusRow(ok: Boolean, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(
            painter = painterResource(if (ok) R.drawable.ic_check else R.drawable.ic_pending),
            contentDescription = if (ok) "έτοιμο" else "εκκρεμεί",
            tint = if (ok) Ok else Muted,
            modifier = Modifier.size(26.dp),
        )
        Spacer(Modifier.width(14.dp))
        Text(text, fontSize = 16.sp, color = if (ok) OnBg else OnBg.copy(alpha = 0.75f))
    }
}


/** Greek capitals carry no stress marks: "Ανάλυση" -> "ΑΝΑΛΥΣΗ", not "ΑΝΆΛΥΣΗ". */
private fun greekCaps(text: String): String =
    java.text.Normalizer.normalize(text.uppercase(Locale.ROOT), java.text.Normalizer.Form.NFD)
        .filter { it != '\u0301' }
        .let { java.text.Normalizer.normalize(it, java.text.Normalizer.Form.NFC) }
