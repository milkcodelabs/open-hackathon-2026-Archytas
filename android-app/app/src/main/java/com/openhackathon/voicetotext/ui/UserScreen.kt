package com.openhackathon.voicetotext.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.waitForUpOrCancellation
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.IconButton
import androidx.compose.material3.TextButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.openhackathon.voicetotext.R
import com.openhackathon.voicetotext.models.ModelDownloader
import com.openhackathon.voicetotext.ui.theme.Bg
import com.openhackathon.voicetotext.ui.theme.Brand
import com.openhackathon.voicetotext.ui.theme.Muted
import com.openhackathon.voicetotext.ui.theme.Ok
import com.openhackathon.voicetotext.ui.theme.OnBg
import com.openhackathon.voicetotext.ui.theme.OnMuted
import com.openhackathon.voicetotext.ui.theme.Surface
import com.openhackathon.voicetotext.ui.theme.SurfaceSoft
import java.util.Locale

/** Text on this screen grows with the system font size up to this factor, so it always fits. */
private const val MAX_FONT_SCALE = 1.15f
/** How long the title must be held down to open dev mode. */
private const val DEV_HOLD_MS = 6_000L

/** Second colour of the brand gradient, and a lighter brand tone for text on the dark background. */
private val Violet = Color(0xFF8B5CF6)
private val BrandLight = Color(0xFF8FA0FF)
/** The glow at the top of the screen. */
private val BgTop = Color(0xFF1A2146)
private val CardEdge = Color.White.copy(alpha = 0.06f)

/**
 * What everyone sees: one screen that fits without scrolling, also with a large system font
 * and display size. A centred block with the title and the setup steps (the permissions the
 * floating button needs, the models on first start), and at the bottom the switch that shows
 * the button. The block takes only the room above the switch, so the switch is always on
 * screen. Everything else (models,
 * layer 2, analysis, the in-app microphone) is dev mode, opened by holding the title down for
 * [DEV_HOLD_MS]; lifting or dragging earlier does nothing.
 */
@Composable
fun UserScreen(state: ScreenState, actions: MainActions, onDevMode: () -> Unit, onTraining: () -> Unit) {
    val d = LocalDensity.current
    CompositionLocalProvider(LocalDensity provides Density(d.density, d.fontScale.coerceAtMost(MAX_FONT_SCALE))) {
        UserContent(state, actions, onDevMode, onTraining)
    }
}

@Composable
private fun UserContent(state: ScreenState, actions: MainActions, onDevMode: () -> Unit, onTraining: () -> Unit) {
    val p by ModelDownloader.progress.collectAsState()
    val ready = state.mic && state.overlay && state.filesPresent
    val on = state.bubbleOn
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Brush.verticalGradient(0f to BgTop, 0.5f to Bg))
            .safeDrawingPadding()
            .padding(start = 18.dp, end = 18.dp, top = 12.dp, bottom = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        SettingsMenu(onTraining)
        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Title(onDevMode)
                Spacer(Modifier.height(32.dp))
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Step(1, state.mic, "Μικρόφωνο", "Για να σε ακούει", "Άδεια") { actions.askMic() }
                    Step(2, state.overlay, "Πάνω από εφαρμογές", "Για το αιωρούμενο κουμπί", "Άδεια") { actions.openOverlaySettings() }
                    Step(3, state.accessibility, "Γράψιμο σε πεδία", "Γράφει όπου πατάς", "Άνοιγμα") { actions.openAccessibilitySettings() }
                    Step(
                        4, state.modelsComplete, "Μοντέλα φωνής",
                        when {
                            state.modelsComplete -> "Δουλεύει χωρίς internet"
                            p.running -> String.format(Locale.US, "Λήψη %d%%  ·  %d / %d MB", (p.fraction * 100).toInt(),
                                p.bytesDone / 1_000_000, p.bytesTotal / 1_000_000)
                            p.error != null -> "Η λήψη σταμάτησε, ξαναδοκίμασε"
                            state.metered -> "${state.missingMb} MB, καλύτερα με Wi-Fi"
                            else -> "${state.missingMb} MB, μία φορά"
                        },
                        if (p.running) null else "Λήψη",
                        progress = if (p.running) p.fraction else null,
                    ) { actions.download() }
                }
            }
        }
        if (on) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 10.dp)) {
                Box(Modifier.size(10.dp).clip(CircleShape).background(Ok))
                Spacer(Modifier.width(8.dp))
                Text("Ενεργό", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ok)
            }
        }
        MainSwitch(on = on, enabled = ready || on) { actions.toggleBubble() }
    }
}

/**
 * The gear at the top right: a small menu with the initial training. Choosing it first shows
 * a notice that the training is a mock (the takes are saved on the phone, nothing is sent,
 * they can be deleted at the end); only "Συνέχεια" opens the recording screen.
 */
@Composable
private fun SettingsMenu(onTraining: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    var notice by remember { mutableStateOf(false) }
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.TopEnd) {
        Box {
            IconButton(onClick = { open = true }) {
                Icon(painterResource(R.drawable.ic_settings), contentDescription = "ρυθμίσεις", tint = OnMuted, modifier = Modifier.size(28.dp))
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = Surface) {
                DropdownMenuItem(
                    text = { Text("Αρχική εκπαίδευση", fontSize = 17.sp, color = OnBg) },
                    onClick = { open = false; notice = true },
                )
            }
        }
    }
    if (notice) {
        AlertDialog(
            onDismissRequest = { notice = false },
            title = { Text("Δοκιμαστική λειτουργία") },
            text = {
                Text(
                    "Η αρχική εκπαίδευση είναι προς το παρόν επίδειξη (mock). Οι ηχογραφήσεις αποθηκεύονται " +
                        "κανονικά στο κινητό, αλλά δεν στέλνονται για εκπαίδευση και δεν αλλάζουν το μοντέλο. " +
                        "Στο τέλος μπορείς να τις διαγράψεις, για να μην πιάνουν χώρο.",
                    fontSize = 16.sp,
                )
            },
            confirmButton = { TextButton(onClick = { notice = false; onTraining() }) { Text("Συνέχεια", fontSize = 16.sp) } },
            dismissButton = { TextButton(onClick = { notice = false }) { Text("Ακύρωση", fontSize = 16.sp, color = OnMuted) } },
            containerColor = Surface, titleContentColor = OnBg, textContentColor = OnBg,
        )
    }
}

/** "Archytas Voice"; held down for [DEV_HOLD_MS] it opens dev mode. */
@Composable
private fun Title(onDevMode: () -> Unit) {
    Text(
        buildAnnotatedString {
            append("Archytas ")
            withStyle(SpanStyle(color = BrandLight)) { append("Voice") }
        },
        fontSize = 32.sp, fontWeight = FontWeight.Bold, color = OnBg, letterSpacing = 0.5.sp, maxLines = 1,
        modifier = Modifier.pointerInput(Unit) {
            awaitEachGesture {
                awaitFirstDown()
                // null only when the time runs out with the finger still down
                if (withTimeoutOrNull(DEV_HOLD_MS) { waitForUpOrCancellation(); true } == null) onDevMode()
            }
        },
    )
}

/** Shows or hides the floating button: brand gradient when it can be turned on, dark with a green edge when on. */
@Composable
private fun MainSwitch(on: Boolean, enabled: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(22.dp)
    val fill = when {
        on -> Brush.horizontalGradient(listOf(SurfaceSoft, SurfaceSoft))
        enabled -> Brush.horizontalGradient(listOf(Brand, Violet))
        else -> Brush.horizontalGradient(listOf(Surface, Surface))
    }
    val content = if (enabled) OnBg else Muted
    Row(
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .height(66.dp)
            .clip(shape)
            .background(fill)
            .border(1.5.dp, if (on) Ok.copy(alpha = 0.7f) else Color.Transparent, shape)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick),
    ) {
        Icon(painterResource(if (on) R.drawable.ic_stop else R.drawable.ic_mic), contentDescription = null,
            tint = content, modifier = Modifier.size(24.dp))
        Spacer(Modifier.width(10.dp))
        Text(if (on) "Απενεργοποίηση" else "Ενεργοποίηση", fontSize = 20.sp, fontWeight = FontWeight.SemiBold,
            color = content, maxLines = 1)
    }
}

/** One setup step: a number, or a tick once done; what it is for; the button that does it. */
@Composable
private fun Step(
    n: Int, ok: Boolean, title: String, why: String, button: String?,
    progress: Float? = null, onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(18.dp)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(Surface)
            .border(1.dp, CardEdge, shape)
            .padding(horizontal = 14.dp, vertical = 12.dp),
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(34.dp)
                .clip(CircleShape)
                .background(if (ok) Ok else Color.Transparent)
                .border(2.dp, if (ok) Ok else Brand, CircleShape),
        ) {
            if (ok) Icon(painterResource(R.drawable.ic_check), contentDescription = "έτοιμο", tint = OnBg, modifier = Modifier.size(20.dp))
            else Text("$n", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = BrandLight)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            // one line each when the step is done; up to two next to its button
            Text(title, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = OnBg, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(why, fontSize = 14.sp, color = OnMuted, maxLines = 2, overflow = TextOverflow.Ellipsis)
            if (progress != null) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth().padding(top = 6.dp).height(6.dp).clip(RoundedCornerShape(3.dp)),
                    color = Brand, trackColor = SurfaceSoft,
                )
            }
        }
        if (!ok && button != null) {
            Spacer(Modifier.width(8.dp))
            Button(onClick = onClick, modifier = Modifier.height(44.dp), contentPadding = PaddingValues(horizontal = 14.dp)) {
                Text(button, fontSize = 15.sp, maxLines = 1)
            }
        }
    }
}
