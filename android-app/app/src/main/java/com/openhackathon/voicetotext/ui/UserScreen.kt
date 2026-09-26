package com.openhackathon.voicetotext.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
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
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.openhackathon.voicetotext.R
import com.openhackathon.voicetotext.models.ModelDownloader
import com.openhackathon.voicetotext.ui.theme.Bg
import com.openhackathon.voicetotext.ui.theme.Brand
import com.openhackathon.voicetotext.ui.theme.Ok
import com.openhackathon.voicetotext.ui.theme.OnBg
import com.openhackathon.voicetotext.ui.theme.OnMuted
import com.openhackathon.voicetotext.ui.theme.Surface
import com.openhackathon.voicetotext.ui.theme.SurfaceSoft
import java.util.Locale

/** Text on this screen grows with the system font size up to this factor, so it always fits. */
private const val MAX_FONT_SCALE = 1.15f

/**
 * What everyone sees: one screen that fits without scrolling, also with a large system font
 * and display size. The permissions the floating button needs and the models on first start,
 * centred, and at the bottom the switch that shows the button. The steps take only the room
 * above the switch, so the switch is always on screen. Everything else (models, layer 2,
 * analysis, the in-app microphone) is dev mode, opened by a long press anywhere on the screen
 * outside the buttons.
 */
@Composable
fun UserScreen(state: ScreenState, actions: MainActions, onDevMode: () -> Unit) {
    val d = LocalDensity.current
    CompositionLocalProvider(LocalDensity provides Density(d.density, d.fontScale.coerceAtMost(MAX_FONT_SCALE))) {
        UserContent(state, actions, onDevMode)
    }
}

@Composable
private fun UserContent(state: ScreenState, actions: MainActions, onDevMode: () -> Unit) {
    val p by ModelDownloader.progress.collectAsState()
    val ready = state.mic && state.overlay && state.filesPresent
    val on = state.bubbleOn
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bg)
            .safeDrawingPadding()
            .pointerInput(Unit) { detectTapGestures(onLongPress = { onDevMode() }) }
            .padding(start = 18.dp, end = 18.dp, top = 16.dp, bottom = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Column(
            modifier = Modifier.weight(1f).fillMaxWidth().padding(vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterVertically),
        ) {
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
        if (on) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 8.dp)) {
                Box(Modifier.size(10.dp).clip(CircleShape).background(Ok))
                Spacer(Modifier.width(8.dp))
                Text("Ενεργό", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ok)
            }
        }
        Button(
            onClick = { actions.toggleBubble() },
            enabled = ready || on,
            colors = ButtonDefaults.buttonColors(containerColor = if (on) SurfaceSoft else Brand),
            modifier = Modifier.fillMaxWidth().height(64.dp),
            shape = RoundedCornerShape(20.dp),
        ) {
            Text(if (on) "Απενεργοποίηση" else "Ενεργοποίηση", fontSize = 20.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
        }
    }
}

/** One setup step: a number, or a tick once done; what it is for; the button that does it. */
@Composable
private fun Step(
    n: Int, ok: Boolean, title: String, why: String, button: String?,
    progress: Float? = null, onClick: () -> Unit,
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Surface)
            .padding(horizontal = 14.dp, vertical = 12.dp),
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(34.dp).clip(CircleShape).background(if (ok) Ok else SurfaceSoft),
        ) {
            if (ok) Icon(painterResource(R.drawable.ic_check), contentDescription = "έτοιμο", tint = OnBg, modifier = Modifier.size(20.dp))
            else Text("$n", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = OnBg)
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
