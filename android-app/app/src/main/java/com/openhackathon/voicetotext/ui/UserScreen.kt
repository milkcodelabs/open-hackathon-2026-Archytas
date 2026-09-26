package com.openhackathon.voicetotext.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
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
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
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

/**
 * What everyone sees: one screen that fits without scrolling. The permissions the floating
 * button needs, the models on first start, and the switch that shows the button. Everything
 * else (models, layer 2, analysis, the in-app microphone) is dev mode, opened by a long press
 * on the title.
 */
@Composable
fun UserScreen(state: ScreenState, actions: MainActions, onDevMode: () -> Unit) {
    val p by ModelDownloader.progress.collectAsState()
    val ready = state.mic && state.overlay && state.filesPresent
    val on = state.bubbleOn
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bg)
            .safeDrawingPadding()
            .padding(horizontal = 24.dp, vertical = 20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "Φωνή σε κείμενο", fontSize = 30.sp, fontWeight = FontWeight.Bold, color = OnBg,
            modifier = Modifier.pointerInput(Unit) { detectTapGestures(onLongPress = { onDevMode() }) },
        )
        Text(
            "Μίλα και γράφεται στα ελληνικά,\nσε όποια εφαρμογή θέλεις.",
            fontSize = 17.sp, color = OnMuted, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 6.dp),
        )
        Spacer(Modifier.weight(1f))
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Step(1, state.mic, "Μικρόφωνο", "Για να ακούει τη φωνή σου", "Άδεια") { actions.askMic() }
            Step(2, state.overlay, "Πάνω από τις εφαρμογές", "Για το αιωρούμενο κουμπί", "Άδεια") { actions.openOverlaySettings() }
            Step(3, state.accessibility, "Γράψιμο στο πεδίο", "Για να γράφει εκεί που πατάς", "Άνοιγμα") { actions.openAccessibilitySettings() }
            Step(
                4, state.modelsComplete, "Μοντέλα φωνής",
                when {
                    state.modelsComplete -> "Στο κινητό, δουλεύει χωρίς internet"
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
        Spacer(Modifier.weight(1f))
        Button(
            onClick = { actions.toggleBubble() },
            enabled = ready || on,
            colors = ButtonDefaults.buttonColors(containerColor = if (on) Ok else Brand),
            modifier = Modifier.fillMaxWidth().height(72.dp),
            shape = RoundedCornerShape(22.dp),
        ) {
            Text(if (on) "Ενεργό  ·  Απενεργοποίηση" else "Ενεργοποίηση", fontSize = 21.sp, fontWeight = FontWeight.SemiBold)
        }
        Text(
            when {
                on -> "Άνοιξε όποια εφαρμογή θέλεις και πάτα το κουμπί για να μιλήσεις."
                ready -> "Εμφανίζει ένα κουμπί που μένει πάνω από κάθε εφαρμογή."
                else -> "Ολοκλήρωσε πρώτα τα βήματα."
            },
            fontSize = 15.sp, color = OnMuted, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 10.dp),
        )
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
            .clip(RoundedCornerShape(18.dp))
            .background(Surface)
            .padding(horizontal = 16.dp, vertical = 14.dp),
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(38.dp).clip(CircleShape).background(if (ok) Ok else SurfaceSoft),
        ) {
            if (ok) Icon(painterResource(R.drawable.ic_check), contentDescription = "έτοιμο", tint = OnBg, modifier = Modifier.size(22.dp))
            else Text("$n", fontSize = 17.sp, fontWeight = FontWeight.Bold, color = OnBg)
        }
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f)) {
            Text(title, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = OnBg)
            Text(why, fontSize = 14.sp, color = OnMuted)
            if (progress != null) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp).height(8.dp).clip(RoundedCornerShape(4.dp)),
                    color = Brand, trackColor = SurfaceSoft,
                )
            }
        }
        if (!ok && button != null) {
            Spacer(Modifier.width(10.dp))
            Button(onClick = onClick, modifier = Modifier.height(46.dp)) { Text(button, fontSize = 15.sp) }
        }
    }
}
