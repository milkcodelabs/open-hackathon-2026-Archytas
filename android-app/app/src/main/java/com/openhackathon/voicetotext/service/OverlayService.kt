package com.openhackathon.voicetotext.service

import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.decoding.Candidate
import com.openhackathon.voicetotext.decoding.WordChoices
import com.openhackathon.voicetotext.llm.LlmCorrector
import com.openhackathon.voicetotext.R
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import android.animation.ArgbEvaluator
import android.animation.ValueAnimator
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.text.TextUtils
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.ViewGroup
import android.view.WindowManager
import android.view.animation.PathInterpolator
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * The floating microphone. Lives above every app as a foreground service.
 *
 * Tap to listen (the circle turns red and pulses), tap again to transcribe (amber, spinner),
 * and the text is typed into whatever field has focus. Long press runs the pushed test.wav
 * instead of the mic. Drag it anywhere; it snaps to the nearer side when released.
 * When the software keyboard is open, the circle is swapped for a bar docked on top of it,
 * and swapped back to its previous position when the keyboard closes.
 * The bar keeps its two targets in the corners, Undo far left and the mic far right, with
 * nothing tappable between them. Floating, a small Undo bubble slides out from behind the
 * circle after each typed sentence and tucks itself back after a few seconds or on the next tap.
 * When the typed sentence has words the recognizer was unsure of, the alternatives for the
 * first such word slide in between the two buttons; tapping one rewrites that word in the
 * field, then the next unsure word is offered.
 * With the LLM switch on and a network, the sentence is first checked by [LlmCorrector]; if
 * it is slow, the phone's sentence is typed at once (a ring spins around the mic) and the
 * answer replaces it when it comes, with the phone's words offered back as the alternative.
 * Docked, the top guess is always typed straight away, then refined in place: a late LLM
 * answer or a tapped word choice rewrites only the words that actually changed, not the whole
 * sentence. Floating, nothing is typed until the person picks one: below [CONFIDENCE_THRESHOLD]
 * the bubble instead grows a header showing the top guess in full and a few alternatives under
 * it, and the tapped one is typed, after which the window shrinks back to exactly where it was.
 * The docked bar's icons are always plain white; the rectangle itself carries the colour -
 * solid blue when ready, solid red while recording, solid orange on an error or for a moment
 * after Undo is tapped - and it never shows the loading state's muted grey-blue, only that
 * fixed palette. Floating keeps its own separate, unrelated per-state circle colour.
 */
class OverlayService : Service() {

    private enum class State(val color: Int, val label: String) {
        LOADING(0xFF6B7290.toInt(), "φορτώνει"),
        IDLE(0xFF4F6BED.toInt(), "έτοιμο"),
        LISTENING(0xFFE5484D.toInt(), "ακούει"),
        THINKING(0xFFF2820D.toInt(), "σκέφτεται"),
        ERROR(0xFF3A3F55.toInt(), "σφάλμα"),
    }

    private lateinit var wm: WindowManager
    private lateinit var root: FrameLayout
    private lateinit var circle: FrameLayout
    private lateinit var icon: ImageView
    private lateinit var pulse: View
    private lateinit var spinner: ProgressBar
    private lateinit var strip: HorizontalScrollView
    private lateinit var chips: LinearLayout
    private lateinit var suggestionBox: LinearLayout
    private lateinit var suggestionHeader: TextView
    private lateinit var dockTrack: View
    private lateinit var undoButton: FrameLayout
    private lateinit var undoSatellite: FrameLayout
    private lateinit var params: WindowManager.LayoutParams

    private val bg = GradientDrawable().apply { shape = GradientDrawable.RECTANGLE }
    private val ring = GradientDrawable().apply { shape = GradientDrawable.OVAL }
    private val stripBg = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        setColor(0xF21C2033.toInt())
    }
    private val panelBg = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        setColor(0xF21C2033.toInt())
    }
    /** The docked bar's flat backdrop, and the mic/Undo buttons riding on it: always this blue,
     *  whatever the mic's state - only the icons change colour when something needs attention. */
    private val trackBg = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        setColor(DOCK_BLUE)
    }
    private val dockButtonBg = GradientDrawable().apply {
        shape = GradientDrawable.OVAL
        setColor(DOCK_BLUE)
    }
    private val ease = PathInterpolator(0.2f, 0f, 0f, 1f)
    private val recorder = AudioRecorder()
    private val main = Handler(Looper.getMainLooper())
    private val argb = ArgbEvaluator()

    /** True while the keyboard is open and the overlay is the bar above it. */
    private var docked = false
    /** Where the circle was before docking; only written on the circle -> bar swap. */
    private var bubbleX = 0
    private var bubbleY = 0
    private var imeHeight = 0

    private var state = State.LOADING
    private var currentColor = State.LOADING.color
    private var colorAnim: ValueAnimator? = null
    private var pulseAnim: ValueAnimator? = null
    private var longPressed = false

    /** Choice strip: shown while [choice] is open in the bar; progress 0 = full bar, 1 = strip open. */
    private var optionsShown = false
    private var optionsProgress = 0f
    private var optionsAnim: ValueAnimator? = null
    /** When the open choice was last hidden by the keyboard going away. */
    private var hiddenSince = 0L

    /** Bumped by every dictation, so a late LLM answer for an older one is dropped. */
    private var dictation = 0
    /** The sentence just typed, which a late LLM answer may still replace. */
    private class Typed(val id: Int, val res: Recognizer.Result, val words: List<String>) {
        val text = words.joinToString(" ")
        /** The person tapped a choice: their decision wins over a late answer. */
        var decided = false
    }
    private var lastTyped: Typed? = null
    private var llmPending = false
    private val llmPool = Executors.newCachedThreadPool()

    /** Sentences offered instead of typing, best first; empty while the panel is closed. */
    private var suggestions: List<Candidate> = emptyList()
    private var suggestionRes: Recognizer.Result? = null
    private var suggestionFor = 0
    /** How far the docked bar window grows upward to hold the panel. */
    private var panelExtra = 0
    /**
     * The circle window (x, y, width, height) before the suggestion panel or the Undo satellite
     * widened it; null when not widened.
     */
    private var circleRest: IntArray? = null

    /** The Undo satellite is out (or sliding out); [satDx], [satDy]: its offset from the circle. */
    private var satelliteShown = false
    private var satDx = 0
    private var satDy = 0
    private val retractSatellite = Runnable { hideSatellite(animate = true) }

    /** Overrides the docked rectangle's state colour for a moment, e.g. Undo's orange flash. */
    private var dockFlash: Int? = null
    private val clearDockFlash = Runnable { dockFlash = null; updateDockColors() }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        running = true
        instance = this
        startInForeground()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        buildBubble()
        Recognizer.restoreSettings(this)
        LlmCorrector.restore(this)
        Thread {
            val ok = Recognizer.load(this)
            main.post {
                setState(if (ok) State.IDLE else State.ERROR)
                if (!ok) toast(Recognizer.lastError ?: "Το μοντέλο δεν βρέθηκε")
            }
        }.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { stopSelf(); return START_NOT_STICKY }
        if (intent?.action == ACTION_RELOAD) {
            // a setting or a model changed in the panel: load what is now wanted
            setState(State.LOADING)
            Thread {
                val ok = Recognizer.load(this)
                main.post { setState(if (ok) State.IDLE else State.ERROR) }
            }.start()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        readyFlow.value = false
        if (instance === this) instance = null
        colorAnim?.cancel(); pulseAnim?.cancel(); optionsAnim?.cancel()
        llmPool.shutdownNow()
        main.removeCallbacks(imePoll)
        main.removeCallbacks(longPress)
        main.removeCallbacks(retractSatellite)
        if (state == State.LISTENING) runCatching { recorder.stop() }
        runCatching { wm.removeView(root) }
        super.onDestroy()
    }

    // ------------------------------------------------------------------ bubble

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildBubble() {
        root = LayoutInflater.from(this).inflate(R.layout.overlay_bubble, null) as FrameLayout
        circle = root.findViewById(R.id.circle)
        icon = root.findViewById(R.id.icon)
        pulse = root.findViewById(R.id.pulse)
        spinner = root.findViewById(R.id.spinner)
        suggestionBox = root.findViewById(R.id.suggestions)
        suggestionHeader = root.findViewById(R.id.suggestion_header)
        circle.background = bg
        pulse.background = ring
        panelBg.cornerRadius = dp(20).toFloat()
        suggestionBox.background = panelBg
        dockTrack = root.findViewById(R.id.dock_track)
        undoButton = root.findViewById(R.id.undo_button)
        undoSatellite = root.findViewById(R.id.undo_satellite)
        trackBg.cornerRadius = barHeight() / 2f
        dockTrack.background = trackBg
        undoButton.background = dockButtonBg
        // the floating satellite is its own dark puck, unaffected by the docked bar's styling
        undoSatellite.background = secondaryBg()
        pressFeedback(undoButton)
        pressFeedback(undoSatellite)
        undoButton.setOnClickListener { onUndo() }
        undoSatellite.setOnClickListener { onUndo() }
        applyColor(State.LOADING.color)
        buildStrip()

        params = WindowManager.LayoutParams(
            bubbleSize(),
            bubbleSize(),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = dp(8)
            y = dp(240)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
            // x/y are absolute display coordinates; do not let the WM inset or shrink this window.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                setFitInsetsTypes(0)
            }
        }
        bubbleX = params.x
        bubbleY = params.y
        shapeCircle()

        root.setOnTouchListener(::onTouch)
        root.alpha = 0f
        wm.addView(root, params)
        listenForIme()
        root.animate().alpha(1f).setDuration(220).start()
    }

    /** The quieter round background of the Undo buttons, so the mic stays the main target. */
    private fun secondaryBg() = GradientDrawable().apply {
        shape = GradientDrawable.OVAL
        setColor(0xFF2E3552.toInt())
        setStroke(dp(1), 0x33FFFFFF)
    }

    private fun pressFeedback(v: View) = v.setOnTouchListener { view, e ->
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN ->
                view.animate().scaleX(0.9f).scaleY(0.9f).setStartDelay(0).setDuration(90).start()
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                view.animate().scaleX(1f).scaleY(1f).setStartDelay(0).setDuration(140).start()
        }
        false
    }

    /** Floating circle's own per-state colour, animated by [colorAnim]; docked never reads this. */
    private fun applyColor(c: Int) {
        currentColor = c
        if (!docked) bg.setColor(c)
        ring.setColor(c and 0x00FFFFFF or 0x55000000)
    }

    /** Blue/Red/Orange only - the loading state's muted grey-blue never reaches the rectangle.
     *  [DOCK_ORANGE] equals [State.THINKING]'s own colour, so transcribing looks the same docked
     *  as it does on the floating circle. */
    private fun dockedBgColor(s: State): Int = when (s) {
        State.LISTENING -> DOCK_RED
        State.THINKING, State.ERROR -> DOCK_ORANGE
        else -> DOCK_BLUE
    }

    /**
     * Docked, the mic button, the Undo button and the track behind them are one flat colour for
     * the state (or [dockFlash] while it is set); the icons are never touched, they stay the
     * vector's own solid white. Floating is untouched by this - see [applyColor].
     */
    private fun updateDockColors() {
        if (!docked) return
        val c = dockFlash ?: dockedBgColor(state)
        bg.setColor(c)
        trackBg.setColor(c)
        dockButtonBg.setColor(c)
    }

    private fun setState(s: State) {
        if (state == s) return
        state = s
        readyFlow.value = s != State.LOADING && s != State.ERROR
        colorAnim?.cancel()
        colorAnim = ValueAnimator.ofObject(argb, currentColor, s.color).apply {
            duration = 220
            addUpdateListener { applyColor(it.animatedValue as Int) }
            start()
        }
        // the docked rectangle snaps to its flat colour instead of crossfading through one
        updateDockColors()
        icon.setImageResource(if (s == State.LISTENING) R.drawable.ic_stop else R.drawable.ic_mic)
        icon.visibility = if (s == State.THINKING) View.GONE else View.VISIBLE
        showSpinner()
        root.alpha = restingAlpha()
        if (s == State.LISTENING) startPulse() else stopPulse()
        updateOptions()
    }

    private fun restingAlpha() = if (state == State.LOADING) 0.65f else 1f

    /** The spinner alone while transcribing; around the mic while a late LLM answer is awaited. */
    private fun showSpinner() {
        val on = state == State.THINKING || (state == State.IDLE && llmPending)
        spinner.visibility = if (on) View.VISIBLE else View.GONE
    }

    private fun setLlmPending(on: Boolean) {
        llmPending = on
        showSpinner()
    }

    private fun startPulse() {
        stopPulse()
        pulseAnim = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 1100
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener {
                val f = it.animatedValue as Float
                if (docked) {
                    // the bar window is too short for the ring, so the icon breathes instead
                    pulse.alpha = 0f
                    val s = 1f + 0.14f * (1f - abs(2f * f - 1f))
                    icon.scaleX = s; icon.scaleY = s
                } else {
                    pulse.scaleX = 1f + f * 0.6f
                    pulse.scaleY = 1f + f * 0.6f
                    pulse.alpha = (1f - f) * 0.55f
                    icon.scaleX = 1f; icon.scaleY = 1f
                }
            }
            start()
        }
    }

    private fun stopPulse() {
        pulseAnim?.cancel(); pulseAnim = null
        pulse.alpha = 0f; pulse.scaleX = 1f; pulse.scaleY = 1f
        icon.animate().scaleX(1f).scaleY(1f).setDuration(160).start()
    }

    private fun bump() {
        circle.animate().scaleX(0.88f).scaleY(0.88f).setDuration(90).withEndAction {
            circle.animate().scaleX(1f).scaleY(1f).setDuration(140).start()
        }.start()
    }

    // ------------------------------------------------------------------ keyboard dock

    private fun bubbleSize() = dp(96)
    private fun circleSize() = dp(68)
    private fun barHeight() = dp(56)
    private fun barMargin() = dp(8)
    private fun stripGap() = dp(8)
    private fun chipHeight() = dp(42)
    private fun imeThreshold() = dp(64)
    private fun panelGap() = dp(6)
    private fun edgeMargin() = dp(8)
    /** Clear of the status bar and the navigation bar, as the snapped circle is. */
    private fun safeTop() = dp(48)
    private fun safeBottom() = dp(72)

    @Suppress("DEPRECATION")
    private fun realMetrics() = DisplayMetrics().also { wm.defaultDisplay.getRealMetrics(it) }

    // Same frame the IME insets in keyboardHeight() are measured against.
    private fun screenW(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) wm.currentWindowMetrics.bounds.width()
        else realMetrics().widthPixels

    private fun screenH(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) wm.currentWindowMetrics.bounds.height()
        else realMetrics().heightPixels

    private fun listenForIme() {
        // Insets delivered to this small window are relative to the window, not the screen,
        // so they are only a hint to re-measure; the poll catches changes they miss.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets -> checkIme(); insets }
        main.post(imePoll)
    }

    private val imePoll = object : Runnable {
        override fun run() {
            if (!running) return
            checkIme()
            main.postDelayed(this, 48)
        }
    }

    /**
     * Keyboard height measured up from the bottom edge of the display, 0 when hidden.
     * Needs Android 11: older releases only report the IME relative to this overlay window,
     * which reads 0 as soon as the bar sits above the keyboard.
     */
    private fun keyboardHeight(): Int {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return 0
        val insets = runCatching {
            WindowInsetsCompat.toWindowInsetsCompat(wm.currentWindowMetrics.windowInsets)
        }.getOrNull() ?: return 0
        val type = WindowInsetsCompat.Type.ime()
        return if (insets.isVisible(type)) insets.getInsets(type).bottom else 0
    }

    private fun checkIme() {
        val kb = keyboardHeight()
        if (kb >= imeThreshold()) {
            val wasDocked = docked
            if (!docked) {
                // the bar has its own permanent Undo
                hideSatellite(animate = false)
                // a widened circle window is not where the circle rests
                val rest = circleRest
                bubbleX = rest?.get(0) ?: params.x
                bubbleY = rest?.get(1) ?: params.y
                circleRest = null
                docked = true
                updateDockColors() // the rectangle snaps straight to its state colour
                // Setting the field's text restarts the keyboard, which can read as closed for
                // a poll or two: only a choice hidden for longer than that is stale.
                if (choice != null && SystemClock.uptimeMillis() - hiddenSince > CHOICE_GRACE_MS) {
                    Log.i(TAG, "choices dropped: the keyboard was closed")
                    choice = null
                }
            }
            // also re-run while docked: the keyboard can change height (emoji panel, rotation)
            imeHeight = kb
            if (suggestions.isNotEmpty()) { updateSuggestionHeader(); expandDocked() } else shapeBar()
            if (!wasDocked) { morphIn(); updateOptions() }
        } else if (docked) {
            docked = false
            imeHeight = 0
            panelExtra = 0
            hiddenSince = SystemClock.uptimeMillis()
            applyColor(currentColor) // the floating circle returns to its own colour
            updateOptions()
            shapeCircle()
            if (suggestions.isNotEmpty()) { updateSuggestionHeader(); expandCircle() }
            morphIn()
        }
    }

    /** Softens the instant window swap between circle and bar. */
    private fun morphIn() {
        root.alpha = 0f
        root.animate().alpha(restingAlpha()).setDuration(200).setInterpolator(ease).start()
        circle.scaleX = 0.94f; circle.scaleY = 0.94f
        circle.animate().scaleX(1f).scaleY(1f).setStartDelay(0).setDuration(260).setInterpolator(ease).start()
    }

    private fun shapeBar() {
        val w = screenW() - 2 * barMargin()
        val h = barHeight()
        // the suggestion panel adds height above the bar; the bar itself stays on the keyboard
        val y = (screenH() - imeHeight - h - panelExtra).coerceAtLeast(0)
        placeWindow(barMargin(), y, w, h + panelExtra)

        // two big targets in the far corners (Fitts), nothing tappable in between
        shapeInner(h, h, h / 2f, Gravity.END or Gravity.BOTTOM)
        dockTrack.visibility = View.VISIBLE
        undoButton.visibility = View.VISIBLE
        updateUndoButton()

        // the word-choice strip opens in the empty middle, between the two buttons
        val p = optionsProgress
        val sw = (w - 2 * h - 2 * stripGap()).coerceAtLeast(0)
        val lp = strip.layoutParams as FrameLayout.LayoutParams
        if (lp.width != sw || lp.leftMargin != h + stripGap()) {
            lp.width = sw
            lp.leftMargin = h + stripGap()
            strip.layoutParams = lp
        }
        strip.alpha = p
        strip.translationX = (1f - p) * dp(32)
        strip.visibility = if (p > 0f) View.VISIBLE else View.GONE
    }

    private fun shapeCircle() {
        val s = bubbleSize()
        // clamp in case the screen rotated while docked
        val x = bubbleX.coerceIn(0, (screenW() - s).coerceAtLeast(0))
        val y = bubbleY.coerceIn(0, (screenH() - s).coerceAtLeast(0))
        placeWindow(x, y, s, s)
        shapeInner(circleSize(), circleSize(), circleSize() / 2f, Gravity.CENTER)
        strip.visibility = View.GONE
        dockTrack.visibility = View.GONE
        undoButton.visibility = View.GONE
    }

    /** Moves and resizes the circle window, keeping the mic at its resting spot on screen. */
    private fun widenCircle(x: Int, y: Int, w: Int, h: Int) {
        val (rx, ry, rw, rh) = restingCircle().also { circleRest = it }
        placeWindow(x, y, w, h)
        val cs = circleSize()
        shapeInner(cs, cs, cs / 2f, Gravity.TOP or Gravity.START, rx - x + (rw - cs) / 2, ry - y + (rh - cs) / 2)
    }

    /** The circle window as it is when nothing has widened it. */
    private fun restingCircle() = circleRest ?: intArrayOf(params.x, params.y, params.width, params.height)

    /** Puts the circle window back exactly as it was before it was widened. */
    private fun unwidenCircle() {
        val rest = circleRest ?: return
        circleRest = null
        placeWindow(rest[0], rest[1], rest[2], rest[3])
        shapeInner(circleSize(), circleSize(), circleSize() / 2f, Gravity.CENTER)
    }

    private fun placeWindow(x: Int, y: Int, w: Int, h: Int) {
        if (params.x == x && params.y == y && params.width == w && params.height == h) return
        params.x = x
        params.y = y
        params.width = w
        params.height = h
        if (root.isAttachedToWindow) runCatching { wm.updateViewLayout(root, params) }
    }

    /** [left], [top]: offsets for a pinned (TOP|START) mic in a window widened around it. */
    private fun shapeInner(w: Int, h: Int, radius: Float, gravity: Int, left: Int = 0, top: Int = 0) {
        val lp = circle.layoutParams as FrameLayout.LayoutParams
        if (lp.width != w || lp.height != h || lp.gravity != gravity || lp.leftMargin != left || lp.topMargin != top) {
            lp.width = w
            lp.height = h
            lp.gravity = gravity
            lp.leftMargin = left
            lp.topMargin = top
            circle.layoutParams = lp
        }
        bg.cornerRadius = radius
        // the ring pulses around the mic wherever it is pinned
        val plp = pulse.layoutParams as FrameLayout.LayoutParams
        if (plp.gravity != gravity || plp.leftMargin != left || plp.topMargin != top) {
            plp.gravity = gravity
            plp.leftMargin = left
            plp.topMargin = top
            pulse.layoutParams = plp
        }
    }

    // ------------------------------------------------------------------ word choices

    /** The sentence as typed and the unsure spots still to offer, left to right. */
    private class Choice(
        var words: List<String>,
        var typed: String,
        val spots: List<WordChoices.Spot>,
    ) {
        var index = 0
        /** How far earlier picks moved the later spots: a pick can change the word count. */
        var shift = 0
        val spot: WordChoices.Spot get() = spots[index]
    }

    private var choice: Choice? = null

    private fun buildStrip() {
        chips = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            val pad = (barHeight() - chipHeight()) / 2
            setPadding(pad, 0, pad, 0)
        }

        stripBg.cornerRadius = barHeight() / 2f
        strip = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
            background = stripBg
            clipToOutline = true
            elevation = dp(10).toFloat()
            alpha = 0f
            visibility = View.GONE
            addView(chips, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
        root.addView(
            strip,
            FrameLayout.LayoutParams(0, barHeight(), Gravity.START or Gravity.BOTTOM),
        )
    }

    /** [current]: what the field holds now, drawn in the bubble's colour with a tick. */
    private fun makeChip(label: String, current: Boolean, tick: Boolean = current, onClick: () -> Unit) = TextView(this).apply {
        text = if (tick) "✓ $label" else label
        setTextColor(Color.WHITE)
        textSize = 16f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        maxLines = 1
        setPadding(dp(18), 0, dp(18), 0)
        background = GradientDrawable().apply {
            cornerRadius = chipHeight() / 2f
            if (current) {
                setColor(State.IDLE.color)
            } else {
                setColor(0xFF2E3552.toInt())
                setStroke(dp(1), 0x33FFFFFF)
            }
        }
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, chipHeight()).apply {
            marginStart = dp(3); marginEnd = dp(3)
        }
        isClickable = true
        setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN ->
                    v.animate().scaleX(0.92f).scaleY(0.92f).setStartDelay(0).setDuration(90).start()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                    v.animate().scaleX(1f).scaleY(1f).setStartDelay(0).setDuration(160).start()
            }
            false
        }
        setOnClickListener { onClick() }
    }

    /** "1/3" before the chips when more than one word is unsure. */
    private fun makeCounter(label: String) = TextView(this).apply {
        text = label
        setTextColor(0x99FFFFFF.toInt())
        textSize = 13f
        gravity = Gravity.CENTER
        setPadding(dp(8), 0, dp(6), 0)
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, chipHeight())
    }

    private fun fillChips(c: Choice) {
        chips.removeAllViews()
        if (c.spots.size > 1) chips.addView(makeCounter("${c.index + 1}/${c.spots.size}"))
        c.spot.options.forEachIndexed { i, o ->
            chips.addView(makeChip(o.text.ifEmpty { "(τίποτα)" }, current = i == 0) { onChoose(i) })
        }
    }

    /**
     * After a sentence was typed: offer its unsure words, if there are any. [forced]: the
     * phone's own sentence when the LLM's was typed instead; where they differ is always asked.
     */
    private fun offerChoices(words: List<String>, typed: String, candidates: List<Candidate>, forced: List<String>?) {
        val spots = WordChoices.find(words, candidates, forced = forced)
        Log.i(TAG, "choices: ${spots.size} spot(s)" + spots.joinToString("") { s ->
            " [" + s.options.joinToString(" | ") { "${it.text} ${(it.probability * 100).toInt()}%" } + "]"
        } + (if (forced != null) ", LLM differs" else "") + (if (docked) "" else ", keyboard closed"))
        if (spots.isEmpty()) { dismissChoice(); return }
        val c = Choice(words, typed, spots)
        choice = c
        if (!docked) hiddenSince = SystemClock.uptimeMillis()
        fillChips(c)
        if (optionsShown) revealChips() else updateOptions()
    }

    /** [i] 0 keeps what was typed; any other option rewrites the spot in the field. */
    private fun onChoose(i: Int) {
        val c = choice ?: return
        lastTyped?.decided = true
        val spot = c.spot
        if (i > 0) {
            val a = spot.start + c.shift
            val b = spot.end + c.shift
            val picked = spot.options[i].words
            val words = c.words.subList(0, a) + picked + c.words.subList(b, c.words.size)
            val text = words.joinToString(" ")
            // the sentence already in the field is overwritten as a whole, never appended to
            val where = TypingAccessibilityService.replaceInjected(this, c.typed, text)
            if (where == "clipboard") toast("Αντιγράφηκε: $text")
            Log.i(TAG, "[choice] $where: ${c.typed} -> $text")
            c.shift += picked.size - (spot.end - spot.start)
            c.words = words
            c.typed = text
        }
        c.index++
        if (c.index < c.spots.size) swapChips(c) else dismissChoice()
    }

    /** The next unsure word: the old chips slide out to the left, the new ones in from the right. */
    private fun swapChips(c: Choice) {
        // the outgoing chips still belong to the previous spot; a second tap must not land on them
        for (i in 0 until chips.childCount) chips.getChildAt(i).isEnabled = false
        chips.animate().alpha(0f).translationX(-dp(24).toFloat())
            .setStartDelay(0).setDuration(150).setInterpolator(ease)
            .withEndAction {
                if (choice !== c) return@withEndAction
                fillChips(c)
                revealChips()
            }.start()
    }

    private fun dismissChoice() {
        choice = null
        updateOptions()
    }

    private fun updateOptions() {
        val show = docked && state == State.IDLE && choice != null
        if (show == optionsShown) return
        optionsShown = show
        optionsAnim?.cancel()
        if (!docked) {
            // the bar is gone, nothing to animate back
            optionsAnim = null
            optionsProgress = 0f
            return
        }
        optionsAnim = ValueAnimator.ofFloat(optionsProgress, if (show) 1f else 0f).apply {
            duration = if (show) 420 else 280
            interpolator = ease
            addUpdateListener {
                optionsProgress = it.animatedValue as Float
                if (docked) shapeBar()
            }
            start()
        }
        if (show) revealChips()
    }

    private fun revealChips() {
        strip.scrollTo(0, 0)
        chips.animate().cancel()
        chips.alpha = 1f
        chips.translationX = 0f
        for (i in 0 until chips.childCount) {
            val c = chips.getChildAt(i)
            c.animate().cancel()
            c.alpha = 0f
            c.translationX = dp(28).toFloat()
            c.scaleX = 0.9f; c.scaleY = 0.9f
            c.animate().alpha(1f).translationX(0f).scaleX(1f).scaleY(1f)
                .setStartDelay(120L + i * 50L).setDuration(340).setInterpolator(ease).start()
        }
    }

    // ------------------------------------------------------------------ sentence suggestions

    /** True when the best sentence is too doubtful to type without asking. */
    private fun isUnsure(res: Recognizer.Result): Boolean {
        val top = res.candidates.firstOrNull() ?: return false
        return top.probability < CONFIDENCE_THRESHOLD && res.candidates.size >= 2
    }

    private fun makeSuggestion(c: Candidate, best: Boolean, onClick: () -> Unit) =
        makeChip(c.text, current = best, tick = false, onClick = onClick).apply {
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            gravity = Gravity.START or Gravity.CENTER_VERTICAL
            minHeight = chipHeight()
            setPadding(dp(18), dp(8), dp(18), dp(8))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(3); bottomMargin = dp(3)
            }
        }

    /** Instead of typing: fill the panel with the best sentences and grow the window to fit it. */
    private fun showSuggestions(res: Recognizer.Result, n: Int) {
        dismissChoice()
        hideSatellite(animate = false)
        closeSuggestions()
        suggestions = res.candidates.take(MAX_SUGGESTIONS)
        suggestionRes = res
        suggestionFor = n
        suggestions.forEachIndexed { i, c ->
            suggestionBox.addView(makeSuggestion(c, best = i == 0) { onSuggestion(c, n) })
        }
        // the header goes first, so it must be sized in before the panel is measured below
        updateSuggestionHeader()
        suggestionBox.visibility = View.VISIBLE
        if (docked) expandDocked() else expandCircle()

        suggestionBox.animate().cancel()
        suggestionBox.alpha = 0f
        suggestionBox.translationY = if (docked) dp(12).toFloat() else 0f
        suggestionBox.animate().alpha(1f).translationY(0f)
            .setStartDelay(0).setDuration(240).setInterpolator(ease).start()
    }

    /** The panel's size for at most [maxW] x [maxH]; [fill] makes it exactly [maxW] wide. */
    private fun measurePanel(maxW: Int, maxH: Int, fill: Boolean): Pair<Int, Int> {
        suggestionBox.measure(
            View.MeasureSpec.makeMeasureSpec(maxW, if (fill) View.MeasureSpec.EXACTLY else View.MeasureSpec.AT_MOST),
            View.MeasureSpec.makeMeasureSpec(maxH, View.MeasureSpec.AT_MOST),
        )
        return suggestionBox.measuredWidth to suggestionBox.measuredHeight.coerceAtMost(maxH)
    }

    /**
     * Floating, since nothing was typed for the person to read: the header carries the full top
     * guess above the chips. Docked, the top guess is already in the field, so it stays hidden.
     */
    private fun updateSuggestionHeader() {
        val top = suggestions.firstOrNull()
        if (!docked && top != null) {
            suggestionHeader.text = top.text
            suggestionHeader.visibility = View.VISIBLE
        } else {
            suggestionHeader.visibility = View.GONE
        }
    }

    private fun placePanel(w: Int, h: Int, left: Int, top: Int) {
        val lp = suggestionBox.layoutParams as FrameLayout.LayoutParams
        if (lp.width != w || lp.height != h || lp.leftMargin != left || lp.topMargin != top ||
            lp.gravity != (Gravity.TOP or Gravity.START)
        ) {
            lp.width = w
            lp.height = h
            lp.leftMargin = left
            lp.topMargin = top
            lp.gravity = Gravity.TOP or Gravity.START
            suggestionBox.layoutParams = lp
        }
    }

    /**
     * Docked: the panel sits above the bar at full bar width and the window grows upward only,
     * so the bar stays on the keyboard; the panel's height is capped below the status bar.
     */
    private fun expandDocked() {
        val w = screenW() - 2 * barMargin()
        val room = (screenH() - imeHeight - barHeight() - panelGap() - safeTop()).coerceAtLeast(0)
        val (_, ph) = measurePanel(w, room, fill = true)
        placePanel(w, ph, 0, 0)
        panelExtra = if (ph > 0) ph + panelGap() else 0
        shapeBar()
    }

    /**
     * Floating: the panel opens on the side of the circle with more room, centred on it. The
     * window widens and moves, but the mic is pinned inside it at its old screen position, so
     * the circle itself neither drifts nor jumps; the window never leaves the screen.
     */
    private fun expandCircle() {
        val (rx, ry, rw, rh) = restingCircle()
        val sw = screenW()
        val sh = screenH()
        val gap = panelGap()

        val roomRight = (sw - edgeMargin() - (rx + rw) - gap).coerceAtLeast(0)
        val roomLeft = (rx - edgeMargin() - gap).coerceAtLeast(0)
        val onRight = roomRight >= roomLeft
        val maxW = minOf(if (onRight) roomRight else roomLeft, dp(300))
        val maxH = (sh - safeTop() - safeBottom()).coerceAtLeast(rh)
        val (pw, ph) = measurePanel(maxW, maxH, fill = false)

        val w = rw + gap + pw
        val h = maxOf(rh, ph)
        // the room above already keeps the window between the side margins
        val x = if (onRight) rx else rx - gap - pw
        // centred on the circle, kept inside the safe band, and always containing the circle
        val lo = minOf(safeTop(), ry).coerceAtLeast(0)
        val hi = (maxOf(sh - safeBottom(), ry + rh).coerceAtMost(sh) - h).coerceAtLeast(lo)
        val y = (ry + rh / 2 - h / 2).coerceIn(lo, hi)
        widenCircle(x, y, w, h)

        // where the old circle window now lies inside the widened one
        val cx = rx - x
        val cy = ry - y
        val px = if (onRight) cx + rw + gap else cx - gap - pw
        val py = (cy + rh / 2 - ph / 2).coerceIn(0, (h - ph).coerceAtLeast(0))
        placePanel(pw, ph, px, py)
    }

    /** A sentence was tapped: type it, empty the panel and shrink the window back. */
    private fun onSuggestion(c: Candidate, n: Int) {
        if (suggestions.isEmpty() || n != suggestionFor || n != dictation) return
        val res = suggestionRes
        val text = c.text
        val where = TypingAccessibilityService.deliver(this, text)
        closeSuggestions()
        if (where == "clipboard") toast("Αντιγράφηκε: $text")
        Log.i(TAG, "[suggestion] $where: $text (${(c.probability * 100).toInt()}%)")
        if (where != "typed") return
        // the person chose: a late LLM answer for this dictation must not overwrite it
        if (res != null) lastTyped = Typed(n, res, c.words).apply { decided = true }
        updateUndoButton()
        showSatellite()
    }

    /** Empties the panel and puts the window back exactly as it was before it grew. */
    private fun closeSuggestions() {
        if (suggestions.isEmpty()) return
        suggestions = emptyList()
        suggestionRes = null
        suggestionBox.animate().cancel()
        suggestionBox.alpha = 1f
        suggestionBox.translationY = 0f
        // the header (child 0) stays put, ready for next time; only the chips are thrown away
        for (i in suggestionBox.childCount - 1 downTo 1) suggestionBox.removeViewAt(i)
        suggestionHeader.visibility = View.GONE
        suggestionBox.visibility = View.GONE
        if (docked) {
            panelExtra = 0
            shapeBar()
        } else {
            unwidenCircle()
        }
    }

    // ------------------------------------------------------------------ undo

    private fun satelliteSize() = dp(48)

    /**
     * After text was typed, floating: the Undo bubble slides out from behind the circle toward
     * the middle of the screen and rests just clear of it, then retracts on its own.
     */
    private fun showSatellite() {
        if (docked || suggestions.isNotEmpty()) return
        main.removeCallbacks(retractSatellite)
        undoSatellite.animate().cancel()

        val (rx, ry, rw, rh) = restingCircle()
        val sw = screenW()
        val sh = screenH()
        val ccx = rx + rw / 2
        val ccy = ry + rh / 2
        val half = satelliteSize() / 2
        val dist = circleSize() / 2 + half + dp(6)
        // toward the screen's middle, and upward unless that would reach under the status bar
        val sx = if (ccx < sw / 2) 1 else -1
        val sy = if (ccy - dist - half < safeTop()) 1 else -1
        satDx = (dist * 0.8f).toInt() * sx
        satDy = (dist * 0.6f).toInt() * sy
        val scx = ccx + satDx
        val scy = ccy + satDy

        // the smallest window holding both the resting circle and the satellite, kept on screen
        val left = minOf(rx, scx - half)
        val top = minOf(ry, scy - half)
        val w = maxOf(rx + rw, scx + half) - left
        val h = maxOf(ry + rh, scy + half) - top
        val x = left.coerceIn(0, (sw - w).coerceAtLeast(0))
        val y = top.coerceIn(0, (sh - h).coerceAtLeast(0))
        widenCircle(x, y, w, h)

        val lp = undoSatellite.layoutParams as FrameLayout.LayoutParams
        lp.gravity = Gravity.TOP or Gravity.START
        lp.leftMargin = scx - half - x
        lp.topMargin = scy - half - y
        undoSatellite.layoutParams = lp

        // start tucked under the circle's centre, then slide out to rest
        if (!satelliteShown) {
            undoSatellite.translationX = -satDx.toFloat()
            undoSatellite.translationY = -satDy.toFloat()
            undoSatellite.scaleX = 0.5f; undoSatellite.scaleY = 0.5f
            undoSatellite.alpha = 0f
        }
        satelliteShown = true
        undoSatellite.isEnabled = true
        undoSatellite.visibility = View.VISIBLE
        undoSatellite.animate().translationX(0f).translationY(0f).scaleX(1f).scaleY(1f).alpha(1f)
            .setStartDelay(60).setDuration(320).setInterpolator(ease).start()
        main.postDelayed(retractSatellite, SATELLITE_MS)
    }

    /** Tucks the satellite back behind the circle; [animate] off hides it at once. */
    private fun hideSatellite(animate: Boolean) {
        main.removeCallbacks(retractSatellite)
        if (!satelliteShown && undoSatellite.visibility != View.VISIBLE) return
        satelliteShown = false
        // a second tap while it slides back must not undo twice
        undoSatellite.isEnabled = false
        undoSatellite.animate().cancel()
        if (!animate) { stowSatellite(); return }
        undoSatellite.animate()
            .translationX(-satDx.toFloat()).translationY(-satDy.toFloat())
            .scaleX(0.5f).scaleY(0.5f).alpha(0f)
            .setStartDelay(0).setDuration(200).setInterpolator(ease)
            .withEndAction { stowSatellite() }
            .start()
    }

    private fun stowSatellite() {
        undoSatellite.visibility = View.GONE
        undoSatellite.translationX = 0f; undoSatellite.translationY = 0f
        undoSatellite.scaleX = 1f; undoSatellite.scaleY = 1f
        undoSatellite.alpha = 1f
        if (!docked && suggestions.isEmpty()) unwidenCircle()
    }

    /** Either Undo button: takes the last typed phrase back out of the field. */
    private fun onUndo() {
        hideSatellite(animate = false)
        // docked, the whole rectangle flashes orange for a moment as the warning/undo colour
        if (docked) {
            main.removeCallbacks(clearDockFlash)
            dockFlash = DOCK_ORANGE
            updateDockColors()
            main.postDelayed(clearDockFlash, UNDO_FLASH_MS)
        }
        when (TypingAccessibilityService.undoLastInjection()) {
            "undone" -> {
                // the open word choice and any late LLM answer were about the removed sentence
                dismissChoice()
                lastTyped = null
                setLlmPending(false)
            }
            "missing" -> toast("Το κείμενο δεν βρέθηκε πια στο πεδίο")
            else -> toast("Τίποτα για αναίρεση")
        }
        if (docked) shapeBar()
    }

    /** The docked Undo button is dimmed while there is nothing of ours to take back. */
    private fun updateUndoButton() {
        undoButton.alpha = if (TypingAccessibilityService.canUndo) 1f else 0.45f
    }

    // ------------------------------------------------------------------ touch

    private fun onMic(e: MotionEvent) =
        e.x >= circle.left && e.x < circle.right && e.y >= circle.top && e.y < circle.bottom

    private var downX = 0f
    private var downY = 0f
    private var startX = 0
    private var startY = 0
    private var moved = false
    private val longPress = Runnable { longPressed = true; bump(); onLongPress() }

    private fun onTouch(v: View, e: MotionEvent): Boolean {
        val slop = ViewConfiguration.get(this).scaledTouchSlop
        // the bar's middle and a widened window's spare room are dead space: only the mic answers
        if (e.actionMasked == MotionEvent.ACTION_DOWN && (docked || circleRest != null) && !onMic(e)) return false
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = e.rawX; downY = e.rawY; startX = params.x; startY = params.y
                moved = false; longPressed = false
                circle.animate().scaleX(0.9f).scaleY(0.9f).setDuration(90).start()
                main.postDelayed(longPress, ViewConfiguration.getLongPressTimeout().toLong())
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = e.rawX - downX
                val dy = e.rawY - downY
                if (docked || suggestions.isNotEmpty()) {
                    // the bar stays pinned to the keyboard, and an open panel pins the circle
                    if (abs(dx) > slop || abs(dy) > slop) main.removeCallbacks(longPress)
                } else if (moved || abs(dx) > slop || abs(dy) > slop) {
                    if (!moved && circleRest != null) {
                        // the satellite widened the window: tuck it away and drag the bare circle
                        hideSatellite(animate = false)
                        startX = params.x; startY = params.y
                        downX = e.rawX; downY = e.rawY
                    }
                    moved = true
                    main.removeCallbacks(longPress)
                    params.x = startX + (e.rawX - downX).toInt()
                    params.y = startY + (e.rawY - downY).toInt()
                    runCatching { wm.updateViewLayout(root, params) }
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                main.removeCallbacks(longPress)
                circle.animate().scaleX(1f).scaleY(1f).setDuration(140).start()
                if (moved) { if (!docked) snapToEdge() }
                else if (!longPressed && e.actionMasked == MotionEvent.ACTION_UP) { v.performClick(); bump(); onTap() }
            }
        }
        return true
    }

    /** Jump to the nearer side and stay clear of the screen edges. */
    private fun snapToEdge() {
        val sw = screenW()
        val sh = screenH()
        val w = params.width
        val h = params.height
        params.x = if (params.x + w / 2 < sw / 2) dp(8) else sw - w - dp(8)
        params.y = params.y.coerceIn(dp(48), (sh - h - dp(72)).coerceAtLeast(dp(48)))
        runCatching { wm.updateViewLayout(root, params) }
    }

    // ------------------------------------------------------------------ actions

    private fun onTap() {
        hideSatellite(animate = false)
        when (state) {
            State.LOADING -> toast("Φορτώνει το μοντέλο...")
            State.ERROR -> toast(Recognizer.lastError ?: "Το μοντέλο δεν φορτώθηκε")
            // speaking again keeps what was typed and closes any open choice (and a late answer)
            State.IDLE -> runCatching {
                choice = null
                closeSuggestions()
                dictation++
                setLlmPending(false)
                recorder.start()
                LlmCorrector.warmUp(this)
                setState(State.LISTENING)
            }.onFailure { toast("Μικρόφωνο: $it"); updateOptions() }
            State.LISTENING -> transcribe(recorder.stop(), "mic")
            State.THINKING -> {}
        }
    }

    /** Whether push-to-talk started the current recording. */
    private var pttActive = false

    /**
     * Push-to-talk with the volume keys, from [TypingAccessibilityService.onKeyEvent] (main
     * thread): key down starts listening as a tap on the bubble does, key up recognises.
     * [heldMs] is the key's own down-to-up time (not when the events reached us). A press
     * shorter than [PTT_MIN_MS], or one that captured less than that much audio, is dropped
     * without typing anything. While the bubble is loading or failed the keys stay volume keys.
     * Returns whether the key was used (then the volume does not change).
     */
    fun pushToTalk(down: Boolean, repeat: Boolean, heldMs: Long = 0L): Boolean {
        if (state == State.LOADING || state == State.ERROR) return false
        if (repeat) return true
        if (down) {
            if (state == State.IDLE) {
                onTap()
                pttActive = state == State.LISTENING
                Log.i(TAG, "push-to-talk down, listening=$pttActive")
            }
        } else if (pttActive) {
            pttActive = false
            if (state == State.LISTENING) {
                val pcm = runCatching { recorder.stop() }.getOrNull()
                val audioMs = (pcm?.size ?: 0) * 1000L / AudioRecorder.SAMPLE_RATE
                Log.i(TAG, "push-to-talk up: held $heldMs ms, audio $audioMs ms")
                if (pcm == null || heldMs < PTT_MIN_MS || audioMs < PTT_MIN_MS) {
                    setState(State.IDLE)
                    updateOptions()
                    toast("Κράτα πατημένο το πλήκτρο έντασης όσο μιλάς.")
                } else transcribe(pcm, "ptt")
            }
        }
        return true
    }

    private fun onLongPress() {
        if (state != State.IDLE) return
        val f = Recognizer.testWav(this)
        if (!f.exists()) { toast("Δεν υπάρχει test.wav"); return }
        transcribe(AudioRecorder.readWav(f), f.name)
    }

    private fun transcribe(wav: FloatArray, id: String) {
        choice = null
        hideSatellite(animate = false)
        closeSuggestions()
        val n = ++dictation
        setLlmPending(false)
        if (wav.size < AudioRecorder.SAMPLE_RATE / 5) { setState(State.IDLE); toast("Πολύ σύντομο"); return }
        setState(State.THINKING)
        Thread {
            // what is already written in the field: layer 2 continues it (read on this thread,
            // before the result is typed, so it never includes the new words)
            val context = TypingAccessibilityService.contextBefore()
            val r = runCatching { Recognizer.recognize(wav, id, context) }
            // layer 3 only when switched on and online; on any failure its words are empty
            val asked = r.getOrNull()
                ?.takeIf { it.text.isNotBlank() && LlmCorrector.shouldRun(this) }
                ?.let { res -> runCatching { llmPool.submit(Callable { LlmCorrector.refine(res) }) }.getOrNull() }
            // a quick answer is typed straight away; a slow one corrects the typed text later
            val quick = asked?.let { runCatching { it.get(LLM_WAIT_MS, TimeUnit.MILLISECONDS) }.getOrNull() }
            val late = if (quick == null) asked else null
            main.post { typeResult(r, quick, id, n, waiting = late != null) }
            if (late != null) {
                val o = runCatching { late.get() }.getOrNull()
                main.post { applyLate(o, n) }
            }
        }.start()
    }

    private fun typeResult(r: Result<Recognizer.Result>, llm: LlmCorrector.Outcome?, id: String, n: Int, waiting: Boolean) {
        setState(State.IDLE)
        r.onSuccess { res ->
            // docked, the top guess always goes straight into the field (it can be undone or
            // corrected in place); only floating, with nowhere to show the sentence, does a
            // doubtful one wait for the person to pick from the panel instead of being typed
            if (llm?.words.isNullOrEmpty() && !docked && isUnsure(res)) {
                Log.i(TAG, "[$id] unsure (${(res.candidates.first().probability * 100).toInt()}%), asking: " +
                    res.candidates.take(MAX_SUGGESTIONS).joinToString(" | ") { it.text } +
                    (if (waiting) ", late LLM answer will be ignored" else ""))
                showSuggestions(res, n)
                return@onSuccess
            }
            val local = res.candidates.firstOrNull()?.words
                ?: res.text.split(' ').filter { it.isNotBlank() }
            val words = llm?.words?.takeIf { it.isNotEmpty() } ?: local
            val text = words.joinToString(" ")
            val where = TypingAccessibilityService.deliver(this, text)
            when {
                text.isBlank() -> toast("Δεν αναγνωρίστηκε τίποτα")
                where == "clipboard" -> toast("Αντιγράφηκε: $text")
                // docked, the text is in the field right above and a toast would cover the choices
                !docked -> toast(text)
            }
            val via = when {
                llm != null -> " via ${llm.provider.label} ${llm.ms} ms" + (llm.error?.let { " (kept: $it)" } ?: "")
                waiting -> ", ${LlmCorrector.provider.label} slower than $LLM_WAIT_MS ms, waiting"
                else -> ""
            }
            Log.i(TAG, "[$id] $where: $text (${res.inferenceMs} ms$via)")
            if (where != "typed") return@onSuccess
            lastTyped = Typed(n, res, words)
            updateUndoButton()
            showSatellite()
            setLlmPending(waiting)
            offerChoices(words, text, res.candidates, forced = local.takeIf { it != words })
        }.onFailure { toast("Σφάλμα: $it") }
    }

    /** A slow LLM answer arrived: it replaces the typed sentence unless the person moved on. */
    private fun applyLate(o: LlmCorrector.Outcome?, n: Int) {
        if (n != dictation) { Log.i(TAG, "[late] dropped, a new dictation started"); return }
        setLlmPending(false)
        val t = lastTyped?.takeIf { it.id == n } ?: return
        val words = o?.words.orEmpty()
        val text = words.joinToString(" ")
        val via = o?.let { "${it.provider.label} ${it.ms} ms" } ?: "LLM"
        val skip = when {
            words.isEmpty() -> "kept the phone's: ${o?.error ?: "no answer"}"
            text == t.text -> "agrees with the phone"
            t.decided -> "dropped, a choice was already tapped: $text"
            else -> null
        }
        if (skip != null) { Log.i(TAG, "[late via $via] $skip"); return }
        val where = TypingAccessibilityService.replaceInjected(this, t.text, text, copyIfMissing = false)
        Log.i(TAG, "[late via $via] $where: ${t.text} -> $text")
        if (where != "typed") return
        lastTyped = Typed(n, t.res, words)
        offerChoices(words, text, t.res.candidates, forced = t.words)
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    // ------------------------------------------------------------------ foreground notification

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, getString(R.string.notif_channel), NotificationManager.IMPORTANCE_LOW)
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, OverlayService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val n: Notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_mic)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(getString(R.string.notif_text))
            .addAction(Notification.Action.Builder(null, "Απόκρυψη", stop).build())
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    companion object {
        private const val TAG = "OverlayService"
        private const val CHANNEL = "overlay"
        private const val NOTIF_ID = 1001
        const val ACTION_STOP = "com.openhackathon.voicetotext.STOP"
        const val ACTION_RELOAD = "com.openhackathon.voicetotext.RELOAD"
        /** How long the bubble waits for the LLM before typing the phone's sentence. */
        private const val LLM_WAIT_MS = 2_500L
        /** A keyboard gone for less than this (restarting after the text changed) keeps the choice. */
        private const val CHOICE_GRACE_MS = 1_500L
        /** Below this the best sentence is not typed; the best [MAX_SUGGESTIONS] are offered. */
        private const val CONFIDENCE_THRESHOLD = 0.85f
        private const val MAX_SUGGESTIONS = 3
        /** How long the floating Undo bubble stays out if it is not touched. */
        private const val SATELLITE_MS = 5_000L
        /** The docked rectangle's three flat colours - Default/Ready, Recording, Undo/Warning. */
        private const val DOCK_BLUE = 0xFF3D5FE0.toInt()
        private const val DOCK_RED = 0xFFE5484D.toInt()
        private const val DOCK_ORANGE = 0xFFF2820D.toInt()
        /** How long the rectangle's orange Undo flash lasts before it reverts to the state colour. */
        private const val UNDO_FLASH_MS = 260L
        @Volatile var running = false
        private val readyFlow = MutableStateFlow(false)
        /** True once the bubble has its model loaded and can listen; false while loading, failed or off. */
        val ready: StateFlow<Boolean> = readyFlow.asStateFlow()
        /** The running bubble, for push-to-talk from the accessibility service. */
        @Volatile var instance: OverlayService? = null
        /** A volume-key press (or its audio) shorter than this is not a dictation. */
        private const val PTT_MIN_MS = 1_500L
    }
}
