package com.openhackathon.voicetotext.service

import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.R
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
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Toast
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.math.abs

/**
 * The floating microphone. Lives above every app as a foreground service.
 *
 * Tap to listen (the circle turns red and pulses), tap again to transcribe (amber, spinner),
 * and the text is typed into whatever field has focus. Long press runs the pushed test.wav
 * instead of the mic. Drag it anywhere; it snaps to the nearer side when released.
 * When the software keyboard is open, the circle is swapped for a bar docked on top of it,
 * and swapped back to its previous position when the keyboard closes.
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
    private lateinit var params: WindowManager.LayoutParams

    private val bg = GradientDrawable().apply { shape = GradientDrawable.RECTANGLE }
    private val ring = GradientDrawable().apply { shape = GradientDrawable.OVAL }
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

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        running = true
        startInForeground()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        buildBubble()
        Recognizer.restoreEngine(this)
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
            // the user switched engine in the panel: drop the old one and load the new one
            setState(State.LOADING)
            Thread {
                Recognizer.releaseUnused()
                val ok = Recognizer.load(this)
                main.post { setState(if (ok) State.IDLE else State.ERROR) }
            }.start()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        colorAnim?.cancel(); pulseAnim?.cancel()
        main.removeCallbacks(imePoll)
        main.removeCallbacks(longPress)
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
        circle.background = bg
        pulse.background = ring
        applyColor(State.LOADING.color)

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

    private fun applyColor(c: Int) {
        currentColor = c
        bg.setColor(c)
        ring.setColor(c and 0x00FFFFFF or 0x55000000)
    }

    private fun setState(s: State) {
        if (state == s) return
        state = s
        colorAnim?.cancel()
        colorAnim = ValueAnimator.ofObject(argb, currentColor, s.color).apply {
            duration = 220
            addUpdateListener { applyColor(it.animatedValue as Int) }
            start()
        }
        icon.setImageResource(if (s == State.LISTENING) R.drawable.ic_stop else R.drawable.ic_mic)
        icon.visibility = if (s == State.THINKING) View.GONE else View.VISIBLE
        spinner.visibility = if (s == State.THINKING) View.VISIBLE else View.GONE
        root.alpha = if (s == State.LOADING) 0.65f else 1f
        if (s == State.LISTENING) startPulse() else stopPulse()
    }

    private fun startPulse() {
        stopPulse()
        pulseAnim = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 1100
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener {
                val f = it.animatedValue as Float
                pulse.scaleX = 1f + f * 0.6f
                pulse.scaleY = 1f + f * 0.6f
                pulse.alpha = (1f - f) * 0.55f
            }
            start()
        }
    }

    private fun stopPulse() {
        pulseAnim?.cancel(); pulseAnim = null
        pulse.alpha = 0f; pulse.scaleX = 1f; pulse.scaleY = 1f
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
    private fun barRadius() = dp(16).toFloat()
    private fun imeThreshold() = dp(64)

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
            if (!docked) {
                bubbleX = params.x
                bubbleY = params.y
                docked = true
            }
            // also re-run while docked: the keyboard can change height (emoji panel, rotation)
            imeHeight = kb
            shapeBar()
        } else if (docked) {
            docked = false
            imeHeight = 0
            shapeCircle()
        }
    }

    private fun shapeBar() {
        val w = screenW() - 2 * barMargin()
        val h = barHeight()
        val y = (screenH() - imeHeight - h).coerceAtLeast(0)
        placeWindow(barMargin(), y, w, h)
        shapeInner(w, h, barRadius())
    }

    private fun shapeCircle() {
        val s = bubbleSize()
        // clamp in case the screen rotated while docked
        val x = bubbleX.coerceIn(0, (screenW() - s).coerceAtLeast(0))
        val y = bubbleY.coerceIn(0, (screenH() - s).coerceAtLeast(0))
        placeWindow(x, y, s, s)
        shapeInner(circleSize(), circleSize(), circleSize() / 2f)
    }

    private fun placeWindow(x: Int, y: Int, w: Int, h: Int) {
        if (params.x == x && params.y == y && params.width == w && params.height == h) return
        params.x = x
        params.y = y
        params.width = w
        params.height = h
        if (root.isAttachedToWindow) runCatching { wm.updateViewLayout(root, params) }
    }

    private fun shapeInner(w: Int, h: Int, radius: Float) {
        val lp = circle.layoutParams as FrameLayout.LayoutParams
        if (lp.width != w || lp.height != h) {
            lp.width = w
            lp.height = h
            lp.gravity = Gravity.CENTER
            circle.layoutParams = lp
        }
        bg.cornerRadius = radius
    }

    // ------------------------------------------------------------------ touch

    private var downX = 0f
    private var downY = 0f
    private var startX = 0
    private var startY = 0
    private var moved = false
    private val longPress = Runnable { longPressed = true; bump(); onLongPress() }

    private fun onTouch(v: View, e: MotionEvent): Boolean {
        val slop = ViewConfiguration.get(this).scaledTouchSlop
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
                if (docked) {
                    // the bar stays pinned to the keyboard
                    if (abs(dx) > slop || abs(dy) > slop) main.removeCallbacks(longPress)
                } else if (moved || abs(dx) > slop || abs(dy) > slop) {
                    moved = true
                    main.removeCallbacks(longPress)
                    params.x = startX + dx.toInt()
                    params.y = startY + dy.toInt()
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
        when (state) {
            State.LOADING -> toast("Φορτώνει το μοντέλο...")
            State.ERROR -> toast(Recognizer.lastError ?: "Το μοντέλο δεν φορτώθηκε")
            State.IDLE -> runCatching { recorder.start(); setState(State.LISTENING) }
                .onFailure { toast("Μικρόφωνο: $it") }
            State.LISTENING -> transcribe(recorder.stop(), "mic")
            State.THINKING -> {}
        }
    }

    private fun onLongPress() {
        if (state != State.IDLE) return
        val f = Recognizer.testWav(this)
        if (!f.exists()) { toast("Δεν υπάρχει test.wav"); return }
        transcribe(AudioRecorder.readWav(f), f.name)
    }

    private fun transcribe(wav: FloatArray, id: String) {
        if (wav.size < AudioRecorder.SAMPLE_RATE / 5) { setState(State.IDLE); toast("Πολύ σύντομο"); return }
        setState(State.THINKING)
        Thread {
            // what is already written in the field: layer 2 continues it (read on this thread,
            // before the result is typed, so it never includes the new words)
            val context = TypingAccessibilityService.contextBefore()
            val r = runCatching { Recognizer.recognize(wav, id, context) }
            main.post {
                setState(State.IDLE)
                r.onSuccess { res ->
                    val where = TypingAccessibilityService.deliver(this, res.text)
                    if (res.text.isBlank()) toast("Δεν αναγνωρίστηκε τίποτα")
                    else if (where == "clipboard") toast("Αντιγράφηκε: ${res.text}")
                    else toast(res.text)
                    Log.i(TAG, "[$id] $where: ${res.text} (${res.inferenceMs} ms)")
                }.onFailure { toast("Σφάλμα: $it") }
            }
        }.start()
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
        private const val CHANNEL = "gvt_overlay"
        private const val NOTIF_ID = 1001
        const val ACTION_STOP = "gr.greekvt.STOP"
        const val ACTION_RELOAD = "gr.greekvt.RELOAD"
        @Volatile var running = false
    }
}
