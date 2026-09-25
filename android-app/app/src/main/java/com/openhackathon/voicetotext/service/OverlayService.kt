package com.openhackathon.voicetotext.service

import com.openhackathon.voicetotext.asr.Recognizer
import com.openhackathon.voicetotext.audio.AudioRecorder
import com.openhackathon.voicetotext.R
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
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
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowManager
import android.view.animation.DecelerateInterpolator
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Toast
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsAnimationCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The floating microphone. Lives above every app as a foreground service.
 *
 * Tap to listen (the circle turns red and pulses), tap again to transcribe (amber, spinner),
 * and the text is typed into whatever field has focus. Long press runs the pushed test.wav
 * instead of the mic. Drag it anywhere; it snaps to the nearer side when released.
 * When the software keyboard is open, the circle stretches into a bar parked just above it.
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

    /** 0 = floating circle, 1 = keyboard-extension bar. */
    private var shapeT = 0f
    private var shapeAnim: ValueAnimator? = null
    private var docked = false
    private var bubbleX = 0
    private var bubbleY = 0
    private var lastImeBottom = 0
    private var imePeak = 0
    private var imeAnimRunning = false

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
        colorAnim?.cancel(); pulseAnim?.cancel(); shapeAnim?.cancel()
        main.removeCallbacks(imePoll)
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
            // Position ourselves; do not let the WM shrink this window around the IME.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                setFitInsetsTypes(0)
            }
        }
        bubbleX = params.x
        bubbleY = params.y

        root.setOnTouchListener(::onTouch)
        root.alpha = 0f
        wm.addView(root, params)
        listenForIme()
        layoutOverlay()
        root.animate().alpha(1f).setDuration(220).start()
        main.post(imePoll)
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
    private fun lerp(a: Int, b: Int, t: Float) = (a + (b - a) * t).roundToInt()
    private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t

    private fun screenW(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) wm.maximumWindowMetrics.bounds.width()
        else resources.displayMetrics.widthPixels

    private fun screenH(): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) wm.maximumWindowMetrics.bounds.height()
        else resources.displayMetrics.heightPixels

    private fun listenForIme() {
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            applyImeInsets(insets, fromAnimation = false)
            insets
        }
        ViewCompat.setWindowInsetsAnimationCallback(
            root,
            object : WindowInsetsAnimationCompat.Callback(
                WindowInsetsAnimationCompat.Callback.DISPATCH_MODE_CONTINUE_ON_SUBTREE,
            ) {
                override fun onStart(
                    animation: WindowInsetsAnimationCompat,
                    bounds: WindowInsetsAnimationCompat.BoundsCompat,
                ): WindowInsetsAnimationCompat.BoundsCompat {
                    if (animation.typeMask and WindowInsetsCompat.Type.ime() != 0) {
                        imeAnimRunning = true
                    }
                    return bounds
                }

                override fun onProgress(
                    insets: WindowInsetsCompat,
                    runningAnimations: MutableList<WindowInsetsAnimationCompat>,
                ): WindowInsetsCompat {
                    if (runningAnimations.any { it.typeMask and WindowInsetsCompat.Type.ime() != 0 }) {
                        applyImeInsets(insets, fromAnimation = true)
                    }
                    return insets
                }

                override fun onEnd(animation: WindowInsetsAnimationCompat) {
                    if (animation.typeMask and WindowInsetsCompat.Type.ime() != 0) {
                        imeAnimRunning = false
                        ViewCompat.getRootWindowInsets(root)?.let { applyImeInsets(it, fromAnimation = false) }
                    }
                }
            },
        )
        ViewCompat.requestApplyInsets(root)
    }

    private val imePoll = object : Runnable {
        override fun run() {
            if (!imeAnimRunning) {
                val insets = currentImeInsets()
                if (insets != null) applyImeInsets(insets, fromAnimation = false)
            }
            if (running) main.postDelayed(this, 48)
        }
    }

    private fun currentImeInsets(): WindowInsetsCompat? {
        val fromView = ViewCompat.getRootWindowInsets(root)
        val fromWm = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            WindowInsetsCompat.toWindowInsetsCompat(wm.currentWindowMetrics.windowInsets)
        } else null
        return when {
            fromView != null && fromWm != null -> {
                // Overlay windows often report 0 IME on the view; WindowMetrics can still see it.
                val type = WindowInsetsCompat.Type.ime()
                if (fromView.getInsets(type).bottom >= fromWm.getInsets(type).bottom) fromView else fromWm
            }
            else -> fromView ?: fromWm
        }
    }

    private fun applyImeInsets(insets: WindowInsetsCompat, fromAnimation: Boolean) {
        val type = WindowInsetsCompat.Type.ime()
        val bottom = insets.getInsets(type).bottom
        val visible = insets.isVisible(type)
        val open = bottom >= imeThreshold() || (fromAnimation && (visible || bottom > 0))
        if (!fromAnimation &&
            bottom == lastImeBottom &&
            open == docked &&
            shapeAnim?.isRunning != true &&
            ((open && shapeT == 1f) || (!open && shapeT == 0f))
        ) return

        if (open) {
            if (!docked) {
                bubbleX = params.x
                bubbleY = params.y
                docked = true
            }
            lastImeBottom = bottom
            if (bottom > imePeak) imePeak = bottom
            if (fromAnimation) {
                shapeAnim?.cancel()
                shapeT = (bottom.toFloat() / imePeak.coerceAtLeast(1)).coerceIn(0f, 1f)
                layoutOverlay()
            } else if (shapeT < 1f) {
                animateShapeTo(1f)
            } else {
                layoutOverlay()
            }
        } else {
            docked = false
            if (fromAnimation) {
                lastImeBottom = bottom
                shapeAnim?.cancel()
                val peak = imePeak.coerceAtLeast(1)
                shapeT = (bottom.toFloat() / peak).coerceIn(0f, 1f)
                layoutOverlay()
                if (bottom == 0) imePeak = 0
            } else if (shapeT > 0f) {
                animateShapeTo(0f)
            } else {
                lastImeBottom = 0
                imePeak = 0
            }
        }
    }

    private fun animateShapeTo(target: Float) {
        if (shapeT == target && shapeAnim?.isRunning != true) {
            layoutOverlay()
            return
        }
        shapeAnim?.cancel()
        shapeAnim = ValueAnimator.ofFloat(shapeT, target).apply {
            duration = 280
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                shapeT = it.animatedValue as Float
                layoutOverlay()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    if (target == 0f) {
                        lastImeBottom = 0
                        imePeak = 0
                    }
                }
            })
            start()
        }
    }

    /** Width, height, corner radius, and Y all follow [shapeT]; Y targets just above the IME. */
    private fun layoutOverlay() {
        val t = shapeT
        val w = lerp(bubbleSize(), screenW() - 2 * barMargin(), t)
        val h = lerp(bubbleSize(), barHeight(), t)
        val x = lerp(bubbleX, barMargin(), t)
        val gap = dp(4)
        val dockY = (screenH() - lastImeBottom - h - gap).coerceAtLeast(dp(48))
        val y = lerp(bubbleY, dockY, t)
        params.width = w
        params.height = h
        params.x = x
        params.y = y
        runCatching { wm.updateViewLayout(root, params) }

        val lp = circle.layoutParams as FrameLayout.LayoutParams
        lp.width = lerp(circleSize(), w, t)
        lp.height = lerp(circleSize(), h, t)
        lp.gravity = Gravity.CENTER
        circle.layoutParams = lp
        bg.cornerRadius = lerp(circleSize() / 2f, barRadius(), t)
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
                if (docked || shapeT > 0.15f) {
                    if (abs(dx) > slop || abs(dy) > slop) main.removeCallbacks(longPress)
                } else if (moved || abs(dx) > slop || abs(dy) > slop) {
                    moved = true
                    main.removeCallbacks(longPress)
                    params.x = startX + dx.toInt()
                    params.y = startY + dy.toInt()
                    wm.updateViewLayout(root, params)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                main.removeCallbacks(longPress)
                circle.animate().scaleX(1f).scaleY(1f).setDuration(140).start()
                if (moved && !docked && shapeT == 0f) snapToEdge()
                else if (!longPressed && e.actionMasked == MotionEvent.ACTION_UP) { v.performClick(); bump(); onTap() }
            }
        }
        return true
    }

    /** Slide to the nearer side and stay clear of the screen edges. */
    private fun snapToEdge() {
        val screenW = resources.displayMetrics.widthPixels
        val screenH = resources.displayMetrics.heightPixels
        val w = if (root.width > 0) root.width else dp(96)
        val h = if (root.height > 0) root.height else dp(96)
        val target = if (params.x + w / 2 < screenW / 2) dp(8) else screenW - w - dp(8)
        params.y = params.y.coerceIn(dp(48), screenH - h - dp(72))
        ValueAnimator.ofInt(params.x, target).apply {
            duration = 220
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                params.x = it.animatedValue as Int
                bubbleX = params.x
                bubbleY = params.y
                runCatching { wm.updateViewLayout(root, params) }
            }
            start()
        }
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
