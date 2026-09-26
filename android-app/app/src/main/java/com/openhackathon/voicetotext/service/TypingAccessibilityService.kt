package com.openhackathon.voicetotext.service

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlin.math.abs

/**
 * Writes recognized text into whichever text field the person was last typing in.
 *
 * Finding that field at the moment of insertion is not enough: by then the user has tapped
 * the floating bubble, and on many keyboards and apps the field is no longer reported as
 * holding input focus. So the service also remembers the last editable node it saw focused
 * and falls back to it, then to a clipboard paste, then to the clipboard alone.
 *
 * It looks at nothing else and stores nothing but that one node reference. The text already
 * in that field (at most the last 200 characters before the cursor) is read at the moment of
 * recognition so the language model can continue the sentence; it is not stored or logged.
 * For undo and corrections it also keeps, in memory only, the last few phrases it typed
 * itself and where each one went.
 *
 * A field's text is always re-read from the app ([AccessibilityNodeInfo.refresh]) before it
 * is searched or rewritten: the framework's node cache can still hold the text from before
 * our own insert, and a phrase looked up in that stale copy is never found.
 */
class TypingAccessibilityService : AccessibilityService() {

    @Volatile private var lastEditable: AccessibilityNodeInfo? = null

    /**
     * One phrase this app typed: [text] starts at [at] in [node]'s text, right after the
     * separator [sep] that was added in front of it. [at] and [text] follow later rewrites.
     */
    private class Injection(var node: AccessibilityNodeInfo?, var at: Int, val sep: String, var text: String)

    /** Our own phrases, newest last. */
    private val injected = ArrayDeque<Injection>()

    override fun onServiceConnected() {
        instance = this
        Log.i(TAG, "connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val e = event ?: return
        when (e.eventType) {
            AccessibilityEvent.TYPE_VIEW_FOCUSED,
            AccessibilityEvent.TYPE_VIEW_CLICKED,
            AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED -> {
                val src = e.source
                if (src == null) {
                    Log.i(TAG, "event ${e.eventType} from ${e.packageName} carried no source")
                    return
                }
                if (src.isEditable) {
                    lastEditable = src
                    Log.i(TAG, "remembered a field in ${src.packageName} (${src.viewIdResourceName})")
                } else {
                    Log.i(TAG, "event ${e.eventType} from ${e.packageName}: ${src.className}, not editable")
                }
            }
        }
    }

    override fun onInterrupt() {}

    override fun onDestroy() {
        if (instance === this) instance = null
        lastEditable = null
        synchronized(injected) { injected.clear() }
        super.onDestroy()
    }

    /** The field holding input focus right now, in any window that is not ours. */
    private fun focusedEditable(): AccessibilityNodeInfo? {
        rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)?.let {
            if (it.isEditable) return it
        }
        for (w in windows) {
            val n = w.root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: continue
            if (n.isEditable) return n
        }
        return null
    }

    /** The remembered field, re-read so a stale reference is not used. */
    private fun rememberedEditable(): AccessibilityNodeInfo? {
        val n = lastEditable ?: return null
        return if (runCatching { n.refresh() }.getOrDefault(false) && n.isEditable) n else null
    }

    /** Fields to try, [first] (where a phrase went) ahead of the focused and remembered ones. */
    private fun targets(first: AccessibilityNodeInfo? = null): List<Pair<String, AccessibilityNodeInfo>> =
        listOfNotNull(
            first?.let { "typed-into" to it },
            focusedEditable()?.let { "focused" to it },
            rememberedEditable()?.let { "remembered" to it },
        ).distinctBy { it.second }

    /** [node]'s text as the app holds it right now ("" while it shows its hint); null if gone. */
    private fun freshText(node: AccessibilityNodeInfo): String? {
        if (!runCatching { node.refresh() }.getOrDefault(false) || !node.isEditable) return null
        return if (node.isShowingHintText) "" else node.text?.toString() ?: ""
    }

    private fun cursorOf(node: AccessibilityNodeInfo, text: String): Int =
        node.textSelectionStart.takeIf { it in 0..text.length } ?: text.length

    /** Adds [text] at the end of the field; what was added and where, or null if it refused. */
    private fun append(node: AccessibilityNodeInfo, text: String): Injection? {
        val existing = freshText(node) ?: return null
        val sep = if (existing.isEmpty() || existing.endsWith(" ")) "" else " "
        val combined = existing + sep + text
        if (!setText(node, combined, combined.length)) return null
        return Injection(node, existing.length + sep.length, sep, text)
    }

    private fun setText(node: AccessibilityNodeInfo, text: String, cursor: Int): Boolean {
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        if (!node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) return false
        select(node, cursor, cursor)
        return true
    }

    private fun select(node: AccessibilityNodeInfo, start: Int, end: Int): Boolean {
        val sel = Bundle().apply {
            putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_START_INT, start)
            putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_END_INT, end)
        }
        return node.performAction(AccessibilityNodeInfo.ACTION_SET_SELECTION, sel)
    }

    /**
     * Rewrites characters [start, end) of [text] (the field's current text) as [new], keeping
     * the cursor on the same text. A field that refuses SET_TEXT gets the span selected and
     * then pasted over, or cut out when [new] is empty.
     */
    private fun splice(node: AccessibilityNodeInfo, text: String, start: Int, end: Int, new: String): Boolean {
        val combined = text.substring(0, start) + new + text.substring(end)
        val cursor = cursorOf(node, text)
        val moved = when {
            cursor >= end -> cursor + new.length - (end - start)
            cursor > start -> start + new.length
            else -> cursor
        }
        if (setText(node, combined, moved)) return true
        if (!select(node, start, end)) return false
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val saved = runCatching { cm.primaryClip }.getOrNull()
        val ok = if (new.isEmpty()) {
            node.performAction(AccessibilityNodeInfo.ACTION_CUT)
        } else {
            cm.setPrimaryClip(ClipData.newPlainText("voice-to-text", new))
            node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
        }
        if (saved != null) cm.setPrimaryClip(saved)
        return ok
    }

    /** Where [inj]'s text starts in [text]: its recorded place if it is still there, else its nearest copy; -1 if gone. */
    private fun locate(text: String, inj: Injection): Int {
        if (inj.text.isEmpty()) return -1
        if (text.startsWith(inj.text, inj.at)) return inj.at
        var best = -1
        var i = text.indexOf(inj.text)
        while (i >= 0) {
            if (best < 0 || abs(i - inj.at) < abs(best - inj.at)) best = i
            i = text.indexOf(inj.text, i + 1)
        }
        return best
    }

    /**
     * Overwrites [inj]'s whole phrase with [new] in the field it went to (or, failing that, the
     * focused or remembered one). [withSep]: its separator goes too, as for undo.
     */
    private fun rewrite(inj: Injection, new: String, withSep: Boolean): String? {
        for ((where, node) in targets(inj.node)) {
            val text = freshText(node) ?: continue
            val at = locate(text, inj)
            if (at < 0) continue
            val start = if (withSep && inj.sep.isNotEmpty() && text.startsWith(inj.sep, at - inj.sep.length)) {
                at - inj.sep.length
            } else at
            if (!splice(node, text, start, at + inj.text.length, new)) continue
            inj.node = node
            inj.at = at
            inj.text = new
            return where
        }
        return null
    }

    /** Some fields refuse SET_TEXT but accept a paste of the clipboard. */
    private fun paste(node: AccessibilityNodeInfo, text: String): Boolean {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("voice-to-text", text))
        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        return node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
    }

    /** Text before the cursor of the field the result will go to; null if none or a hint. */
    private fun textBeforeCursor(): String? {
        val node = focusedEditable() ?: rememberedEditable() ?: return null
        if (node.isPassword) return null
        val text = freshText(node)?.takeIf { it.isNotEmpty() } ?: return null
        return text.substring(0, cursorOf(node, text)).takeLast(CONTEXT_CHARS)
    }

    private fun insert(text: String): String? {
        for ((where, node) in targets()) {
            append(node, text)?.let { remember(it); return where }
            val before = freshText(node) ?: ""
            val cursor = cursorOf(node, before)
            if (paste(node, text)) { remember(Injection(node, cursor, "", text)); return "$where/paste" }
        }
        return null
    }

    private fun remember(inj: Injection) = synchronized(injected) {
        injected.addLast(inj)
        while (injected.size > UNDO_DEPTH) injected.removeFirst()
    }

    private fun hasUndo() = synchronized(injected) { injected.isNotEmpty() }

    /**
     * Removes the newest phrase we typed from the field, wherever it now sits, keeping the
     * cursor on the same text. A phrase no longer found (edited, or another field) is forgotten.
     */
    private fun undo(): String {
        val inj = synchronized(injected) { injected.removeLastOrNull() } ?: return "nothing"
        return if (rewrite(inj, "", withSep = true) != null) "undone" else "missing"
    }

    /**
     * Overwrites the phrase we typed as [old] with [new], all of it. Without a record of it
     * (it went in by clipboard, or was forgotten) the last copy of [old] in the field is used.
     */
    private fun overwrite(old: String, new: String): String? {
        val inj = synchronized(injected) { injected.lastOrNull { it.text == old } }
            ?: Injection(null, Int.MAX_VALUE, "", old)
        return rewrite(inj, new, withSep = false)
    }

    companion object {
        private const val TAG = "TypingA11y"
        @Volatile var instance: TypingAccessibilityService? = null

        val enabled: Boolean get() = instance != null

        /** Enough characters for the last words of the sentence; nothing more is read. */
        private const val CONTEXT_CHARS = 200
        /** How many of our own phrases can be undone one after another. */
        private const val UNDO_DEPTH = 20

        val canUndo: Boolean get() = instance?.hasUndo() == true

        /**
         * Takes the last phrase this app typed back out of the field it went to. Returns
         * "undone", "missing" (the text is no longer there) or "nothing" (nothing to undo).
         */
        fun undoLastInjection(): String {
            val s = instance ?: return "nothing"
            val r = runCatching { s.undo() }.getOrElse { Log.w(TAG, "undo failed", it); "missing" }
            Log.i(TAG, "undo: $r")
            return r
        }

        /**
         * The text already written before the cursor in the field being typed into, for the
         * language model's context. Stays on the phone; never stored or logged.
         */
        fun contextBefore(): String? = runCatching { instance?.textBeforeCursor() }.getOrNull()

        /** Types the text where the person was writing; copies it if there is nowhere to type. */
        fun deliver(ctx: Context, text: String): String {
            if (text.isBlank()) return "empty"
            instance?.insert(text)?.let { Log.i(TAG, "inserted via $it"); return "typed" }
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("voice-to-text", text))
            Log.i(TAG, "no editable field, copied to the clipboard")
            return "clipboard"
        }

        /**
         * Replaces a phrase typed earlier: the whole of [old] (exactly as it was delivered or
         * last replaced) is selected in the field and overwritten with [new], never appended
         * to. If it cannot be found any more, [new] goes to the clipboard unless
         * [copyIfMissing] is off.
         */
        fun replaceInjected(ctx: Context, old: String, new: String, copyIfMissing: Boolean = true): String {
            if (old.isBlank()) return deliver(ctx, new)
            if (old == new) return "typed"
            runCatching { instance?.overwrite(old, new) }
                .onFailure { Log.w(TAG, "replace failed", it) }
                .getOrNull()?.let {
                    Log.i(TAG, "replaced via $it")
                    return "typed"
                }
            if (!copyIfMissing) {
                Log.i(TAG, "typed text not found (edited or left the field), correction dropped")
                return "missing"
            }
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("voice-to-text", new))
            Log.i(TAG, "typed text not found, copied the correction to the clipboard")
            return "clipboard"
        }
    }
}
