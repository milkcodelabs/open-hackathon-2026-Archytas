package com.openhackathon.voicetotext.service

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

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
 * For undo it also keeps, in memory only, the last few phrases it typed itself.
 */
class TypingAccessibilityService : AccessibilityService() {

    @Volatile private var lastEditable: AccessibilityNodeInfo? = null

    /** What each insert put in the field, separator included, newest last; only our own text. */
    private val injected = ArrayDeque<String>()

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

    /** What was added to the field (the separator and [text]), or null if it refused. */
    private fun append(node: AccessibilityNodeInfo, text: String): String? {
        val existing = if (node.isShowingHintText) "" else (node.text?.toString() ?: "")
        val sep = if (existing.isEmpty() || existing.endsWith(" ")) "" else " "
        val combined = existing + sep + text
        return if (setText(node, combined, combined.length)) sep + text else null
    }

    private fun setText(node: AccessibilityNodeInfo, text: String, cursor: Int): Boolean {
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        if (!node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) return false
        val sel = Bundle().apply {
            putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_START_INT, cursor)
            putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_END_INT, cursor)
        }
        node.performAction(AccessibilityNodeInfo.ACTION_SET_SELECTION, sel)
        return true
    }

    /**
     * Swaps the last occurrence of [old] (what this app typed) for [new], keeping the cursor
     * where it was relative to the rest of the text, so words typed since then survive.
     */
    private fun replace(old: String, new: String): String? {
        for ((where, node) in listOf("focused" to focusedEditable(), "remembered" to rememberedEditable())) {
            if (node == null || node.isShowingHintText) continue
            val text = node.text?.toString() ?: continue
            val at = text.lastIndexOf(old)
            if (at < 0) continue
            val combined = text.substring(0, at) + new + text.substring(at + old.length)
            val cursor = node.textSelectionStart.takeIf { it in 0..text.length } ?: text.length
            val moved = when {
                cursor >= at + old.length -> cursor + new.length - old.length
                cursor > at -> at + new.length
                else -> cursor
            }
            if (setText(node, combined, moved)) return where
        }
        return null
    }

    /** Some fields refuse SET_TEXT but accept a paste of the clipboard. */
    private fun paste(node: AccessibilityNodeInfo, text: String): Boolean {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("greek_vt", text))
        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        return node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
    }

    /** Text before the cursor of the field the result will go to; null if none or a hint. */
    private fun textBeforeCursor(): String? {
        val node = focusedEditable() ?: rememberedEditable() ?: return null
        if (node.isShowingHintText || node.isPassword) return null
        val text = node.text?.toString() ?: return null
        val cursor = node.textSelectionStart.takeIf { it in 0..text.length } ?: text.length
        return text.substring(0, cursor).takeLast(CONTEXT_CHARS)
    }

    private fun insert(text: String): String? {
        for ((where, node) in listOf("focused" to focusedEditable(), "remembered" to rememberedEditable())) {
            if (node == null) continue
            append(node, text)?.let { remember(it); return where }
            if (paste(node, text)) { remember(text); return "$where/paste" }
        }
        return null
    }

    private fun remember(piece: String) = synchronized(injected) {
        injected.addLast(piece)
        while (injected.size > UNDO_DEPTH) injected.removeFirst()
    }

    /** A phrase we typed was rewritten (word choice, late LLM answer): undo must remove the new words. */
    private fun rememberReplaced(old: String, new: String) = synchronized(injected) {
        val last = injected.lastOrNull() ?: return
        if (last.endsWith(old)) injected[injected.size - 1] = last.dropLast(old.length) + new
    }

    private fun hasUndo() = synchronized(injected) { injected.isNotEmpty() }

    /**
     * Removes the newest phrase we typed from the field, wherever it now sits, keeping the
     * cursor on the same text. A phrase no longer found (edited, or another field) is forgotten.
     */
    private fun undo(): String {
        val piece = synchronized(injected) { injected.removeLastOrNull() } ?: return "nothing"
        // the field may have trimmed the separator in front of it
        val where = replace(piece, "") ?: piece.trimStart().takeIf { it != piece }?.let { replace(it, "") }
        return if (where != null) "undone" else "missing"
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
            val r = runCatching { s.undo() }.getOrDefault("missing")
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
            cm.setPrimaryClip(ClipData.newPlainText("greek_vt", text))
            Log.i(TAG, "no editable field, copied to the clipboard")
            return "clipboard"
        }

        /**
         * Changes text delivered earlier: [old] becomes [new] in the field it went to, or on
         * the clipboard if it cannot be found there any more (unless [copyIfMissing] is off).
         */
        fun replace(ctx: Context, old: String, new: String, copyIfMissing: Boolean = true): String {
            if (old.isBlank()) return deliver(ctx, new)
            runCatching { instance?.replace(old, new) }.getOrNull()?.let {
                instance?.rememberReplaced(old, new)
                Log.i(TAG, "replaced via $it")
                return "typed"
            }
            if (!copyIfMissing) {
                Log.i(TAG, "typed text not found (edited or left the field), correction dropped")
                return "missing"
            }
            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("greek_vt", new))
            Log.i(TAG, "typed text not found, copied the correction to the clipboard")
            return "clipboard"
        }
    }
}
