import { summarizeBehavior, textToKeystrokes } from './features.js';

export const PRINTABLE_CODE_RE =
  /^(Key[A-Z]|Digit[0-9]|Space|Period|Comma|Slash|Semicolon|Quote|BracketLeft|BracketRight|Minus|Equal|Backquote|Backslash|Numpad[0-9]|NumpadDecimal|NumpadAdd|NumpadSubtract)$/;

const LONG_PAUSE_MS = 2200;
const MIN_VALID_HOLD_SEC = 0.008;
const MAX_HOLD_SEC = 1.2;
const PENDING_TTL_MS = 1000;
// IME edits that arrive as "delete, then insert" within this window are one edit (README §6.2):
// iOS / Samsung Korean keyboards and Safari replace the composing syllable this way, and a final
// consonant moving to the next syllable (간 + ㅏ → 가나) can pass through an intermediate "가".
const IME_PAIR_MS = 30;
const HANGUL_JAMO_RE = /[\u1100-\u11ff\u3130-\u318f]/;

const CODE_SHORT_LABELS = {
  Space: '␣',
  Period: '.',
  Comma: ',',
  Slash: '/',
  Semicolon: ';',
  Quote: "'",
  BracketLeft: '[',
  BracketRight: ']',
  Minus: '-',
  Equal: '=',
  Backquote: '`',
  Backslash: '\\',
};

export function formatKeyLabel(key, code) {
  if (code === 'Space' || key === ' ') return '␣';
  if (key && key !== 'Process' && key !== 'Unidentified' && [...key].length === 1) {
    return key;
  }
  if (CODE_SHORT_LABELS[code]) return CODE_SHORT_LABELS[code];
  if (code?.startsWith('Key')) return code.slice(3);
  if (code?.startsWith('Digit')) return code.slice(5);
  if (code?.startsWith('Numpad')) return code.slice(6);
  return code || '?';
}

const eventTime = (e) =>
  e && typeof e.timeStamp === 'number' && e.timeStamp > 0 ? e.timeStamp : performance.now();

function holdSec(stroke) {
  if (stroke.source !== 'key' || stroke.upTime === null || stroke.upTime === undefined) return null;
  const h = (stroke.upTime - stroke.downTime) / 1000.0;
  return h < MIN_VALID_HOLD_SEC ? null : h;
}

function changedKeys(prev, next, selection, inputType) {
  let start = 0;
  let end = prev.length;
  let nextEnd = next.length;
  if (selection) {
    let { start: selectedStart, end: selectedEnd } = selection;
    const delta = prev.length - next.length;
    if (selectedStart === selectedEnd && delta > 0) {
      if (inputType.endsWith('Backward')) selectedStart -= delta;
      if (inputType.endsWith('Forward')) selectedEnd += delta;
    }
    const changedEnd = selectedEnd - delta;
    if (selectedStart >= 0 && selectedEnd <= prev.length && changedEnd >= selectedStart &&
        prev.slice(0, selectedStart).every((key, i) => key === next[i]) &&
        prev.slice(selectedEnd).every((key, i) => key === next[changedEnd + i])) {
      // A caret or selection disambiguates repeated characters. During composition, the
      // selected syllable can itself contain jamo that were already recorded.
      start = selectedStart;
      end = selectedEnd;
      nextEnd = changedEnd;
      if (!inputType.includes('Composition') && !selection.composing) {
        return { start, end, added: next.slice(start, nextEnd) };
      }
    }
  }

  while (start < end && start < nextEnd && prev[start] === next[start]) start++;
  while (end > start && nextEnd > start && prev[end - 1] === next[nextEnd - 1]) {
    end--;
    nextEnd--;
  }
  return { start, end, added: next.slice(start, nextEnd) };
}

/**
 * Device-agnostic keystroke recorder.
 *
 * Physical keyboards: a printable keydown supplies the physical code and hold for the
 * text received in `input`. Edits remove the affected strokes and break interrupted transitions.
 *
 * Touch keyboards (Android Gboard, iOS) usually report `code === ''` / key 'Unidentified'
 * and fire keyup together with keydown, which made every phone capture empty
 * ("추출된 연속 리듬 구간이 부족합니다"). Text changes are therefore also read from `input`
 * events: the value is expanded into keystrokes (Hangul syllables -> 2-beolsik jamo keys),
 * diffed against the previous value, and keystrokes not already produced by a keydown become
 * "soft" strokes timed at the input event with an unknown hold (Hold_Valid = 0).
 *
 * A separate event log supplies the behavioural profile (edit habits, Shift/CapsLock
 * strategy, pause rhythm).
 */
export class SmartKeystrokeRecorder {
  constructor(inputEl, { onLiveUpdate, onSubmit } = {}) {
    this.inputEl = inputEl;
    this.onLiveUpdate = onLiveUpdate;
    this.onSubmit = onSubmit;
    this.reset();
    this._bind();
  }

  reset() {
    this.lastInputValue = this.inputEl.value;
    this.prevKeys = textToKeystrokes(this.lastInputValue);
    this.stack = this.prevKeys.map((char) => this._makeStroke({ char, t: performance.now(), source: 'input', synthetic: true }));
    this.poppedStrokes = [];
    this.rawEvents = []; // Behaviour log: { kind, side, t, up, rep }
    this.activeDown = new Map(); // code -> FIFO of { stroke?, raw } awaiting keyup
    this.pendingKeyStrokes = []; // { t, stroke } keydown strokes awaiting their input event
    this.pendingKeyDeletes = [];
    this.beforeInput = null;
    if (this.imeChain?.timer) clearTimeout(this.imeChain.timer);
    this.imeChain = null; // deletion without a Backspace keydown, waiting for its IME re-insert
    this.justEdited = false;
    this.backspaceCount = 0;
    this.hadPaste = false;
    this.firstDownTime = null;
    this.lastDownTime = null;
    this.lastStroke = null;
  }

  // Field erased back to empty: rhythm stack starts over, behaviour log is kept
  _clearTextState() {
    this.stack = [];
    this.poppedStrokes = [];
    this.justEdited = false;
    this.backspaceCount = 0;
    this.firstDownTime = null;
    this.lastDownTime = null;
    this.lastStroke = null;
  }

  _makeStroke({ code, char, t, source, synthetic = false }) {
    return {
      code: code || null,
      char: char ?? null,
      key: code ? code : `char:${char}`,
      label: formatKeyLabel(char || '', code),
      downTime: t,
      upTime: null,
      source,
      synthetic,
      editedBefore: false,
      pausedBefore: false,
      cleanFromPrev: false,
    };
  }

  _joinStroke(stroke, prev) {
    stroke.editedBefore = Boolean(prev && (this.justEdited || (this.lastStroke && prev !== this.lastStroke)));
    stroke.pausedBefore = Boolean(prev && stroke.downTime - prev.downTime > LONG_PAUSE_MS);
    stroke.cleanFromPrev = !prev || (!stroke.editedBefore && !stroke.pausedBefore && !prev.synthetic && !stroke.synthetic);
    if (!stroke.synthetic) {
      if (this.firstDownTime === null) this.firstDownTime = stroke.downTime;
      this.lastDownTime = stroke.downTime;
      this.lastStroke = stroke;
      this.justEdited = false;
    }
  }

  _markDown(code, entry) {
    const q = this.activeDown.get(code) || [];
    q.push(entry);
    this.activeDown.set(code, q);
  }

  _bind() {
    this.inputEl.addEventListener('paste', (e) => {
      e.preventDefault();
      if (this.inputEl.readOnly) return;
      this.hadPaste = true;
      if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
    });

    this.inputEl.addEventListener('keydown', (e) => {
      if (this.inputEl.readOnly && e.key !== 'Enter') return;
      const now = eventTime(e);
      if (this.inputEl.value !== this.lastInputValue) this.reset();

      if (e.key === 'Enter') {
        if (e.repeat) { e.preventDefault(); return; }
        if (e.isComposing) {
          // Allow Korean IME composition to commit final syllable before reading inputEl.value
          setTimeout(() => {
            if (this.onSubmit) this.onSubmit();
          }, 15);
          return;
        }
        e.preventDefault();
        if (this.onSubmit) this.onSubmit();
        return;
      }

      const code = e.code || '';
      if (e.key === 'Backspace' || e.key === 'Delete' || code === 'Backspace' || code === 'Delete') {
        const forward = e.key === 'Delete' || (e.key !== 'Backspace' && code === 'Delete');
        const raw = { kind: 'bs', side: null, t: now, up: null, rep: Boolean(e.repeat), direction: forward ? 'Forward' : 'Backward' };
        if (!e.repeat) this._markDown(code || e.key, { raw });
        this.pendingKeyDeletes.push(raw);
        return;
      }

      if (code === 'ShiftLeft' || code === 'ShiftRight' || e.key === 'Shift' || code === 'CapsLock' || e.key === 'CapsLock') {
        if (e.repeat) return;
        const isCaps = code === 'CapsLock' || e.key === 'CapsLock';
        const raw = {
          kind: isCaps ? 'caps' : 'shift',
          side: code === 'ShiftLeft' ? 'L' : code === 'ShiftRight' ? 'R' : null,
          t: now,
          up: null,
          rep: false,
        };
        this.rawEvents.push(raw);
        this._markDown(code || e.key, { raw });
        return;
      }

      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (!PRINTABLE_CODE_RE.test(code)) return; // touch keyboards: text arrives via `input`
      const char = e.key && [...e.key].length === 1 ? e.key : null;
      const composing = e.isComposing || e.key === 'Process';
      if (e.repeat) {
        this.pendingKeyStrokes.push({ t: now, char, composing, code, stroke: null });
        return;
      }

      const stroke = this._makeStroke({ code, char, t: now, source: 'key' });
      // Capital letters: was it typed with CapsLock or with Shift? (a stable personal habit)
      if (/^[A-Z]$/.test(e.key || '')) {
        stroke.capital = e.getModifierState?.('CapsLock') ? 'caps' : e.shiftKey ? 'shift' : null;
      }
      const raw = { kind: code === 'Space' ? 'space' : 'char', side: null, t: now, up: null, rep: false };
      this.rawEvents.push(raw);
      this._markDown(code, { stroke, raw });
      this.pendingKeyStrokes.push({ t: now, char, composing, code, stroke });
    });

    this.inputEl.addEventListener('keyup', (e) => {
      const now = eventTime(e);
      const code = e.code || e.key;
      const q = this.activeDown.get(code);
      if (!q || q.length === 0) return;
      const entry = q.shift();
      if (entry.stroke && entry.stroke.upTime === null) entry.stroke.upTime = now;
      if (entry.raw && entry.raw.up === null) entry.raw.up = now;
      this.pendingKeyDeletes = this.pendingKeyDeletes.filter((raw) => raw !== entry.raw && !raw.rep);
      this.pendingKeyStrokes = this.pendingKeyStrokes.filter((pending) => pending.code !== code || pending.composing);
      if (q.length === 0) this.activeDown.delete(code);
    });

    this.inputEl.addEventListener('beforeinput', (e) => {
      if (this.inputEl.value !== this.lastInputValue) this.reset();
      const { value, selectionStart, selectionEnd } = this.inputEl;
      if (selectionStart === null || selectionStart === undefined) return;
      this.beforeInput = {
        type: e.inputType,
        composing: e.isComposing,
        start: textToKeystrokes(value.slice(0, selectionStart)).length,
        end: textToKeystrokes(value.slice(0, selectionEnd)).length,
      };
    });

    this.inputEl.addEventListener('input', (e) => {
      const selection = this.beforeInput?.type === e.inputType ? this.beforeInput : null;
      this.beforeInput = null;
      this.lastInputValue = this.inputEl.value;
      this._syncFromValue(eventTime(e), e.inputType || '', selection);
      if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
    });
  }

  /**
   * Diff the field value (in keystroke space) and add strokes no keydown produced.
   * A deletion that no Backspace keydown explains is held for IME_PAIR_MS: if the IME re-inserts
   * text right away (composition replacement, final consonant moving on), the whole chain is applied
   * as one net change, so composing Hangul never looks like a typo. Only a deletion that stays a
   * deletion is a (soft-keyboard) Backspace.
   */
  _syncFromValue(now, inputType, selection) {
    const nextKeys = textToKeystrokes(this.lastInputValue);
    this.pendingKeyDeletes = this.pendingKeyDeletes.filter(({ t, direction }) =>
      now - t <= PENDING_TTL_MS && (!inputType || inputType.endsWith(direction) || inputType.includes('Composition')));
    if (this.imeChain) {
      const last = this.imeChain.events[this.imeChain.events.length - 1];
      const consecutiveDeletes = inputType.endsWith('Backward') && last.inputType.endsWith('Backward');
      if (now - last.t <= IME_PAIR_MS && this.pendingKeyDeletes.length === 0 && !consecutiveDeletes) {
        this.imeChain.events.push({ t: now, keys: nextKeys, inputType });
        this._armImeTimer();
        return;
      }
      this._resolveImeChain();
    }

    const diff = changedKeys(this.prevKeys, nextKeys, selection, inputType);
    const compositionDelete = inputType === 'deleteCompositionText' || inputType === 'deleteByComposition';
    const hangulShortened = diff.added.length === 0 && !inputType.endsWith('Forward') &&
      this.prevKeys.slice(diff.start, diff.end).some((key) => HANGUL_JAMO_RE.test(key));
    if (this.pendingKeyDeletes.length === 0 && (compositionDelete || hangulShortened)) {
      this.imeChain = { selection, events: [{ t: now, keys: nextKeys, inputType }], timer: null };
      this._armImeTimer();
      return;
    }
    this._applyDiff(diff, nextKeys, now, inputType);
  }

  _armImeTimer() {
    if (this.imeChain.timer) clearTimeout(this.imeChain.timer);
    // Wall-clock fallback for a deletion nothing follows (event timestamps decide everything else)
    this.imeChain.timer = setTimeout(() => {
      if (this.imeChain) {
        this._resolveImeChain();
        if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
      }
    }, IME_PAIR_MS * 3);
  }

  /** Apply a held deletion chain as one net change. */
  _resolveImeChain() {
    const chain = this.imeChain;
    if (!chain) return;
    if (chain.timer) clearTimeout(chain.timer);
    this.imeChain = null;
    const first = chain.events[0];
    const final = chain.events[chain.events.length - 1];
    if (chain.events.length === 1) {
      const diff = changedKeys(this.prevKeys, final.keys, chain.selection, first.inputType);
      this._applyDiff(diff, final.keys, first.t, first.inputType);
      return;
    }
    const diff = changedKeys(this.prevKeys, final.keys, chain.selection, 'insertCompositionText');
    const times = diff.added.map((_, i) => {
      const prefix = final.keys.slice(0, diff.start + i + 1);
      return chain.events.find(({ keys }) => prefix.every((key, j) => key === keys[j]))?.t ?? final.t;
    });
    this._applyDiff(diff, final.keys, first.t, 'insertCompositionText', times);
  }

  _applyDiff({ start, end, added }, nextKeys, now, inputType, addedTimes = null) {
    this.prevKeys = nextKeys;
    if (start === end && added.length === 0) return;

    this.pendingKeyStrokes = this.pendingKeyStrokes.filter((e) => now - e.t <= PENDING_TTL_MS);
    const removed = end - start;
    const composition = inputType.includes('Composition');
    const isReplacement = removed > 0 && added.length > 0;
    if (removed > 0) {
      this.justEdited = true;
      for (const stroke of this.stack.slice(start, end)) {
        stroke.poppedAt = now;
        this.poppedStrokes.push(stroke);
      }
      const physicalDelete = this.pendingKeyDeletes.pop();
      this.pendingKeyDeletes = [];
      if (physicalDelete || (!composition && !isReplacement && !inputType.startsWith('insert'))) {
        this.rawEvents.push(physicalDelete || { kind: 'bs', side: null, t: now, up: null, rep: false });
        this.backspaceCount += 1;
      }
    }

    const times = addedTimes || added.map(() => now);
    const externalText = ['insertReplacementText', 'insertFromPaste', 'insertFromDrop'].includes(inputType) || inputType.startsWith('history');
    const bulk = (added.length > 1 && new Set(times).size !== times.length) ||
      (isReplacement && !composition) || externalText;
    const inserted = added.map((char, i) => {
      const t = times[i];
      const idx = externalText ? -1 : this.pendingKeyStrokes.findIndex((entry) =>
        entry.t <= t + IME_PAIR_MS && (entry.char === char || (entry.composing && (composition || HANGUL_JAMO_RE.test(char)))));
      const pending = idx >= 0 ? this.pendingKeyStrokes.splice(idx, 1)[0] : null;
      if (pending?.stroke) {
        pending.stroke.char = char;
        pending.stroke.label = formatKeyLabel(char, pending.stroke.code);
        return pending.stroke;
      }
      const synthetic = bulk || Boolean(pending);
      if (!synthetic) this.rawEvents.push({ kind: char === ' ' ? 'space' : 'char', side: null, t, up: null, rep: false });
      return this._makeStroke({ char, t, source: 'input', synthetic });
    });
    if (end < this.stack.length) {
      const suffix = this.stack[end];
      suffix.cleanFromPrev = false;
      suffix.editedBefore = true;
    }
    let prev = this.stack[start - 1];
    for (const stroke of inserted) {
      this._joinStroke(stroke, prev);
      prev = stroke;
    }
    this.stack.splice(start, removed, ...inserted);
    if (nextKeys.length === 0) this._clearTextState();
  }

  getLiveStats() {
    const text = this.inputEl.value;
    let cpm = 0;
    if (this.firstDownTime && this.lastDownTime && this.lastDownTime > this.firstDownTime) {
      const elapsedMin = (this.lastDownTime - this.firstDownTime) / 60000;
      if (elapsedMin > 0.004) {
        cpm = Math.round(this.stack.length / elapsedMin);
      }
    }
    return {
      text,
      cpm,
      backspaceCount: this.backspaceCount,
      hadPaste: this.hadPaste,
      keyCount: this.stack.length,
    };
  }

  /** Behaviour events in seconds (holds only where a real keyup was observed). */
  getBehaviorEvents() {
    return this.rawEvents.map((r) => {
      let hold = r.up === null || r.up === undefined ? null : (r.up - r.t) / 1000.0;
      if (hold !== null && (hold < MIN_VALID_HOLD_SEC || (r.kind !== 'bs' && hold > 5.0))) hold = null;
      return { kind: r.kind, side: r.side, t: r.t / 1000.0, hold, rep: r.rep };
    });
  }

  /**
   * Build visualizable keystroke timeline (surviving strokes + popped typos + stitch points)
   * with display timestamps compressed across long pauses so Piano-Roll stays readable.
   */
  buildVisualTimeline() {
    if (this.stack.length === 0) return { events: [], poppedEvents: [] };
    const t0 = this.stack[0].downTime;
    const events = [];
    let compressedOffset = 0;

    for (let i = 0; i < this.stack.length; i++) {
      const cur = this.stack[i];
      const prev = i > 0 ? this.stack[i - 1] : null;
      const up = cur.upTime ?? cur.downTime + 80.0;
      const holdMs = Math.max(12, Math.min(600, up - cur.downTime));

      let rawGapFromPrevDown = prev ? cur.downTime - prev.downTime : 0;
      if (prev && (!cur.cleanFromPrev || rawGapFromPrevDown > 520)) {
        const visualGap = 155;
        compressedOffset += rawGapFromPrevDown - visualGap;
      }

      const startMs = Math.max(0, cur.downTime - t0 - compressedOffset);
      const endMs = startMs + holdMs;
      const prevEvent = events.length > 0 ? events[events.length - 1] : null;
      const udMs =
        prev && cur.cleanFromPrev ? cur.downTime - (prev.upTime ?? prev.downTime + 80.0) : null;
      const ddMs = prev && cur.cleanFromPrev ? cur.downTime - prev.downTime : null;

      events.push({
        idx: i,
        code: cur.code,
        label: cur.label || formatKeyLabel(cur.char || '', cur.code),
        startMs: Math.round(startMs),
        endMs: Math.round(endMs),
        holdMs: Math.round(holdMs),
        holdKnown: holdSec(cur) !== null,
        ddMs: ddMs !== null ? Math.round(ddMs) : null,
        udMs: udMs !== null && holdSec(prev) !== null ? Math.round(udMs) : null,
        isRollover: udMs !== null && holdSec(prev) !== null && udMs < 0,
        stitchedBefore: i > 0 && !cur.cleanFromPrev,
        editedBefore: Boolean(cur.editedBefore),
        visualOverlapPrev: prevEvent && cur.cleanFromPrev && startMs < prevEvent.endMs,
      });
    }

    return {
      events,
      poppedCount: this.poppedStrokes.length,
    };
  }

  /**
   * Reconstruct the clean digraph sequence from surviving strokes, bridging over
   * Backspace-repaired mistakes or long pauses. Unknown holds stay `null` and are imputed
   * (and flagged) when the model tensor is built.
   */
  /**
   * `resolvePending: false` peeks without settling an IME edit that is still waiting for its
   * re-insert (live previews call this on every input event; settling there would turn every
   * composition step back into a deletion).
   */
  extractCleanStroke({ resolvePending = true } = {}) {
    if (resolvePending) {
      if (this.inputEl.value !== this.lastInputValue) this.reset();
      this._resolveImeChain();
    }
    const uniqueHolds = [];
    const uniqueHoldValid = [];
    const hCurr = [];
    const hNext = [];
    const hvCurr = [];
    const hvNext = [];
    const dds = [];
    const uds = [];
    const stitchFlags = [];
    const pairKeys = [];
    let prevAcceptedNextIdx = null;

    for (let i = 0; i < this.stack.length - 1; i++) {
      const c0 = this.stack[i];
      const c1 = this.stack[i + 1];
      if (!c1.cleanFromPrev || c0.synthetic || c1.synthetic) continue;

      const h0 = holdSec(c0);
      const h1 = holdSec(c1);
      if ((h0 !== null && h0 > MAX_HOLD_SEC) || (h1 !== null && h1 > MAX_HOLD_SEC)) continue;
      const dd = (c1.downTime - c0.downTime) / 1000.0;
      if (dd < 0.006 || dd > 2.2) continue;
      const ud = h0 !== null ? (c1.downTime - c0.upTime) / 1000.0 : null;

      const isNewSegment = dds.length === 0 || prevAcceptedNextIdx !== i;
      const isStitched = dds.length > 0 && (prevAcceptedNextIdx !== i || !c0.cleanFromPrev);
      hCurr.push(h0);
      hNext.push(h1);
      hvCurr.push(h0 !== null);
      hvNext.push(h1 !== null);
      if (isNewSegment) {
        uniqueHolds.push(h0);
        uniqueHoldValid.push(h0 !== null);
      }
      uniqueHolds.push(h1);
      uniqueHoldValid.push(h1 !== null);
      dds.push(dd);
      uds.push(ud);
      stitchFlags.push(isStitched);
      pairKeys.push([c0.key, c1.key]);
      prevAcceptedNextIdx = i + 1;
    }

    const knownHolds = uniqueHolds.filter((h) => h !== null);
    const knownUds = uds.filter((u) => u !== null);
    const meanHoldMs = knownHolds.length
      ? Math.round((knownHolds.reduce((a, b) => a + b, 0) / knownHolds.length) * 1000)
      : 0;
    const meanDdSec = dds.length > 0 ? dds.reduce((a, b) => a + b, 0) / dds.length : 0;
    const cpm = meanDdSec > 0 ? Math.round(60.0 / meanDdSec) : 0;
    const rolloverPct = knownUds.length
      ? Math.round((knownUds.filter((u) => u < 0).length / knownUds.length) * 100)
      : 0;
    const validFrac = uniqueHoldValid.length
      ? uniqueHoldValid.filter(Boolean).length / uniqueHoldValid.length
      : 1;
    const inputMode = validFrac >= 0.8 ? 'keyboard' : validFrac <= 0.2 ? 'touch' : 'mixed';

    return {
      typedText: this.inputEl.value,
      hadPaste: this.hadPaste,
      backspaceCount: this.backspaceCount,
      cleanDigraphCount: dds.length,
      cpm,
      holdMs: meanHoldMs,
      rolloverPct,
      inputMode,
      behavior: summarizeBehavior(this.getBehaviorEvents()),
      capitalMethods: this.stack.filter((s) => s.capital).map((s) => s.capital),
      timeline: this.buildVisualTimeline(),
      stroke: {
        holds: uniqueHolds,
        holdValid: uniqueHoldValid,
        hCurr,
        hNext,
        hvCurr,
        hvNext,
        dds,
        uds,
        stitchFlags,
        pairKeys,
      },
    };
  }
}
