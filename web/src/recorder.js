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

/**
 * Device-agnostic keystroke recorder.
 *
 * Physical keyboards: every printable keydown becomes a stroke with its physical code and a
 * real hold time (keydown -> keyup). Backspace pops the mistyped stroke and marks a stitch
 * boundary so only the interrupted transition is dropped.
 *
 * Touch keyboards (Android Gboard, iOS) usually report `code === ''` / key 'Unidentified'
 * and fire keyup together with keydown, which made every phone capture empty
 * ("추출된 연속 리듬 구간이 부족합니다"). Text changes are therefore also read from `input`
 * events: the value is expanded into keystrokes (Hangul syllables -> 2-beolsik jamo keys),
 * diffed against the previous value, and keystrokes not already produced by a keydown become
 * "soft" strokes timed at the input event with an unknown hold (Hold_Valid = 0).
 *
 * In parallel every key action is appended to a raw event log used by the behavioural
 * profile (Backspace habits, Shift/CapsLock strategy, pause rhythm).
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
    this.stack = []; // Surviving strokes: { code, char, key, downTime, upTime, source, cleanFromPrev, editedBefore, pausedBefore, synthetic }
    this.poppedStrokes = [];
    this.rawEvents = []; // Behaviour log: { kind, side, t, up, rep }
    this.activeDown = new Map(); // code -> FIFO of { stroke?, raw } awaiting keyup
    this.pendingKeyStrokes = []; // { t, stroke } keydown strokes awaiting their input event
    this.pendingKeyDeletes = [];
    this.pendingRepeats = [];
    if (this.imeChain?.timer) clearTimeout(this.imeChain.timer);
    this.imeChain = null; // deletion without a Backspace keydown, waiting for its IME re-insert
    this.prevKeys = textToKeystrokes(this.inputEl?.value || '');
    this.justEdited = false;
    this.backspaceCount = 0;
    this.totalKeyPresses = 0;
    this.hadPaste = false;
    this.firstDownTime = null;
    this.lastDownTime = null;
  }

  // Field erased back to empty: rhythm stack starts over, behaviour log is kept
  _clearTextState() {
    this.stack = [];
    this.poppedStrokes = [];
    this.justEdited = false;
    this.backspaceCount = 0;
    this.firstDownTime = null;
    this.lastDownTime = null;
  }

  _pushStroke({ code, char, t, source, synthetic = false }) {
    const prev = this.stack[this.stack.length - 1];
    const hadLongPause = Boolean(prev) && t - prev.downTime > LONG_PAUSE_MS;
    const entry = {
      code: code || null,
      char: char ?? null,
      key: code ? code : `char:${char}`,
      label: formatKeyLabel(char || '', code),
      downTime: t,
      upTime: null,
      source,
      synthetic,
      editedBefore: Boolean(prev) && this.justEdited,
      pausedBefore: hadLongPause,
      cleanFromPrev: !prev ? true : !this.justEdited && !hadLongPause && !prev.synthetic && !synthetic,
    };
    if (!synthetic) {
      if (this.firstDownTime === null) this.firstDownTime = t;
      this.lastDownTime = t;
      this.totalKeyPresses += 1;
      this.justEdited = false;
    }
    this.stack.push(entry);
    return entry;
  }

  _markDown(code, entry) {
    const q = this.activeDown.get(code) || [];
    q.push(entry);
    this.activeDown.set(code, q);
  }

  _popStroke(t) {
    this.justEdited = true;
    if (this.stack.length > 0) {
      const popped = this.stack.pop();
      popped.poppedAt = t;
      this.poppedStrokes.push(popped);
    }
  }

  _bind() {
    this.inputEl.addEventListener('paste', (e) => {
      e.preventDefault();
      this.hadPaste = true;
      if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
    });

    this.inputEl.addEventListener('keydown', (e) => {
      const now = eventTime(e);

      if (e.key === 'Enter') {
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
        const raw = { kind: 'bs', side: null, t: now, up: null, rep: Boolean(e.repeat) };
        this.rawEvents.push(raw);
        if (!e.repeat) this._markDown(code || 'Backspace', { raw });
        this.backspaceCount += 1;
        this._popStroke(now);
        this.pendingKeyDeletes.push(now);
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
      if (e.repeat) {
        this.pendingRepeats.push(now); // OS auto-repeat inserts text but is not a keystroke
        return;
      }

      // If input was cleared externally (e.g. select-all + delete), resync stack
      if (this.inputEl.value.length === 0 && this.stack.length > 3) {
        this.stack = [];
        this.poppedStrokes = [];
        this.justEdited = false;
      }

      const stroke = this._pushStroke({ code, char: e.key && [...e.key].length === 1 ? e.key : null, t: now, source: 'key' });
      stroke.label = formatKeyLabel(e.key, code);
      // Capital letters: was it typed with CapsLock or with Shift? (a stable personal habit)
      if (/^[A-Z]$/.test(e.key || '')) {
        stroke.capital = e.getModifierState?.('CapsLock') ? 'caps' : e.shiftKey ? 'shift' : null;
      }
      const raw = { kind: code === 'Space' ? 'space' : 'char', side: null, t: now, up: null, rep: false };
      this.rawEvents.push(raw);
      this._markDown(code, { stroke, raw });
      this.pendingKeyStrokes.push({ t: now, stroke });
    });

    this.inputEl.addEventListener('keyup', (e) => {
      const now = eventTime(e);
      const code = e.code || (e.key === 'Shift' ? 'Shift' : e.key === 'CapsLock' ? 'CapsLock' : e.key === 'Backspace' ? 'Backspace' : '');
      const q = this.activeDown.get(code);
      if (!q || q.length === 0) return;
      const entry = q.shift();
      if (entry.stroke && entry.stroke.upTime === null) entry.stroke.upTime = now;
      if (entry.raw && entry.raw.up === null) entry.raw.up = now;
      if (q.length === 0) this.activeDown.delete(code);
    });

    this.inputEl.addEventListener('input', (e) => {
      this._syncFromValue(eventTime(e), e.inputType || '');
      if (this.inputEl.value.length === 0 && this.backspaceCount > 0) {
        // User erased everything back to empty — clean slate for the rhythm stack
        this._clearTextState();
      }
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
  _syncFromValue(now, inputType) {
    const nextKeys = textToKeystrokes(this.inputEl.value);
    if (this.imeChain) {
      const last = this.imeChain.events[this.imeChain.events.length - 1];
      if (now - last.t <= IME_PAIR_MS) {
        this.imeChain.events.push({ t: now, keys: nextKeys, inputType });
        this._armImeTimer();
        return;
      }
      this._resolveImeChain();
    }

    const prev = this.prevKeys;
    let p = 0;
    while (p < prev.length && p < nextKeys.length && prev[p] === nextKeys[p]) p++;
    const removed = prev.length - p;
    const added = nextKeys.length - p;
    const pendingDeletes = this.pendingKeyDeletes.filter((t) => now - t <= PENDING_TTL_MS).length;
    const compositionDelete = inputType === 'deleteCompositionText' || inputType === 'deleteByComposition';
    if ((removed > pendingDeletes && added === 0) || compositionDelete) {
      this.imeChain = { base: prev, events: [{ t: now, keys: nextKeys, inputType }], timer: null };
      this._armImeTimer();
      return;
    }
    this._applyDiff(prev, nextKeys, now, inputType);
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
    const fk = final.keys;
    let p = 0;
    while (p < chain.base.length && p < fk.length && chain.base[p] === fk[p]) p++;
    if (chain.events.length === 1 || fk.length === p) {
      // Nothing re-inserted: a real deletion (composition deletes never count as Backspace)
      const compositionDelete = first.inputType === 'deleteCompositionText' || first.inputType === 'deleteByComposition';
      this._applyDiff(chain.base, fk, first.t, compositionDelete ? 'insertCompositionText' : first.inputType || 'deleteContentBackward');
      return;
    }
    // Each net-added key gets the time of the first event in which it appeared
    const times = [];
    for (let j = p; j < fk.length; j++) {
      const ev = chain.events.find((e) => {
        let q = 0;
        while (q < e.keys.length && q < fk.length && e.keys[q] === fk[q]) q++;
        return q > j;
      });
      times.push(ev ? ev.t : final.t);
    }
    this._applyDiff(chain.base, fk, first.t, 'insertCompositionText', { addedTimes: times, imeChain: true });
  }

  _applyDiff(prev, nextKeys, now, inputType, { addedTimes = null, imeChain = false } = {}) {
    let p = 0;
    while (p < prev.length && p < nextKeys.length && prev[p] === nextKeys[p]) p++;
    const removed = prev.length - p;
    const added = nextKeys.slice(p);
    this.prevKeys = nextKeys;

    const fresh = (q) => q.filter((t) => now - t <= PENDING_TTL_MS);
    this.pendingKeyStrokes = this.pendingKeyStrokes.filter((e) => now - e.t <= PENDING_TTL_MS);
    this.pendingKeyDeletes = fresh(this.pendingKeyDeletes);
    this.pendingRepeats = fresh(this.pendingRepeats);

    // Deletions not already handled by a Backspace keydown (soft keyboards, Ctrl+Backspace, autocorrect)
    const handledDeletes = Math.min(removed, this.pendingKeyDeletes.length);
    this.pendingKeyDeletes.splice(0, handledDeletes);
    const extraDeletes = removed - handledDeletes;
    const isReplacement = extraDeletes > 0 && added.length > 0;
    if (extraDeletes > 0) {
      for (let i = 0; i < extraDeletes; i++) this._popStroke(now);
      if (!isReplacement && handledDeletes === 0 && !inputType.startsWith('insert')) {
        // One user deletion action without a physical Backspace keydown (touch keyboards)
        this.rawEvents.push({ kind: 'bs', side: null, t: now, up: null, rep: false });
        this.backspaceCount += 1;
      }
    }

    // Insertions: first match keydown-created strokes, then auto-repeats, the rest are soft
    // Pair each added key with the oldest waiting keydown stroke (by identity: a held IME edit can
    // settle after the next key is already down, so "the last N strokes" would be the wrong ones)
    const matched = Math.min(added.length, this.pendingKeyStrokes.length);
    this.pendingKeyStrokes.splice(0, matched).forEach(({ stroke: s }, i) => {
      // The character the field actually received is authoritative: IME keydowns report
      // key = 'Process', so the label would otherwise stay the Latin key name (D, K, …)
      if (s && s.char !== added[i]) {
        s.char = added[i];
        s.label = formatKeyLabel(added[i], s.code);
      }
    });
    let rest = added.slice(matched);
    let restTimes = (addedTimes || added.map(() => now)).slice(matched);
    const repeats = Math.min(rest.length, this.pendingRepeats.length);
    this.pendingRepeats.splice(0, repeats);
    for (let i = 0; i < repeats; i++) this._pushStroke({ code: null, char: rest[i], t: restTimes[i], source: 'input', synthetic: true });
    rest = rest.slice(repeats);
    restTimes = restTimes.slice(repeats);

    // Several keys in one edit are synthetic (autocomplete, paste) unless an IME chain spread them
    // over distinct taps; a replacement inside an IME chain is a re-segmentation, not autocorrect.
    const distinctTimes = new Set(restTimes).size === restTimes.length;
    const bulk =
      (rest.length > 1 && !(imeChain && distinctTimes)) ||
      (isReplacement && !imeChain) ||
      inputType === 'insertReplacementText' ||
      inputType === 'insertFromPaste';
    rest.forEach((ch, i) => {
      if (bulk) {
        this._pushStroke({ code: null, char: ch, t: restTimes[i], source: 'input', synthetic: true });
      } else {
        this._pushStroke({ code: null, char: ch, t: restTimes[i], source: 'input' });
        this.rawEvents.push({ kind: ch === ' ' ? 'space' : 'char', side: null, t: restTimes[i], up: null, rep: false });
      }
    });
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
    if (resolvePending) this._resolveImeChain();
    const holds = [];
    const holdValid = [];
    const dds = [];
    const uds = [];
    const stitchFlags = [];
    const pairKeys = [];

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

      if (holds.length === 0) {
        holds.push(h0);
        holdValid.push(h0 !== null);
      }
      holds.push(h1);
      holdValid.push(h1 !== null);
      dds.push(dd);
      uds.push(ud);
      stitchFlags.push(i > 0 && !c0.cleanFromPrev);
      pairKeys.push([c0.key, c1.key]);
    }

    const knownHolds = holds.filter((h) => h !== null);
    const knownUds = uds.filter((u) => u !== null);
    const meanHoldMs = knownHolds.length
      ? Math.round((knownHolds.reduce((a, b) => a + b, 0) / knownHolds.length) * 1000)
      : 0;
    const meanDdSec = dds.length > 0 ? dds.reduce((a, b) => a + b, 0) / dds.length : 0;
    const cpm = meanDdSec > 0 ? Math.round(60.0 / meanDdSec) : 0;
    const rolloverPct = knownUds.length
      ? Math.round((knownUds.filter((u) => u < 0).length / knownUds.length) * 100)
      : 0;
    const validFrac = holdValid.length ? holdValid.filter(Boolean).length / holdValid.length : 1;
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
        holds,
        holdValid,
        dds,
        uds,
        stitchFlags,
        pairKeys,
      },
    };
  }
}
