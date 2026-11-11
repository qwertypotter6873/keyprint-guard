import * as ort from 'onnxruntime-web';

// Single-threaded SIMD WASM avoids SharedArrayBuffer COOP/COEP issues on GitHub Pages
ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
ort.env.wasm.numThreads = 1;

const PRINTABLE_CODE_RE =
  /^(Key[A-Z]|Digit[0-9]|Space|Period|Comma|Slash|Semicolon|Quote|BracketLeft|BracketRight|Minus|Equal|Backquote|Backslash|Numpad[0-9]|NumpadDecimal|NumpadAdd|NumpadSubtract)$/;

/**
 * Character-level accuracy (%) between typed string and target prompt.
 */
export function computeTypingAccuracy(typed, target) {
  const a = String(typed || '');
  const b = String(target || '');
  if (!a.length) return 100;
  const len = Math.max(a.length, b.length);
  let matches = 0;
  const minLen = Math.min(a.length, b.length);
  for (let i = 0; i < minLen; i++) {
    if (a[i] === b[i]) matches++;
  }
  return Math.round((matches / len) * 100);
}

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

function formatKeyLabel(key, code) {
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

/**
 * Backspace-resilient physical keystroke recorder.
 * - Captures sub-ms physical scan codes (works identically for English & Korean 2-beolsik IME).
 * - Maintains a live keystroke stack: when Backspace/Delete is pressed, pops the mistyped stroke
 *   and marks a clean stitch boundary (`justEdited = true`) so the contaminated edit gap is
 *   isolated while all valid neuromuscular digraphs before and after are preserved!
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
    this.stack = []; // Surviving physical strokes: { code, label, downTime, upTime, cleanFromPrev, editedBefore, pausedBefore }
    this.poppedStrokes = []; // Mistyped strokes removed via Backspace (for visualization)
    this.activeDown = new Map(); // code -> stack reference
    this.justEdited = false;
    this.backspaceCount = 0;
    this.totalKeyPresses = 0;
    this.hadPaste = false;
    this.firstDownTime = null;
    this.lastDownTime = null;
  }

  _bind() {
    this.inputEl.addEventListener('paste', (e) => {
      e.preventDefault();
      this.hadPaste = true;
      if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
    });

    this.inputEl.addEventListener('keydown', (e) => {
      const now = performance.now();

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

      if (e.key === 'Backspace' || e.key === 'Delete') {
        this.backspaceCount += 1;
        this.justEdited = true;
        if (this.stack.length > 0) {
          const popped = this.stack.pop();
          popped.poppedAt = now;
          this.poppedStrokes.push(popped);
        }
        return;
      }

      // Ignore OS auto-repeat and pure modifier keys (Shift, Ctrl, Alt, Meta, Arrows)
      if (e.repeat) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (!PRINTABLE_CODE_RE.test(e.code)) return;

      // If input was cleared externally (e.g. select-all + delete), resync stack
      if (this.inputEl.value.length === 0 && this.stack.length > 3) {
        this.stack = [];
        this.poppedStrokes = [];
        this.justEdited = false;
      }

      // Mark long pause (> 2.2s) as a stitch boundary instead of throwing an error
      const hadLongPause =
        this.stack.length > 0 && now - this.stack[this.stack.length - 1].downTime > 2200;

      const entry = {
        code: e.code,
        label: formatKeyLabel(e.key, e.code),
        downTime: now,
        upTime: null,
        editedBefore: this.stack.length > 0 && this.justEdited,
        pausedBefore: hadLongPause,
        cleanFromPrev: this.stack.length === 0 ? true : !this.justEdited && !hadLongPause,
      };

      if (this.firstDownTime === null) this.firstDownTime = now;
      this.lastDownTime = now;
      this.totalKeyPresses += 1;
      this.justEdited = false;

      this.stack.push(entry);
      this.activeDown.set(e.code, entry);
    });

    this.inputEl.addEventListener('keyup', (e) => {
      const now = performance.now();
      const entry = this.activeDown.get(e.code);
      if (entry && entry.upTime === null) {
        entry.upTime = now;
        this.activeDown.delete(e.code);
      }
    });

    this.inputEl.addEventListener('input', () => {
      if (this.inputEl.value.length === 0 && this.backspaceCount > 0) {
        // User erased everything back to empty — give a clean slate automatically
        const pasted = this.hadPaste;
        this.reset();
        this.hadPaste = pasted;
      }
      if (this.onLiveUpdate) this.onLiveUpdate(this.getLiveStats());
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
        // Compress long thinking/editing pause on visual timeline while marking stitch!
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
        label: cur.label || formatKeyLabel('', cur.code),
        startMs: Math.round(startMs),
        endMs: Math.round(endMs),
        holdMs: Math.round(holdMs),
        ddMs: ddMs !== null ? Math.round(ddMs) : null,
        udMs: udMs !== null ? Math.round(udMs) : null,
        isRollover: udMs !== null && udMs < 0,
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
   * Reconstruct clean consecutive digraph sequence from surviving stack strokes,
   * seamlessly bridging over any Backspace-repaired mistakes or mid-sentence pauses.
   */
  extractCleanStroke() {
    const holds = [];
    const dds = [];
    const uds = [];
    const stitchFlags = [];

    for (let i = 0; i < this.stack.length - 1; i++) {
      const c0 = this.stack[i];
      const c1 = this.stack[i + 1];
      if (!c1.cleanFromPrev) {
        // Skip ONLY the single transition interrupted by Backspace or a long pause
        continue;
      }

      const up0 = c0.upTime ?? c0.downTime + 80.0;
      const up1 = c1.upTime ?? c1.downTime + 80.0;
      const h0 = (up0 - c0.downTime) / 1000.0;
      const h1 = (up1 - c1.downTime) / 1000.0;
      const dd = (c1.downTime - c0.downTime) / 1000.0;
      const ud = (c1.downTime - up0) / 1000.0;

      if (h0 < 0.005 || h0 > 1.2 || h1 < 0.005 || h1 > 1.2) continue;
      if (dd < 0.006 || dd > 2.2) continue;

      if (holds.length === 0) {
        holds.push(h0);
      }
      holds.push(h1);
      dds.push(dd);
      uds.push(ud);
      stitchFlags.push(i > 0 && !c0.cleanFromPrev);
    }

    const meanHoldMs =
      holds.length > 0
        ? Math.round((holds.reduce((a, b) => a + b, 0) / holds.length) * 1000)
        : 0;
    const meanDdSec = dds.length > 0 ? dds.reduce((a, b) => a + b, 0) / dds.length : 0;
    const cpm = meanDdSec > 0 ? Math.round(60.0 / meanDdSec) : 0;
    const rolloverPct =
      uds.length > 0 ? Math.round((uds.filter((u) => u < 0).length / uds.length) * 100) : 0;

    return {
      typedText: this.inputEl.value,
      hadPaste: this.hadPaste,
      backspaceCount: this.backspaceCount,
      cleanDigraphCount: dds.length,
      cpm,
      holdMs: meanHoldMs,
      rolloverPct,
      timeline: this.buildVisualTimeline(),
      stroke: {
        holds,
        dds,
        uds,
        stitchFlags,
      },
    };
  }
}

export function validatePasswordCapture(capture, expectedPassword) {
  if (!expectedPassword || expectedPassword.length < 6) {
    return {
      ok: false,
      code: 'SHORT_PASSWORD',
      message: '비밀번호는 최소 6자 이상으로 설정해 주세요.',
    };
  }
  if (capture.hadPaste) {
    return {
      ok: false,
      code: 'PASTE_BLOCKED',
      message: '붙여넣기는 사용할 수 없습니다. 직접 키보드로 타이핑해 주세요.',
    };
  }
  if (capture.typedText !== expectedPassword) {
    return {
      ok: false,
      code: 'PASSWORD_MISMATCH',
      message: '비밀번호 문자열이 일치하지 않습니다. 철자를 확인하고 다시 입력해 주세요.',
    };
  }
  if (capture.cleanDigraphCount < 4) {
    return {
      ok: false,
      code: 'INSUFFICIENT_CLEAN_KEYS',
      message: '유효한 연속 타건 구간이 너무 적습니다(과도한 수정). 한 번만 자연스럽게 다시 입력해 주세요.',
    };
  }
  return { ok: true };
}

export function validateSentenceCapture(capture, targetSentence) {
  if (capture.hadPaste) {
    return {
      ok: false,
      code: 'PASTE_BLOCKED',
      message: '붙여넣기는 사용할 수 없습니다. 타자연습 문장을 직접 타이핑해 주세요.',
    };
  }
  const typedTrim = capture.typedText.trim();
  const targetTrim = String(targetSentence || '').trim();
  const minChars = Math.max(8, Math.floor(targetTrim.length * 0.72));
  if (typedTrim.length < minChars) {
    return {
      ok: false,
      code: 'INCOMPLETE_SENTENCE',
      message: `문장을 끝까지 타이핑한 뒤 Enter를 눌러주세요 (${typedTrim.length} / ${targetTrim.length}자).`,
    };
  }
  if (capture.cleanDigraphCount < 8) {
    return {
      ok: false,
      code: 'INSUFFICIENT_CLEAN_KEYS',
      message: '추출된 연속 리듬 구간이 부족합니다. 문장을 자연스럽게 이어서 타이핑해 주세요.',
    };
  }
  return {
    ok: true,
    accuracy: computeTypingAccuracy(typedTrim, targetTrim),
  };
}

/**
 * Exact JavaScript counterpart of src/preprocessing.py with Stitch-Boundary Acceleration Reset.
 */
export function build9ChannelTensor(stroke, config) {
  const { holds, dds, uds, stitchFlags = [] } = stroke;
  const m = dds.length;
  const bounds = config.outlier_bounds;
  const eps = 1e-4;
  const clip = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

  const cHolds = holds.map((v) => clip(v, bounds.H));
  const cDds = dds.map((v) => clip(v, bounds.DD));
  const cUds = uds.map((v) => clip(v, bounds.UD));

  const meanH = Math.max(eps, cHolds.reduce((a, b) => a + b, 0) / cHolds.length);
  const meanDD = Math.max(eps, cDds.reduce((a, b) => a + b, 0) / cDds.length);

  const data = new Float32Array(9 * m);
  const chMean = config.channel_mean;
  const chStd = config.channel_std;

  for (let t = 0; t < m; t++) {
    const hCurr = cHolds[t];
    const hNext = cHolds[t + 1];
    const dd = cDds[t];
    const ud = cUds[t];
    const relH = hCurr / meanH;
    const relDD = dd / meanDD;
    const dutyRatio = Math.min(5.0, Math.max(0.0, hCurr / Math.max(dd, 0.02)));
    const rolloverMag = Math.max(0.0, -ud);
    // Reset Local_Accel at Backspace/pause stitch boundaries so zero artificial jump enters CNN
    const localAccel = t === 0 || stitchFlags[t] ? 0.0 : dd - cDds[t - 1];

    const rawChannels = [
      hCurr,
      hNext,
      dd,
      ud,
      relH,
      relDD,
      dutyRatio,
      rolloverMag,
      localAccel,
    ];

    for (let c = 0; c < 9; c++) {
      data[c * m + t] = (rawChannels[c] - chMean[c]) / chStd[c];
    }
  }

  return {
    tensorData: data,
    dims: [1, 9, m],
    cleanStroke: { holds: cHolds, dds: cDds, uds: cUds },
  };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Merge a newly embedded sample into a variable-length template (works for both
 * fixed passwords and diverse multi-sentence typing practice!).
 */
export function mergeTemplateSample(existing, { embedding, cpm, holdMs, rolloverPct }) {
  const prevExemplars = existing?.exemplars ? [...existing.exemplars] : [];
  prevExemplars.push(embedding.map((v) => Math.round(v * 100000) / 100000));
  // Keep up to 12 most recent exemplars so synced payload stays tiny (~2 KB)
  const exemplars = prevExemplars.slice(-12);

  const dim = embedding.length;
  const centroid = new Array(dim).fill(0);
  for (const ex of exemplars) {
    for (let d = 0; d < dim; d++) centroid[d] += ex[d];
  }
  const cNorm = Math.hypot(...centroid) || 1e-8;
  for (let d = 0; d < dim; d++) {
    centroid[d] = Math.round((centroid[d] / cNorm) * 100000) / 100000;
  }

  const prevCount = existing?.count || 0;
  const nextCount = prevCount + 1;
  const blend = (oldVal, newVal) =>
    prevCount === 0 ? newVal : Math.round((oldVal * prevCount + newVal) / nextCount);

  return {
    centroid,
    exemplars,
    count: nextCount,
    cpm: blend(existing?.cpm || 0, cpm || 0),
    holdMs: blend(existing?.holdMs || 0, holdMs || 0),
    rolloverPct: blend(existing?.rolloverPct || 0, rolloverPct || 0),
  };
}

export class KeyPrintEngine {
  constructor() {
    this.session = null;
    this.config = null;
    this.loadTimeMs = 0;
  }

  async init(modelUrl = './models/keyprint_net.onnx', configUrl = './models/model_config.json') {
    const t0 = performance.now();
    const [cfgResp, session] = await Promise.all([
      fetch(configUrl).then((r) => {
        if (!r.ok) throw new Error(`설정 로드 실패: HTTP ${r.status}`);
        return r.json();
      }),
      ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      }),
    ]);
    this.config = cfgResp;
    this.session = session;

    const dummy = new ort.Tensor('float32', new Float32Array(90), [1, 9, 10]);
    await this.session.run({ keystroke_seq: dummy });
    this.loadTimeMs = Math.round((performance.now() - t0) * 10) / 10;
    return this.config;
  }

  async embedStroke(stroke) {
    const t0 = performance.now();
    const { tensorData, dims, cleanStroke } = build9ChannelTensor(stroke, this.config);
    const inputTensor = new ort.Tensor('float32', tensorData, dims);
    const results = await this.session.run({ keystroke_seq: inputTensor });
    const rawEmb = Array.from(results.biometric_embedding.data);
    const norm = Math.hypot(...rawEmb) || 1e-8;
    const embedding = rawEmb.map((v) => v / norm);
    const latencyMs = Math.round((performance.now() - t0) * 100) / 100;
    return { embedding, latencyMs, cleanStroke };
  }

  scoreTemplate(embedding, template, { domain = 'password', threshold = 0.49 } = {}) {
    const cSim = dot(embedding, template.centroid);
    let maxExSim = cSim;
    if (Array.isArray(template.exemplars) && template.exemplars.length > 0) {
      maxExSim = -1;
      for (const ex of template.exemplars) {
        const s = dot(embedding, ex);
        if (s > maxExSim) maxExSim = s;
      }
    }
    const raw = 0.65 * cSim + 0.35 * maxExSim;
    let similarity = raw;

    const cohort =
      domain === 'sentence' && Array.isArray(this.config?.sentence_cohort_centroids)
        ? this.config.sentence_cohort_centroids
        : this.config?.cohort_centroids;

    if (Array.isArray(cohort) && cohort.length > 0) {
      const topM = Math.min(domain === 'sentence' ? 16 : 12, cohort.length);
      const uScores = cohort
        .map((r) => dot(r, template.centroid))
        .sort((a, b) => a - b)
        .slice(-topM);
      const qScores = cohort
        .map((r) => dot(r, embedding))
        .sort((a, b) => a - b)
        .slice(-topM);

      const meanArr = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
      const stdArr = (arr, mu) =>
        Math.sqrt(arr.reduce((a, b) => a + (b - mu) * (b - mu), 0) / arr.length) + 0.05;

      const muU = meanArr(uScores);
      const sdU = stdArr(uScores, muU);
      const muQ = meanArr(qScores);
      const sdQ = stdArr(qScores, muQ);

      const zAs = (raw - 0.5 * (muU + muQ)) / (0.5 * (sdU + sdQ));
      similarity = 1.0 / (1.0 + Math.exp(-0.85 * (zAs - 0.15)));
    }

    return {
      similarity,
      centroidSim: cSim,
      exemplarSim: maxExSim,
      accepted: similarity >= threshold,
      threshold,
    };
  }

  identifyTypist(embedding, candidates, { openSetThreshold = 0.44 } = {}) {
    if (!Array.isArray(candidates) || candidates.length === 0) return null;

    const scored = candidates.map((cand) => {
      const res = this.scoreTemplate(embedding, cand.template, {
        domain: 'sentence',
        threshold: openSetThreshold,
      });
      return {
        id: cand.id,
        name: cand.name,
        color: cand.color || '#2563eb',
        isSample: Boolean(cand.isSample),
        similarity: res.similarity,
        centroidSim: res.centroidSim,
        template: cand.template,
      };
    });

    scored.sort((a, b) => b.similarity - a.similarity);

    const temp = 8.5;
    const maxS = scored[0].similarity;
    const exps = scored.map((it) => Math.exp(temp * (it.similarity - maxS)));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1.0;
    scored.forEach((it, idx) => {
      it.posterior = exps[idx] / sumExp;
      it.rank = idx + 1;
    });

    const winner = scored[0];
    const identified = winner.similarity >= openSetThreshold;

    return {
      identified,
      winner,
      openSetThreshold,
      rankings: scored,
    };
  }
}
