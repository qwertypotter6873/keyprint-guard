/**
 * Pure feature code shared by the recorder, the ONNX engine and the parity test.
 * No DOM / ONNX dependencies so it runs in Node (web/scripts/parity_check.mjs).
 *
 * Exact JavaScript counterparts of:
 *   - src/capture_sim.py::build_input_tensor   -> buildInputTensor()
 *   - src/keyboard_layout.py::key_pair_features -> keyPairFeatures()
 *   - src/behavior_profile.py                   -> behavior*()
 */

// ---------------------------------------------------------------------------------------
// Hangul -> 2-beolsik keystrokes (soft keyboards only report composed syllables)
// ---------------------------------------------------------------------------------------
const CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
const JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
const JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
// Compound jamo that need two keystrokes on a 2-beolsik keyboard
const COMPOUND = {
  ㅘ: ['ㅗ', 'ㅏ'], ㅙ: ['ㅗ', 'ㅐ'], ㅚ: ['ㅗ', 'ㅣ'], ㅝ: ['ㅜ', 'ㅓ'], ㅞ: ['ㅜ', 'ㅔ'], ㅟ: ['ㅜ', 'ㅣ'], ㅢ: ['ㅡ', 'ㅣ'],
  ㄳ: ['ㄱ', 'ㅅ'], ㄵ: ['ㄴ', 'ㅈ'], ㄶ: ['ㄴ', 'ㅎ'], ㄺ: ['ㄹ', 'ㄱ'], ㄻ: ['ㄹ', 'ㅁ'], ㄼ: ['ㄹ', 'ㅂ'],
  ㄽ: ['ㄹ', 'ㅅ'], ㄾ: ['ㄹ', 'ㅌ'], ㄿ: ['ㄹ', 'ㅍ'], ㅀ: ['ㄹ', 'ㅎ'], ㅄ: ['ㅂ', 'ㅅ'],
};

function jamoKeys(j) {
  return COMPOUND[j] ? COMPOUND[j] : [j];
}

/** Expand text into the keystroke characters that produced it (Hangul -> jamo keys). */
export function textToKeystrokes(text) {
  const out = [];
  for (const ch of String(text || '')) {
    const cp = ch.codePointAt(0);
    if (cp >= 0xac00 && cp <= 0xd7a3) {
      const idx = cp - 0xac00;
      const cho = Math.floor(idx / 588);
      const jung = Math.floor((idx % 588) / 28);
      const jong = idx % 28;
      out.push(CHO[cho], ...jamoKeys(JUNG[jung]));
      if (jong) out.push(...jamoKeys(JONG[jong]));
    } else if (COMPOUND[ch]) {
      out.push(...COMPOUND[ch]);
    } else {
      out.push(ch);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------
// Key identity channels
// ---------------------------------------------------------------------------------------
export function resolveKey(key, layout) {
  if (!key) return null;
  if (key.startsWith('char:')) return layout.char_to_code[key.slice(5)] ?? null;
  return key;
}

function geometry(code, layout) {
  return (code && layout.keys[code]) || layout.unknown;
}

function handOf(finger) {
  if (finger >= 0 && finger <= 3) return 0;
  if (finger >= 6 && finger <= 9) return 1;
  return null;
}

export function keyPairFeatures(codeA, codeB, layout) {
  const [ra, ca, fa] = geometry(codeA, layout);
  const [rb, cb, fb] = geometry(codeB, layout);
  const dist = Math.hypot(ra - rb, ca - cb);
  const ha = handOf(fa);
  const hb = handOf(fb);
  const sameHand = ha === null || hb === null ? 0.5 : ha === hb ? 1.0 : 0.0;
  const sameFinger = fa >= 0 && fa === fb && codeA !== codeB ? 1.0 : 0.0;
  return [ra / layout.row_scale, ca / layout.col_scale, rb / layout.row_scale, cb / layout.col_scale, dist, sameHand, sameFinger];
}

/**
 * Stroke -> scaled [C, M] tensor (9 timing + 7 key identity + Hold_Valid).
 * Unknown holds (touch keyboards) are imputed with the Train median and flagged.
 */
export function buildInputTensor(stroke, config) {
  const { holds, dds, uds, stitchFlags = [], pairKeys = [], holdValid = [] } = stroke;
  const m = dds.length;
  const bounds = config.outlier_bounds;
  const layout = config.keyboard_layout;
  const eps = 1e-4;
  const clip = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
  const hImp = clip(config.touch_hold_impute_sec ?? 0.09, bounds.H);

  const cHolds = holds.map((v) => clip(v === null || v === undefined ? hImp : v, bounds.H));
  const cDds = dds.map((v) => clip(v, bounds.DD));
  const cUds = uds.map((v, t) => clip(v === null || v === undefined ? cDds[t] - hImp : v, bounds.UD));

  const meanH = Math.max(eps, cHolds.reduce((a, b) => a + b, 0) / cHolds.length);
  const meanDD = Math.max(eps, cDds.reduce((a, b) => a + b, 0) / cDds.length);

  const nCh = config.in_channels;
  const data = new Float32Array(nCh * m);
  const chMean = config.channel_mean;
  const chStd = config.channel_std;

  for (let t = 0; t < m; t++) {
    const hCurr = cHolds[t];
    const hNext = cHolds[t + 1];
    const dd = cDds[t];
    const ud = cUds[t];
    const localAccel = t === 0 || stitchFlags[t] ? 0.0 : dd - cDds[t - 1];
    const [ka, kb] = pairKeys[t] || [null, null];
    const valid = holdValid.length ? holdValid[t] && holdValid[t + 1] : true;
    const raw = [
      hCurr,
      hNext,
      dd,
      ud,
      hCurr / meanH,
      dd / meanDD,
      Math.min(5.0, Math.max(0.0, hCurr / Math.max(dd, 0.02))),
      Math.max(0.0, -ud),
      localAccel,
      ...keyPairFeatures(resolveKey(ka, layout), resolveKey(kb, layout), layout),
      valid ? 1.0 : 0.0,
    ];
    for (let c = 0; c < nCh; c++) {
      data[c * m + t] = (raw[c] - chMean[c]) / chStd[c];
    }
  }
  return { tensorData: data, dims: [1, nCh, m], cleanStroke: { holds: cHolds, dds: cDds, uds: cUds } };
}

// ---------------------------------------------------------------------------------------
// Behavioural profile
// ---------------------------------------------------------------------------------------
export const BEHAVIOR_FEATURES = [
  'bs_rate', 'bs_burst_len', 'bs_single_frac', 'bs_wipe_frac', 'bs_detect_lat', 'bs_resume_lat',
  'bs_repeat_ivl', 'bs_hold', 'pre_err_var', 'caps_ratio', 'shift_lead', 'shift_hold',
  'shift_right_ratio', 'pause_500', 'pause_1000', 'space_after_rel', 'space_before_rel',
];

const PUNCT_CODES = new Set([
  'Period', 'Comma', 'Slash', 'Semicolon', 'Quote', 'BracketLeft', 'BracketRight',
  'Minus', 'Equal', 'Backquote', 'Backslash',
]);

/** Canonical behaviour kind for a KeyboardEvent.code (or a produced character). */
export function kindOf(label) {
  if (label === 'Backspace' || label === 'Delete') return { kind: 'bs', side: null };
  if (label === 'ShiftLeft' || label === 'ShiftRight' || label === 'Shift') {
    return { kind: 'shift', side: label === 'ShiftLeft' ? 'L' : label === 'ShiftRight' ? 'R' : null };
  }
  if (label === 'CapsLock') return { kind: 'caps', side: null };
  if (label === 'Space' || label === ' ') return { kind: 'space', side: null };
  if ([...String(label)].length === 1 || label.startsWith('Key') || label.startsWith('Digit') || PUNCT_CODES.has(label)) {
    return { kind: 'char', side: null };
  }
  return { kind: 'other', side: null };
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

function std(arr) {
  const mu = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mu) * (b - mu), 0) / arr.length);
}

const isTyping = (e) => e.kind === 'char' || e.kind === 'space' || e.kind === 'bs';
const isTextual = (e) => e.kind === 'char' || e.kind === 'space';

/** Per-event values for every behavioural feature. Events: {kind, t (s), hold (s|null), side, rep}. */
export function behaviorEventValues(events) {
  const ev = [...events].sort((a, b) => a.t - b.t);
  const vals = Object.fromEntries(BEHAVIOR_FEATURES.map((n) => [n, []]));
  const presses = ev.filter(isTyping);

  const cc = [];
  for (let i = 0; i + 1 < presses.length; i++) {
    const a = presses[i];
    const b = presses[i + 1];
    if (isTextual(a) && isTextual(b)) {
      const d = b.t - a.t;
      if (d > 0.008 && d < 2.0) cc.push(d);
    }
  }
  const tempo = cc.length >= 3 ? median(cc) : null;
  const baseStd = cc.length >= 4 ? std(cc.map(Math.log)) : 0.0;

  for (const p of presses) vals.bs_rate.push(p.kind === 'bs' ? 1.0 : 0.0);

  const n = presses.length;
  let i = 0;
  while (i < n) {
    if (presses[i].kind !== 'bs') {
      i += 1;
      continue;
    }
    let j = i;
    while (j + 1 < n && presses[j + 1].kind === 'bs') j += 1;
    const length = j - i + 1;
    vals.bs_burst_len.push(Math.log(length));
    vals.bs_single_frac.push(length === 1 ? 1.0 : 0.0);
    vals.bs_wipe_frac.push(length >= 4 ? 1.0 : 0.0);
    if (tempo !== null && i > 0 && isTextual(presses[i - 1])) {
      const d = presses[i].t - presses[i - 1].t;
      if (d > 0.0 && d < 5.0) vals.bs_detect_lat.push(Math.log(d / tempo));
    }
    if (tempo !== null && j + 1 < n && isTextual(presses[j + 1])) {
      const d = presses[j + 1].t - presses[j].t;
      if (d > 0.0 && d < 5.0) vals.bs_resume_lat.push(Math.log(d / tempo));
    }
    for (let q = i; q < j; q++) {
      const a = presses[q];
      const b = presses[q + 1];
      if (!a.rep && !b.rep) {
        const d = b.t - a.t;
        if (d > 0.02 && d < 2.0) vals.bs_repeat_ivl.push(Math.log(d));
      }
    }
    if (baseStd > 1e-6 && i >= 4) {
      let allText = true;
      for (let k = i - 4; k < i; k++) allText = allText && isTextual(presses[k]);
      if (allText) {
        const ivl = [];
        for (let k = i - 4; k < i - 1; k++) ivl.push(presses[k + 1].t - presses[k].t);
        if (ivl.every((v) => v > 0.008)) vals.pre_err_var.push(std(ivl.map(Math.log)) / baseStd);
      }
    }
    i = j + 1;
  }

  for (const e of ev) {
    if (e.kind === 'bs' && !e.rep && e.hold !== null && e.hold !== undefined && e.hold > 0.008 && e.hold < 1.5) {
      vals.bs_hold.push(Math.log(e.hold));
    }
  }

  for (let idx = 0; idx < ev.length; idx++) {
    const e = ev[idx];
    if (e.rep || (e.kind !== 'shift' && e.kind !== 'caps')) continue;
    vals.caps_ratio.push(e.kind === 'caps' ? 1.0 : 0.0);
    if (e.kind !== 'shift') continue;
    if (e.hold !== null && e.hold !== undefined && e.hold > 0.008 && e.hold < 2.0) vals.shift_hold.push(Math.log(e.hold));
    if (e.side) vals.shift_right_ratio.push(e.side === 'R' ? 1.0 : 0.0);
    let nxt = null;
    for (let k = idx + 1; k < ev.length; k++) {
      if (isTyping(ev[k])) {
        nxt = ev[k];
        break;
      }
    }
    if (nxt && nxt.kind === 'char') {
      const d = nxt.t - e.t;
      if (d > 0.0 && d < 2.0) vals.shift_lead.push(Math.log(d));
    }
  }

  for (let k = 0; k + 1 < presses.length; k++) {
    const a = presses[k];
    const b = presses[k + 1];
    const d = b.t - a.t;
    if (d > 0) {
      vals.pause_500.push(d > 0.5 ? 1.0 : 0.0);
      vals.pause_1000.push(d > 1.0 ? 1.0 : 0.0);
    }
    if (tempo !== null && d > 0.008 && d < 2.0) {
      if (a.kind === 'char' && b.kind === 'space') vals.space_after_rel.push(Math.log(d / tempo));
      else if (a.kind === 'space' && b.kind === 'char') vals.space_before_rel.push(Math.log(d / tempo));
    }
  }
  return vals;
}

/** Profile statistics {feature: [sum, count]}. */
export function summarizeBehavior(events) {
  const vals = behaviorEventValues(events);
  const out = {};
  for (const k of BEHAVIOR_FEATURES) {
    out[k] = [vals[k].reduce((a, b) => a + b, 0), vals[k].length];
  }
  return out;
}

export function mergeBehavior(a, b) {
  if (!b) return a ? { ...a } : null;
  if (!a) return Object.fromEntries(Object.entries(b).map(([k, v]) => [k, [...v]]));
  const out = {};
  for (const k of BEHAVIOR_FEATURES) {
    const [sa, na] = a[k] || [0, 0];
    const [sb, nb] = b[k] || [0, 0];
    out[k] = [sa + sb, na + nb];
  }
  return out;
}

/** Two-covariance log-likelihood ratio, summed over scored features. */
export function behaviorLlr(query, enroll, behaviorConfig) {
  const parts = {};
  let total = 0;
  if (!query || !enroll || !behaviorConfig) return { total, parts };
  const clipV = behaviorConfig.llr_clip ?? 4.0;
  for (const f of behaviorConfig.features) {
    if (!f.scored || f.sigma2 === undefined) continue;
    const [sq, nq] = query[f.name] || [0, 0];
    const [se, ne] = enroll[f.name] || [0, 0];
    if (nq < 1 || ne < 1) continue;
    const v0 = f.sigma2 * (1.0 / nq + 1.0 / ne) + 2.0 * f.delta2;
    const v1 = v0 + 2.0 * f.tau2;
    const d = sq / nq - se / ne;
    let val = 0.5 * Math.log(v1 / v0) - 0.5 * d * d * (1.0 / v0 - 1.0 / v1);
    val = Math.max(-clipV, Math.min(clipV, val));
    parts[f.name] = val;
    total += val;
  }
  return { total, parts };
}

export function fuseScore(similarity, llr, beta) {
  const s = Math.min(1 - 1e-6, Math.max(1e-6, similarity));
  const logit = Math.log(s / (1 - s)) + beta * llr;
  return 1 / (1 + Math.exp(-logit));
}

function dotVec(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Template similarity shared by every score: 0.65·cos(centroid) + 0.35·max cos(exemplar). */
export function rawTemplateSimilarity(embedding, template) {
  const c = dotVec(embedding, template.centroid);
  let ex = c;
  if (Array.isArray(template.exemplars) && template.exemplars.length > 0) {
    ex = -1;
    for (const e of template.exemplars) ex = Math.max(ex, dotVec(embedding, e));
  }
  return { raw: 0.65 * c + 0.35 * ex, centroidSim: c, exemplarSim: ex };
}

/**
 * Password score for user-chosen strings (README §5.6): both encoders' template similarities are
 * standardised by impostor statistics measured on many strings typed by real people and summed,
 *   z = (raw9 − μ9)/σ9 + (raw17 − μ17)/σ17,
 * then shown as similarity = sigmoid(slope·(z − τ)), so 0.5 is the calibrated decision point.
 */
export function dualPasswordScore(emb9, emb17, tmpl9, tmpl17, prm, slope = 1.0) {
  const r9 = rawTemplateSimilarity(emb9, tmpl9);
  const r17 = rawTemplateSimilarity(emb17, tmpl17);
  const z = (r9.raw - prm.mu9) / prm.sd9 + (r17.raw - prm.mu17) / prm.sd17;
  return { z, raw9: r9.raw, raw17: r17.raw, centroidSim: r9.centroidSim, similarity: 1 / (1 + Math.exp(-slope * (z - prm.tau))) };
}

/** Several captures typed in a row scored as one query: mean embedding, re-normalised. */
export function combineEmbeddings(list) {
  const dim = list[0].length;
  const out = new Array(dim).fill(0);
  for (const e of list) for (let d = 0; d < dim; d++) out[d] += e[d];
  const n = Math.hypot(...out) || 1e-8;
  return out.map((v) => v / n);
}

/** Open-set rule: the top candidate must clear the threshold AND lead the runner-up by `margin`. */
export function openSetAccept(scoresDesc, threshold, margin = 0) {
  if (!scoresDesc.length) return false;
  const second = scoresDesc.length > 1 ? scoresDesc[1] : -Infinity;
  return scoresDesc[0] >= threshold && scoresDesc[0] - second >= margin;
}

/** Human-readable summary of a behaviour profile for the UI. */
export function describeBehavior(stats) {
  if (!stats) return null;
  const mean = (k) => {
    const [s, n] = stats[k] || [0, 0];
    return n > 0 ? { value: s / n, n } : null;
  };
  const bs = mean('bs_rate');
  const caps = mean('caps_ratio');
  const right = mean('shift_right_ratio');
  const single = mean('bs_single_frac');
  const pause = mean('pause_500');
  return {
    backspacePct: bs ? Math.round(bs.value * 1000) / 10 : null,
    corrections: stats.bs_burst_len ? stats.bs_burst_len[1] : 0,
    singleFixPct: single ? Math.round(single.value * 100) : null,
    capsStyle: caps ? (caps.value >= 0.5 ? 'CapsLock' : 'Shift') : null,
    rightShiftPct: right ? Math.round(right.value * 100) : null,
    shiftCount: right ? right.n : 0,
    pausePct: pause ? Math.round(pause.value * 100) : null,
  };
}
