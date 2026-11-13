import * as ort from 'onnxruntime-web';
import {
  attemptCapitalMethod,
  behaviorLlr,
  buildInputTensor,
  capitalMethodLlr,
  dualParamsForLength,
  dualPasswordScore,
  fuseScore,
  mergeBehavior,
  openSetAccept,
} from './features.js';

export { combineEmbeddings } from './features.js';

export { SmartKeystrokeRecorder } from './recorder.js';
export { describeBehavior } from './features.js';
export {
  composingTailIndex,
  computeLiveAccuracy,
  computeTypingAccuracy,
  validatePasswordCapture,
  validateSentenceCapture,
} from './validation.js';

// v4 profile & compatibility identifiers.
// IMPORTANT: `modelHash` and `preprocessVersion` are strictly embedding-space compatibility
// guards (preventing vectors from incompatible ONNX weights or preprocessing formulas from
// mixing in localStorage / BroadcastChannel / MQTT). They are NOT cryptographic ownership
// or user authentication credentials.
export const SCHEMA_VERSION = 4;
export const DEFAULT_PREPROCESS_VERSION = 'kp-preproc-v4';

// Single-threaded SIMD WASM avoids SharedArrayBuffer COOP/COEP issues on GitHub Pages
if (typeof window !== 'undefined' && window.location?.href) {
  ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
}
ort.env.wasm.numThreads = 1;

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Pure-JS FIPS 180-4 SHA-256 fallback when `crypto.subtle` is unavailable (e.g. plain HTTP LAN). */
export function sha256HexPureJs(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const len = bytes.length;
  const bitLen = len * 8;
  const totalLen = ((len + 9 + 63) >> 6) << 6;
  const padded = new Uint8Array(totalLen);
  padded.set(bytes);
  padded[len] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(totalLen - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(totalLen - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;
  const w = new Uint32Array(64);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));

  for (let offset = 0; offset < totalLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7].map((v) => v.toString(16).padStart(8, '0')).join('');
}

export async function sha256Hex(input) {
  const bytes =
    typeof input === 'string'
      ? new TextEncoder().encode(input)
      : input instanceof Uint8Array
        ? input
        : new Uint8Array(input);
  if (globalThis.crypto?.subtle?.digest) {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
  return sha256HexPureJs(bytes);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Merge a newly embedded sample into a variable-length template (works for both
 * fixed passwords and diverse multi-sentence typing practice!). Behavioural statistics are
 * pooled as (sum, count) so the evidence behind every feature keeps accumulating.
 */
function appendExemplar(prev, embedding, keep = 12) {
  const prevExemplars = prev?.exemplars ? [...prev.exemplars] : [];
  prevExemplars.push(embedding.map((v) => Math.round(v * 100000) / 100000));
  // Keep up to 12 most recent exemplars so synced payload stays tiny (~2 KB)
  const exemplars = prevExemplars.slice(-keep);
  const dim = embedding.length;
  const centroid = new Array(dim).fill(0);
  for (const ex of exemplars) {
    for (let d = 0; d < dim; d++) centroid[d] += ex[d];
  }
  const cNorm = Math.hypot(...centroid) || 1e-8;
  for (let d = 0; d < dim; d++) {
    centroid[d] = Math.round((centroid[d] / cNorm) * 100000) / 100000;
  }
  return { centroid, exemplars };
}

/**
 * `embedding17` (passwords only) keeps a parallel template in the 17-channel encoder space, so a
 * password attempt can be scored by both encoders (dualPasswordScore).
 */
export function mergeTemplateSample(
  existing,
  { embedding, embedding17, cpm, holdMs, rolloverPct, behavior, inputMode, capitalMethods = null }
) {
  const { centroid, exemplars } = appendExemplar(existing, embedding);
  const enc17 = embedding17 ? appendExemplar(existing?.enc17, embedding17) : existing?.enc17 || null;
  // Attempts whose capitals were typed with CapsLock vs Shift: [caps, shift]
  const capsUse = [...(existing?.capsUse || [0, 0])];
  const method = attemptCapitalMethod(capitalMethods);
  if (method === 'caps') capsUse[0] += 1;
  else if (method === 'shift') capsUse[1] += 1;

  const prevCount = existing?.count || 0;
  const nextCount = prevCount + 1;
  const blend = (oldVal, newVal) =>
    prevCount === 0 ? newVal : Math.round((oldVal * prevCount + newVal) / nextCount);

  const modeCounts = { keyboard: 0, touch: 0, mixed: 0, ...(existing?.inputModeCounts || {}) };
  if (!existing?.inputModeCounts && existing?.inputMode) modeCounts[existing.inputMode] += prevCount;
  modeCounts[inputMode || 'keyboard'] = (modeCounts[inputMode || 'keyboard'] || 0) + 1;
  const dominant = Object.entries(modeCounts).sort((a, b) => b[1] - a[1])[0][0];
  const total = modeCounts.keyboard + modeCounts.touch + modeCounts.mixed;
  const templateMode = modeCounts[dominant] === total ? dominant : 'mixed';

  const mergedBehavior = mergeBehavior(existing?.behavior || null, behavior || null);
  return {
    ...(existing?.target ? { target: existing.target } : {}),
    ...(enc17 ? { enc17 } : {}),
    ...(capsUse[0] + capsUse[1] > 0 ? { capsUse } : {}),
    centroid,
    exemplars,
    count: nextCount,
    cpm: blend(existing?.cpm || 0, cpm || 0),
    holdMs: blend(existing?.holdMs || 0, holdMs || 0),
    rolloverPct: blend(existing?.rolloverPct || 0, rolloverPct || 0),
    behavior: mergedBehavior
      ? Object.fromEntries(Object.entries(mergedBehavior).map(([k, [sv, n]]) => [k, [Math.round(sv * 1e6) / 1e6, n]]))
      : null,
    inputMode: templateMode,
    inputModeCounts: modeCounts,
  };
}

export class KeyPrintEngine {
  constructor() {
    this.session = null; // 17-channel encoder: typing-practice sentences (key identity + touch-trained)
    this.passwordSession = null; // timing-only encoder: user-chosen passwords (content-agnostic)
    this.config = null;
    this.loadTimeMs = 0;
    this.schemaVersion = SCHEMA_VERSION;
    this.preprocessVersion = DEFAULT_PREPROCESS_VERSION;
    this.modelHash = null;
    this.computedModelHash = null;
  }

  async init(modelUrl = './models/keyprint_net.onnx', configUrl = './models/model_config.json') {
    const t0 = performance.now();
    const sessionOpts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const baseHref =
      typeof window !== 'undefined' && window.location?.href ? window.location.href : 'http://localhost/';

    const [cfgResp, buf17] = await Promise.all([
      fetch(configUrl).then((r) => {
        if (!r.ok) throw new Error(`설정 로드 실패: HTTP ${r.status}`);
        return r.json();
      }),
      fetch(modelUrl).then((r) => {
        if (!r.ok) throw new Error(`모델 로드 실패: HTTP ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    this.config = cfgResp;

    const pm = this.config.password_model;
    let buf9 = null;
    if (pm?.file) {
      const pwUrl = new URL(pm.file, new URL(configUrl, baseHref)).href;
      buf9 = await fetch(pwUrl).then((r) => {
        if (!r.ok) throw new Error(`비밀번호 인코더 로드 실패: HTTP ${r.status}`);
        return r.arrayBuffer();
      });
    }

    const bytes17 = new Uint8Array(buf17);
    const bytes9 = buf9 ? new Uint8Array(buf9) : null;
    const sha17 = await sha256Hex(bytes17);
    const sha9 = bytes9 ? await sha256Hex(bytes9) : '';
    const computedHash = (await sha256Hex(`${sha17}:${sha9}`)).slice(0, 16);

    // Integrity guard against stale/corrupt cached ONNX weights vs model_config.json.
    // Note: modelHash & preprocessVersion are embedding-space compatibility guards, NOT ownership auth.
    if (this.config.model_hash && this.config.model_hash !== computedHash) {
      throw new Error(
        `모델 가중치 해시 불일치 (설정 ${this.config.model_hash} ≠ 파일 ${computedHash})`
      );
    }

    this.schemaVersion = SCHEMA_VERSION;
    this.preprocessVersion = this.config.preprocess_version || DEFAULT_PREPROCESS_VERSION;
    this.computedModelHash = computedHash;
    this.modelHash = this.config.model_hash || computedHash;

    this.session = await ort.InferenceSession.create(bytes17, sessionOpts);
    if (bytes9) {
      this.passwordSession = await ort.InferenceSession.create(bytes9, sessionOpts);
    }

    const nCh = this.config.in_channels || 9;
    await this.session.run({ keystroke_seq: new ort.Tensor('float32', new Float32Array(nCh * 10), [1, nCh, 10]) });
    if (this.passwordSession) {
      const pCh = pm.in_channels;
      await this.passwordSession.run({ keystroke_seq: new ort.Tensor('float32', new Float32Array(pCh * 10), [1, pCh, 10]) });
    }
    this.loadTimeMs = Math.round((performance.now() - t0) * 10) / 10;
    return this.config;
  }

  async _run(session, tensorData, nCh, m) {
    const inputTensor = new ort.Tensor('float32', tensorData, [1, nCh, m]);
    const results = await session.run({ keystroke_seq: inputTensor });
    const rawEmb = Array.from(results.biometric_embedding.data);
    const norm = Math.hypot(...rawEmb) || 1e-8;
    return rawEmb.map((v) => v / norm);
  }

  /**
   * 64-D embedding of a clean stroke. `domain: 'password'` uses the timing-only password encoder
   * (its inputs are channels 0..8 of the same tensor, laid out channel-major, so a prefix slice).
   */
  async embedStroke(stroke, { domain = 'sentence' } = {}) {
    const t0 = performance.now();
    const { tensorData, dims, cleanStroke } = buildInputTensor(stroke, this.config);
    const [, nCh, m] = dims;
    if (m === 0) {
      throw new Error('추출된 연속 리듬 구간이 없습니다.');
    }
    let embedding;
    if ((domain === 'password' || domain === 'guard') && this.passwordSession) {
      const pCh = this.config.password_model.in_channels;
      embedding = await this._run(this.passwordSession, tensorData.slice(0, pCh * m), pCh, m);
    } else {
      embedding = await this._run(this.session, tensorData, nCh, m);
    }
    const latencyMs = Math.round((performance.now() - t0) * 100) / 100;
    return { embedding, latencyMs, cleanStroke };
  }

  /** Both domain embeddings (the principles map can switch between sentence and password space). */
  async embedAll(stroke) {
    const sentence = await this.embedStroke(stroke, { domain: 'sentence' });
    const password = this.passwordSession ? await this.embedStroke(stroke, { domain: 'password' }) : sentence;
    return {
      sentence: sentence.embedding,
      password: password.embedding,
      latencyMs: Math.round((sentence.latencyMs + (this.passwordSession ? password.latencyMs : 0)) * 100) / 100,
    };
  }

  /** Validation-calibrated decision thresholds for the current input device. */
  passwordThreshold(inputMode = 'keyboard') {
    const base = this.config?.calibrated_threshold ?? 0.49;
    if (inputMode === 'touch') return this.config?.calibrated_threshold_touch ?? base;
    return base;
  }

  arenaThreshold(inputMode = 'keyboard') {
    const base = this.config?.arena_open_set_threshold ?? 0.42;
    if (inputMode === 'touch') return this.config?.arena_open_set_threshold_touch ?? base;
    return base;
  }

  /** Extra practice on an existing profile: one sentence must verify 1:1 above this (README §5.6). */
  arenaAddThreshold(inputMode = 'keyboard') {
    const base = this.config?.arena_add_threshold ?? this.arenaThreshold('keyboard');
    if (inputMode === 'touch') return this.config?.arena_add_threshold_touch ?? base;
    return base;
  }

  /** Typing practice: the winner must also lead the runner-up by this margin (README §5.6). */
  arenaMargin(inputMode = 'keyboard') {
    const base = this.config?.arena_margin ?? 0;
    if (inputMode === 'touch') return this.config?.arena_margin_touch ?? base;
    return base;
  }

  /** Sentences per profile / per identification (deployed protocol from the pipeline). */
  arenaProtocol() {
    return {
      enroll: this.config?.arena_enroll_sentences ?? 3,
      query: this.config?.arena_query_sentences ?? 1,
    };
  }

  /** True when the dual-encoder password score is available for this template. */
  usesDualPassword(template) {
    return Boolean(this.config?.password_dual && template?.enc17);
  }

  /** Mobile (touch) password verdicts are shown as reference only (README §5.6). */
  passwordTouchReferenceOnly() {
    return Boolean(this.config?.password_dual?.touch_reference_only);
  }

  behaviorWeight(domain, inputMode = 'keyboard') {
    const beta = this.config?.behavior_profile?.beta;
    if (typeof beta === 'number') return beta;
    if (!beta) return 0;
    if (domain === 'password') return beta.password ?? 0;
    return inputMode === 'touch' ? beta.sentence_touch ?? 0 : beta.sentence_keyboard ?? 0;
  }

  scoreTemplate(
    embedding,
    template,
    {
      domain = 'password',
      threshold = null,
      thresholdOffset = 0,
      behavior = null,
      inputMode = 'keyboard',
      embedding17 = null,
      capitalMethods = null,
    } = {}
  ) {
    const dual = this.config?.password_dual;
    if (domain === 'password' && dual && embedding17 && template?.enc17) {
      // User-chosen passwords: both encoders, impostor-standardised on many real-typist strings,
      // with parameters for this password's length (README §5.6–5.7)
      const prm = { ...dualParamsForLength(dual, inputMode, template.target?.length) };
      const base = dualPasswordScore(embedding, embedding17, template, template.enc17, prm, dual.slope ?? 1.0);
      let z = base.z;
      let capitalLlr = 0;
      const cm = dual.capital_method;
      const qMethod = attemptCapitalMethod(capitalMethods);
      const used = template.capsUse ? template.capsUse[0] + template.capsUse[1] : 0;
      if (cm && inputMode === 'keyboard' && qMethod && used > 0) {
        // Shift vs CapsLock habit for capitals (keyboard only: phones hide modifier keys)
        capitalLlr = capitalMethodLlr(qMethod === 'caps', template.capsUse[0], used, cm.caps_share, cm.alpha ?? 2.0);
        z += cm.weight * capitalLlr;
        prm.tau += cm.tau_offset ?? 0;
      }
      const r = { ...base, z, similarity: 1 / (1 + Math.exp(-(dual.slope ?? 1.0) * (z - prm.tau))) };
      const thr = threshold ?? 0.5 + thresholdOffset;
      return {
        method: 'dual',
        capitalLlr,
        similarity: r.similarity,
        embeddingSimilarity: r.similarity,
        behaviorLlr: 0,
        behaviorWeight: 0,
        behaviorParts: {},
        centroidSim: r.centroidSim,
        exemplarSim: null,
        z: r.z,
        accepted: r.similarity >= thr,
        threshold: thr,
        // template adaptation only on a confident accept (guards against an impostor drifting in)
        adapt: r.similarity >= thr && r.z >= prm.tau + (dual.adapt_margin ?? 1.0),
      };
    }

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

    const touch = inputMode === 'touch';
    const cfg = this.config || {};
    const cohort =
      domain === 'sentence'
        ? (touch && cfg.sentence_cohort_centroids_touch) || cfg.sentence_cohort_centroids || cfg.cohort_centroids
        : (touch && cfg.cohort_centroids_touch) || cfg.cohort_centroids;

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

    const embeddingSimilarity = similarity;
    let behaviorLlrTotal = 0;
    let behaviorParts = {};
    const beta = this.behaviorWeight(domain, inputMode);
    if (beta > 0 && behavior && template.behavior) {
      const res = behaviorLlr(behavior, template.behavior, cfg.behavior_profile);
      behaviorLlrTotal = res.total;
      behaviorParts = res.parts;
      similarity = fuseScore(similarity, res.total, beta);
    }

    const thr =
      threshold ?? (domain === 'password' ? this.passwordThreshold(inputMode) : this.arenaThreshold(inputMode)) + thresholdOffset;
    return {
      method: 'as_norm',
      adapt: false,
      similarity,
      embeddingSimilarity,
      behaviorLlr: behaviorLlrTotal,
      behaviorWeight: beta,
      behaviorParts,
      centroidSim: cSim,
      exemplarSim: maxExSim,
      accepted: similarity >= thr,
      threshold: thr,
    };
  }

  identifyTypist(
    embedding,
    candidates,
    { openSetThreshold = null, margin = null, behavior = null, inputMode = 'keyboard', domain = 'sentence' } = {}
  ) {
    if (!Array.isArray(candidates) || candidates.length === 0) return null;
    if (openSetThreshold === null) {
      openSetThreshold = domain === 'sentence' ? this.arenaThreshold(inputMode) : this.passwordThreshold(inputMode);
    }
    if (margin === null) margin = domain === 'sentence' ? this.arenaMargin(inputMode) : 0;

    const scored = candidates.map((cand) => {
      const res = this.scoreTemplate(embedding, cand.template, {
        domain,
        threshold: openSetThreshold,
        behavior,
        inputMode,
      });
      return {
        id: cand.id,
        name: cand.name,
        color: cand.color || '#2563eb',
        isSample: Boolean(cand.isSample),
        similarity: res.similarity,
        embeddingSimilarity: res.embeddingSimilarity,
        behaviorLlr: res.behaviorLlr,
        behaviorWeight: res.behaviorWeight,
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
    const identified = openSetAccept(
      scored.map((it) => it.similarity),
      openSetThreshold,
      margin
    );
    const lead = scored.length > 1 ? winner.similarity - scored[1].similarity : Infinity;

    return {
      identified,
      winner,
      openSetThreshold,
      margin,
      lead,
      rejectedByMargin: !identified && winner.similarity >= openSetThreshold,
      rankings: scored,
    };
  }
}
