import * as ort from 'onnxruntime-web';
import {
  behaviorLlr,
  buildInputTensor,
  dualPasswordScore,
  fuseScore,
  mergeBehavior,
  openSetAccept,
} from './features.js';

export { combineEmbeddings } from './features.js';

export { SmartKeystrokeRecorder } from './recorder.js';
export { describeBehavior } from './features.js';
export { computeTypingAccuracy, validatePasswordCapture, validateSentenceCapture } from './validation.js';

// Single-threaded SIMD WASM avoids SharedArrayBuffer COOP/COEP issues on GitHub Pages
ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
ort.env.wasm.numThreads = 1;

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
export function mergeTemplateSample(existing, { embedding, embedding17, cpm, holdMs, rolloverPct, behavior, inputMode }) {
  const { centroid, exemplars } = appendExemplar(existing, embedding);
  const enc17 = embedding17 ? appendExemplar(existing?.enc17, embedding17) : existing?.enc17 || null;

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
  }

  async init(modelUrl = './models/keyprint_net.onnx', configUrl = './models/model_config.json') {
    const t0 = performance.now();
    const sessionOpts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const [cfgResp, session] = await Promise.all([
      fetch(configUrl).then((r) => {
        if (!r.ok) throw new Error(`설정 로드 실패: HTTP ${r.status}`);
        return r.json();
      }),
      ort.InferenceSession.create(modelUrl, sessionOpts),
    ]);
    this.config = cfgResp;
    this.session = session;

    const pm = this.config.password_model;
    if (pm?.file) {
      const pwUrl = new URL(pm.file, new URL(configUrl, window.location.href)).href;
      this.passwordSession = await ort.InferenceSession.create(pwUrl, sessionOpts);
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
    { domain = 'password', threshold = null, thresholdOffset = 0, behavior = null, inputMode = 'keyboard', embedding17 = null } = {}
  ) {
    const dual = this.config?.password_dual;
    if (domain === 'password' && dual && embedding17 && template?.enc17) {
      // User-chosen passwords: both encoders, impostor-standardised on many real-typist strings
      const prm = inputMode === 'touch' ? dual.touch : dual.keyboard;
      const r = dualPasswordScore(embedding, embedding17, template, template.enc17, prm, dual.slope ?? 1.0);
      const thr = threshold ?? 0.5 + thresholdOffset;
      return {
        method: 'dual',
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
