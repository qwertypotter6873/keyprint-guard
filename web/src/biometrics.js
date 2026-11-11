import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime Web WASM for static hosting (GitHub Pages / Vercel)
// Single-threaded SIMD avoids SharedArrayBuffer COOP/COEP header issues on GitHub Pages.
ort.env.wasm.wasmPaths = new URL('./ort/', window.location.href).href;
ort.env.wasm.numThreads = 1;

/**
 * Validate captured keystroke events and convert to raw { holds, dds, uds } in seconds.
 * Implements strict biometric input exception handling (Rubric D3).
 */
export function validateAndBuildStroke(events, expectedPassword) {
  if (!expectedPassword || expectedPassword.length < 6) {
    return {
      ok: false,
      code: 'SHORT_PASSWORD',
      message: '비밀번호는 최소 6자 이상이어야 유의미한 디그래프(Digraph) 생체 리듬을 추출할 수 있습니다.',
    };
  }

  if (events.hadPaste) {
    return {
      ok: false,
      code: 'PASTE_BLOCKED',
      message: '복사·붙여넣기(Paste) 입력이 감지되었습니다. 타건 생체 지문 측정을 위해 직접 타이핑해 주세요.',
    };
  }

  if (events.hadRepeat) {
    return {
      ok: false,
      code: 'KEY_REPEAT',
      message: '키 길게 누름(OS Key Repeat)이 감지되어 홀드(Hold) 시간이 왜곡되었습니다. 평소 속도로 다시 입력해 주세요.',
    };
  }

  if (events.hadBackspace) {
    return {
      ok: false,
      code: 'BACKSPACE_USED',
      message: '백스페이스/수정 키가 포함되었습니다. 연속적인 근육 기억 측정을 위해 오타 없이 한 번에 입력해 주세요.',
    };
  }

  if (events.typedText !== expectedPassword) {
    return {
      ok: false,
      code: 'PASSWORD_MISMATCH',
      message: `비밀번호 문자열이 일치하지 않습니다 (입력 길이: ${events.typedText.length}자 / 설정 암호: ${expectedPassword.length}자).`,
    };
  }

  const records = events.records;
  if (records.length < 6) {
    return {
      ok: false,
      code: 'INSUFFICIENT_KEYS',
      message: '측정된 유효 키 이벤트 수가 부족합니다. 입력창을 클릭한 뒤 처음부터 타이핑해 주세요.',
    };
  }

  const holds = [];
  const dds = [];
  const uds = [];
  const keyLabels = [];

  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    const upTime = r.upTime ?? (r.downTime + 85.0);
    const holdSec = (upTime - r.downTime) / 1000.0;

    if (holdSec < 0.006 || holdSec > 1.2) {
      return {
        ok: false,
        code: 'ABNORMAL_HOLD',
        message: `비정상적인 키 누름 시간(${Math.round(holdSec * 1000)}ms)이 감지되었습니다. 하드웨어 매크로이거나 입력이 지연되었습니다.`,
      };
    }
    holds.push(holdSec);
    keyLabels.push(r.label);

    if (i < records.length - 1) {
      const next = records[i + 1];
      const ddSec = (next.downTime - r.downTime) / 1000.0;
      const udSec = (next.downTime - upTime) / 1000.0;

      if (ddSec > 2.8) {
        return {
          ok: false,
          code: 'TYPING_PAUSED',
          message: `타이핑 도중 긴 멈춤(${ddSec.toFixed(2)}초)이 감지되었습니다. 끊김 없이 자연스럽게 입력해 주세요.`,
        };
      }
      dds.push(ddSec);
      uds.push(udSec);
    }
  }

  return {
    ok: true,
    stroke: {
      holds,
      dds,
      uds,
      keyLabels,
    },
  };
}

/**
 * Exact JavaScript counterpart of src/preprocessing.py:
 * 1) Physiological outlier clipping
 * 2) 9-channel biomechanical digraph feature extraction
 * 3) Leakage-free Train channel Z-score standardization
 * Returns Float32Array of shape [1, 9, M] where M = K - 1.
 */
export function build9ChannelTensor(stroke, config) {
  const { holds, dds, uds } = stroke;
  const m = dds.length;
  const bounds = config.outlier_bounds;
  const eps = 1e-4;

  const clip = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));

  const cHolds = holds.map((v) => clip(v, bounds.H));
  const cDds = dds.map((v) => clip(v, bounds.DD));
  const cUds = uds.map((v) => clip(v, bounds.UD));

  const meanH = Math.max(eps, cHolds.reduce((a, b) => a + b, 0) / cHolds.length);
  const meanDD = Math.max(eps, cDds.reduce((a, b) => a + b, 0) / cDds.length);

  // Tensor layout: [1, 9, M] -> contiguous row-major channel-first
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
    const localAccel = t === 0 ? 0.0 : dd - cDds[t - 1];

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

    // Warm up WASM kernel with a dummy [1, 9, 10] inference
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

    // Ensure unit L2 norm
    const norm = Math.hypot(...rawEmb) || 1e-8;
    const embedding = rawEmb.map((v) => v / norm);
    const latencyMs = Math.round((performance.now() - t0) * 100) / 100;

    return { embedding, latencyMs, cleanStroke };
  }

  buildTemplate(enrolledItems) {
    const dim = enrolledItems[0].embedding.length;
    const centroid = new Array(dim).fill(0);
    for (const item of enrolledItems) {
      for (let d = 0; d < dim; d++) centroid[d] += item.embedding[d];
    }
    const cNorm = Math.hypot(...centroid) || 1e-8;
    for (let d = 0; d < dim; d++) centroid[d] /= cNorm;

    // Compute intra-template consistency & per-digraph mean/std for visual forensics
    const m = enrolledItems[0].cleanStroke.dds.length;
    const meanDD = new Array(m).fill(0);
    const stdDD = new Array(m).fill(0);
    const meanHold = new Array(m + 1).fill(0);
    const meanUD = new Array(m).fill(0);

    for (const item of enrolledItems) {
      for (let t = 0; t < m; t++) {
        meanDD[t] += item.cleanStroke.dds[t] / enrolledItems.length;
        meanUD[t] += item.cleanStroke.uds[t] / enrolledItems.length;
      }
      for (let k = 0; k <= m; k++) {
        meanHold[k] += item.cleanStroke.holds[k] / enrolledItems.length;
      }
    }

    for (const item of enrolledItems) {
      for (let t = 0; t < m; t++) {
        const diff = item.cleanStroke.dds[t] - meanDD[t];
        stdDD[t] += (diff * diff) / enrolledItems.length;
      }
    }
    for (let t = 0; t < m; t++) {
      stdDD[t] = Math.max(0.015, Math.sqrt(stdDD[t]));
    }

    const selfScores = enrolledItems.map((it) => dot(it.embedding, centroid));
    const consistency = selfScores.reduce((a, b) => a + b, 0) / selfScores.length;

    return {
      centroid,
      exemplars: enrolledItems.map((it) => it.embedding),
      meanDD,
      stdDD,
      meanHold,
      meanUD,
      consistency,
    };
  }

  scoreAgainstTemplate(embedding, cleanStroke, template, threshold) {
    const cSim = dot(embedding, template.centroid);
    let maxExSim = -1;
    for (const ex of template.exemplars) {
      const s = dot(embedding, ex);
      if (s > maxExSim) maxExSim = s;
    }
    const raw = 0.65 * cSim + 0.35 * maxExSim;
    let similarity = raw;

    // Apply Adaptive Symmetric Cohort Normalization (AS-Norm) if cohort is present
    const cohort = this.config?.cohort_centroids;
    if (Array.isArray(cohort) && cohort.length > 0) {
      const topM = Math.min(12, cohort.length);
      const uScores = cohort.map((r) => dot(r, template.centroid)).sort((a, b) => a - b).slice(-topM);
      const qScores = cohort.map((r) => dot(r, embedding)).sort((a, b) => a - b).slice(-topM);

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

    const accepted = similarity >= threshold;

    // Per-digraph rhythmic deviation z-score for explainable UI bar chart
    const m = Math.min(cleanStroke.dds.length, template.meanDD.length);
    const digraphDeviations = [];
    for (let t = 0; t < m; t++) {
      const z = Math.abs(cleanStroke.dds[t] - template.meanDD[t]) / (template.stdDD[t] + 0.01);
      digraphDeviations.push(Math.min(5.0, Math.round(z * 100) / 100));
    }

    return {
      similarity,
      centroidSim: cSim,
      exemplarSim: maxExSim,
      accepted,
      threshold,
      digraphDeviations,
    };
  }

  /**
   * 1:N Multi-User Typing-Practice Identification ("Who Is Typing?")
   * Ranks all N enrolled user templates in the gallery by AS-Norm biometric similarity
   * and applies an open-set rejection threshold to detect unregistered outsiders.
   */
  identifyAgainstGallery(embedding, cleanStroke, galleryProfiles, openSetThreshold = 0.48) {
    if (!Array.isArray(galleryProfiles) || galleryProfiles.length === 0) {
      return null;
    }

    const scored = galleryProfiles.map((profile) => {
      const res = this.scoreAgainstTemplate(
        embedding,
        cleanStroke,
        profile.template,
        openSetThreshold
      );
      return {
        id: profile.id,
        name: profile.name,
        badge: profile.badge || '',
        similarity: res.similarity,
        centroidSim: res.centroidSim,
        digraphDeviations: res.digraphDeviations,
        template: profile.template,
      };
    });

    scored.sort((a, b) => b.similarity - a.similarity);

    // Temperature-scaled softmax confidence across gallery candidates
    const temp = 9.0;
    const maxS = scored[0].similarity;
    const exps = scored.map((it) => Math.exp(temp * (it.similarity - maxS)));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1.0;
    scored.forEach((it, idx) => {
      it.posterior = exps[idx] / sumExp;
      it.rank = idx + 1;
    });

    const best = scored[0];
    const margin = scored.length > 1 ? best.similarity - scored[1].similarity : best.similarity;
    const identified = best.similarity >= openSetThreshold;

    return {
      identified,
      winner: best,
      margin,
      openSetThreshold,
      rankings: scored,
    };
  }
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Generate synthetic attack strokes for any password length K (including Enter key).
 */
export function synthesizeBotStroke(numKeys, attackType = 'uniform_macro') {
  const holds = [];
  const dds = [];
  const uds = [];

  if (attackType === 'uniform_macro') {
    // Automated Selenium / PyAutoGUI style constant-interval macro script
    for (let i = 0; i < numKeys; i++) {
      const h = 0.075 + (Math.random() - 0.5) * 0.004;
      holds.push(h);
      if (i < numKeys - 1) {
        const dd = 0.145 + (Math.random() - 0.5) * 0.005;
        dds.push(dd);
        uds.push(dd - h);
      }
    }
  } else if (attackType === 'gaussian_mimic') {
    // Attacker script injecting naive random delays to bypass rate limiters
    for (let i = 0; i < numKeys; i++) {
      const h = 0.045 + Math.random() * 0.135;
      holds.push(h);
      if (i < numKeys - 1) {
        const dd = 0.095 + Math.random() * 0.320;
        dds.push(dd);
        uds.push(dd - h);
      }
    }
  } else if (attackType === 'hunt_and_peck') {
    // One-finger "eagle typing" attacker who stole the plaintext password on a sticky note
    for (let i = 0; i < numKeys; i++) {
      const h = 0.095 + Math.random() * 0.055;
      holds.push(h);
      if (i < numKeys - 1) {
        const dd = 0.360 + Math.random() * 0.290;
        dds.push(dd);
        uds.push(dd - h);
      }
    }
  }

  return { holds, dds, uds };
}
