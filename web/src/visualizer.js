/**
 * Pure lightweight SVG visualizers & 64-D hypersphere geometry utilities
 * for the "원리 & 프로필 구조" (How It Works & Profile Formation) view.
 * Zero external charting libraries required.
 */

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

function normalizeVec(v) {
  const norm = Math.hypot(...v) || 1e-8;
  return v.map((x) => x / norm);
}

/**
 * Create a realistic pre-loaded keystroke capture ("딥러닝 생체지문") including
 * genuine 2-beolsik Korean consonant-vowel rollovers (UD < 0) and 1 Backspace stitch
 * so Step 1 is immediately rich and interactive even before the user types.
 */
export function createPresetCapture(mode = 'normal') {
  const scale = mode === 'slow' ? 1.62 : 1.0;
  // [label, code, rawHoldMs, rawDdFromPrevMs, stitchedBefore]
  const spec = [
    ['ㄷ', 'KeyE', 88, 0, false],
    ['ㅣ', 'KeyL', 76, 62, false], // UD = 62 - 88 = -26ms (Rollover!)
    ['ㅂ', 'KeyQ', 82, 114, false],
    ['ㄹ', 'KeyF', 91, 128, false],
    ['ㅓ', 'KeyJ', 79, 69, false], // UD = 69 - 91 = -22ms (Rollover!)
    ['ㄴ', 'KeyS', 86, 152, true], // Backspace stitch boundary (typo 'ㅜ' popped before 'ㄴ')
    ['ㅣ', 'KeyL', 74, 58, false], // UD = 58 - 86 = -28ms (Rollover!)
    ['ㅇ', 'KeyD', 84, 108, false],
    ['␣', 'Space', 95, 134, false],
    ['ㅅ', 'KeyT', 83, 118, false],
    ['ㅐ', 'KeyO', 77, 64, false], // UD = 64 - 83 = -19ms (Rollover!)
    ['ㅇ', 'KeyD', 85, 112, false],
    ['ㅊ', 'KeyC', 89, 124, false],
    ['ㅔ', 'KeyP', 78, 65, false], // UD = 65 - 89 = -24ms (Rollover!)
  ];

  const jitter = (i) => (mode === 'slow' ? ((i % 3) - 1) * 3 : 0);
  const events = [];
  const holds = [];
  const holdValid = [];
  const dds = [];
  const uds = [];
  const stitchFlags = [];
  const pairKeys = [];

  let cursorMs = 0;
  for (let i = 0; i < spec.length; i++) {
    const [label, code, baseHold, baseDd, stitched] = spec[i];
    const holdMs = Math.round(baseHold * scale + jitter(i));
    const ddMs = i === 0 ? null : Math.round((stitched ? 155 : baseDd * scale) + jitter(i + 1));
    if (i > 0) cursorMs += ddMs;
    const startMs = cursorMs;
    const endMs = startMs + holdMs;
    const prev = i > 0 ? events[i - 1] : null;
    const udMs = prev && !stitched ? startMs - prev.endMs : null;

    events.push({
      idx: i,
      code,
      label,
      startMs,
      endMs,
      holdMs,
      ddMs: stitched ? null : ddMs,
      udMs,
      isRollover: udMs !== null && udMs < 0,
      stitchedBefore: stitched,
      editedBefore: stitched,
    });
  }

  for (let i = 0; i < events.length - 1; i++) {
    const c0 = events[i];
    const c1 = events[i + 1];
    if (c1.stitchedBefore) continue;
    if (holds.length === 0) {
      holds.push(c0.holdMs / 1000);
      holdValid.push(true);
    }
    holds.push(c1.holdMs / 1000);
    holdValid.push(true);
    dds.push(c1.ddMs / 1000);
    uds.push(c1.udMs / 1000);
    stitchFlags.push(i > 0 && c0.stitchedBefore);
    pairKeys.push([c0.code, c1.code]);
  }

  const meanHoldMs = Math.round((holds.reduce((a, b) => a + b, 0) / holds.length) * 1000);
  const meanDdSec = dds.reduce((a, b) => a + b, 0) / dds.length;
  const cpm = Math.round(60 / meanDdSec);
  const rolloverPct = Math.round((uds.filter((u) => u < 0).length / uds.length) * 100);

  return {
    isPreset: true,
    presetMode: mode,
    sourceLabel:
      mode === 'slow'
        ? '예시 파형 · 같은 사람이 느리게 쳤을 때 (×0.62배속)'
        : '예시 파형 · 고속 연타 + 오타 백스페이스 봉합',
    typedText: '딥러닝 생체지문',
    backspaceCount: 1,
    cleanDigraphCount: dds.length,
    cpm,
    holdMs: meanHoldMs,
    rolloverPct,
    timeline: { events, poppedCount: 1 },
    inputMode: 'keyboard',
    behavior: null,
    stroke: { holds, holdValid, dds, uds, stitchFlags, pairKeys },
  };
}

/**
 * STEP 1A: Piano-Roll Keystroke Timeline SVG
 * Visualizes Hold bars, Rollover overlaps (UD < 0), and Backspace stitch boundaries.
 */
export function renderPianoRollSvg(capture, { maxEvents = 120 } = {}) {
  const allEvents = capture?.timeline?.events || [];
  if (allEvents.length < 2) {
    return `<div class="viz-empty">최소 2글자 이상 타이핑하면 실시간 건반 리듬 타임라인이 그려집니다.</div>`;
  }

  // Render full sentence sequence (no 14-key cutoff; container scrolls smoothly if very long)
  const slice = allEvents.length > maxEvents ? allEvents.slice(-maxEvents) : allEvents;

  const baseMs = slice[0].startMs;
  const normEvents = slice.map((e) => ({
    ...e,
    s: e.startMs - baseMs,
    e: e.endMs - baseMs,
  }));
  const totalMs = Math.max(400, normEvents[normEvents.length - 1].e + 60);

  const W = 710;
  const padL = 56;
  const padR = 88;
  const padT = 34;
  const rowH = 19;
  const H = padT + normEvents.length * rowH + 34;
  const plotW = W - padL - padR;
  const xOf = (ms) => padL + (ms / totalMs) * plotW;

  const gridParts = [];
  const stepMs = totalMs <= 1200 ? 200 : totalMs <= 2400 ? 400 : 600;
  for (let t = 0; t <= totalMs; t += stepMs) {
    const x = xOf(t).toFixed(1);
    gridParts.push(
      `<line x1="${x}" y1="${padT - 8}" x2="${x}" y2="${H - 24}" stroke="rgba(24,24,27,0.06)" stroke-width="1"/>`,
      `<text x="${x}" y="${H - 8}" text-anchor="middle" fill="#9ca3af" font-size="10.5" font-family="var(--font-mono)">${t}ms</text>`
    );
  }

  const bgBands = [];
  const rowItems = [];

  normEvents.forEach((ev, i) => {
    const yCenter = padT + i * rowH + 7;
    const x1 = xOf(ev.s);
    const x2 = Math.max(x1 + 8, xOf(ev.e));
    const prev = i > 0 ? normEvents[i - 1] : null;

    // Alternating subtle horizontal guide
    rowItems.push(
      `<line x1="${padL}" y1="${yCenter}" x2="${W - padR}" y2="${yCenter}" stroke="rgba(24,24,27,0.035)" stroke-width="1"/>`
    );

    // 1) Rollover highlight band (when current key starts BEFORE previous key released!)
    if (prev && ev.isRollover && ev.s < prev.e) {
      const rx1 = xOf(ev.s);
      const rx2 = Math.max(rx1 + 3, xOf(prev.e));
      const prevY = padT + (i - 1) * rowH + 7;
      bgBands.push(
        `<rect x="${rx1.toFixed(1)}" y="${(prevY - 6).toFixed(1)}" width="${(rx2 - rx1).toFixed(
          1
        )}" height="${(rowH + 12).toFixed(1)}" rx="3" fill="rgba(37,99,235,0.14)"/>`,
        `<line x1="${rx1.toFixed(1)}" y1="${prevY + 5}" x2="${rx1.toFixed(
          1
        )}" y2="${yCenter - 5}" stroke="#2563eb" stroke-width="1.2" stroke-dasharray="2 2"/>`
      );
    }

    // 2) Backspace / Pause Stitch Scissors Marker
    if (prev && ev.stitchedBefore) {
      const sx = ((xOf(prev.e) + x1) / 2).toFixed(1);
      bgBands.push(
        `<line x1="${sx}" y1="${padT - 12}" x2="${sx}" y2="${H - 24}" stroke="#d97706" stroke-width="1.5" stroke-dasharray="4 3"/>`,
        `<rect x="${(Number(sx) - 54).toFixed(1)}" y="${padT - 26}" width="108" height="17" rx="8.5" fill="#fffbeb" stroke="#fcd34d"/>`,
        `<text x="${sx}" y="${padT - 14}" text-anchor="middle" fill="#b45309" font-size="10" font-weight="700">✂ 오타 격리 · 봉합</text>`
      );
    }

    // Keycap badge on left
    const keyBg = ev.isRollover ? '#eff6ff' : '#f4f4f5';
    const keyStroke = ev.isRollover ? '#bfdbfe' : 'rgba(24,24,27,0.12)';
    const keyColor = ev.isRollover ? '#1d4ed8' : '#27272a';
    rowItems.push(
      `<rect x="10" y="${(yCenter - 7.5).toFixed(1)}" width="34" height="15" rx="4.5" fill="${keyBg}" stroke="${keyStroke}"/>`,
      `<text x="27" y="${(yCenter + 3.5).toFixed(1)}" text-anchor="middle" fill="${keyColor}" font-size="10.5" font-weight="700" font-family="var(--font-mono)">${esc(
        ev.label
      )}</text>`
    );

    // Hold duration bar
    const barFill = ev.stitchedBefore ? '#059669' : ev.isRollover ? '#2563eb' : '#18181b';
    rowItems.push(
      `<rect x="${x1.toFixed(1)}" y="${(yCenter - 5).toFixed(1)}" width="${(x2 - x1).toFixed(
        1
      )}" height="10" rx="5" fill="${barFill}"/>`
    );

    // Micro timing annotation on right of bar
    let metaLabel = `${ev.holdMs}ms`;
    let metaColor = '#71717a';
    if (ev.isRollover) {
      metaLabel = `${ev.holdMs}ms · 겹침 ${ev.udMs}ms`;
      metaColor = '#2563eb';
    } else if (ev.stitchedBefore) {
      metaLabel = `${ev.holdMs}ms · 가속도 0 리셋`;
      metaColor = '#059669';
    }
    rowItems.push(
      `<text x="${(x2 + 6).toFixed(1)}" y="${(yCenter + 3.5).toFixed(
        1
      )}" fill="${metaColor}" font-size="10" font-weight="600" font-family="var(--font-mono)">${metaLabel}</text>`
    );

    // Full-row interactive hover hit target
    const tipTitle = `타건 #${ev.idx + 1} · 키 [${ev.label}]`;
    const tipLines = [
      `누름(Hold): ${ev.holdMs} ms (${ev.s}ms → ${ev.e}ms)`,
      ev.ddMs !== null
        ? `키 간격(DD): ${ev.ddMs} ms`
        : ev.stitchedBefore
          ? '전환 상태: ✂ 오타 격리 후 리듬 봉합'
          : '전환 상태: 첫 타건 시작점',
      ev.udMs !== null
        ? `*비행(UD): ${ev.udMs} ms ${ev.isRollover ? '(연타 롤오버 중첩!)' : '(일반 순차 타건)'}`
        : ev.stitchedBefore
          ? '*보정: 가속도(Ch8) = 0.0 리셋 완료'
          : '',
    ]
      .filter(Boolean)
      .join('||');

    rowItems.push(
      `<rect class="viz-hit viz-row-hit" x="8" y="${(yCenter - 9.5).toFixed(1)}" width="${
        W - 16
      }" height="${rowH}" rx="5" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(
        tipLines
      )}"/>`
    );
  });

  return `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" aria-label="건반 피아노 롤 타건 타임라인">
    ${gridParts.join('')}
    ${bgBands.join('')}
    ${rowItems.join('')}
  </svg>`;
}

/**
 * STEP 1B: Tempo-Invariance Waveform Comparison SVG
 * Demonstrates why raw DD (ms) shifts dramatically when typing slower, while
 * normalized relative cadence (Rel_DD = DD / mean(DD)) stays invariant!
 */
export function renderTempoInvarianceSvg(capture) {
  const dds = capture?.stroke?.dds || [];
  // Touch keyboards report no hold time: show the same median the model imputes
  const holds = (capture?.stroke?.holds || []).map((h) => (h === null || h === undefined ? 0.088 : h));
  const stitches = capture?.stroke?.stitchFlags || [];

  if (dds.length < 3) {
    return `<div class="viz-empty">연속 타건 전이 구간이 3개 이상 모이면 속도 불변 정규화 파형이 표시됩니다.</div>`;
  }

  // Plot the entire variable-length sentence (matching KeyPrintNet [1, 17, L] full-sequence input!)
  const m = dds.length;
  const rawFastMs = dds.slice(0, m).map((v) => v * 1000);
  // Simulated 0.62x slower condition (1.61x interval + slight natural +/- 4ms micro-variance)
  const rawSlowMs = rawFastMs.map((v, i) => v * 1.61 + ((i % 3) - 1) * 4.5);

  const meanFast = rawFastMs.reduce((a, b) => a + b, 0) / m;
  const meanSlow = rawSlowMs.reduce((a, b) => a + b, 0) / m;
  const relFast = rawFastMs.map((v) => v / meanFast);
  const relSlow = rawSlowMs.map((v) => v / meanSlow);

  // Cosine similarity between fast & slow normalized rhythm vectors
  const cosMatch =
    (dot(normalizeVec(relFast), normalizeVec(relSlow)) * 100).toFixed(1);

  const W = 710;
  const H = 252;
  const padL = 54;
  const padR = 24;
  const plotW = W - padL - padR;
  const xAt = (i) => padL + (i / Math.max(1, m - 1)) * plotW;

  // Top subplot: Raw ms (y: 34..112)
  const maxMs = Math.max(320, ...rawSlowMs) * 1.08;
  const yRaw = (ms) => 110 - (Math.min(maxMs, Math.max(0, ms)) / maxMs) * 72;

  // Bottom subplot: Normalized Rel_DD (y: 162..234), range [0.3, 1.9]
  const yRel = (r) => 232 - ((Math.min(1.9, Math.max(0.3, r)) - 0.3) / 1.6) * 70;
  const yRelOne = yRel(1.0).toFixed(1);

  const poly = (vals, yFn) =>
    vals.map((v, i) => `${xAt(i).toFixed(1)},${yFn(v).toFixed(1)}`).join(' ');

  const stitchMarkers = [];
  for (let i = 0; i < m; i++) {
    if (stitches[i]) {
      const sx = xAt(i).toFixed(1);
      stitchMarkers.push(
        `<line x1="${sx}" y1="32" x2="${sx}" y2="234" stroke="#d97706" stroke-width="1.2" stroke-dasharray="3 3"/>`
      );
    }
  }

  // Relative Hold dots & full-height interactive scrub bands
  const relHolds = holds.slice(0, m);
  const meanH = relHolds.reduce((a, b) => a + b, 0) / Math.max(1, relHolds.length);
  const stepW = m > 1 ? plotW / (m - 1) : plotW;

  const fastDots = relFast
    .map(
      (v, i) =>
        `<circle cx="${xAt(i).toFixed(1)}" cy="${yRel(v).toFixed(
          1
        )}" r="3.4" fill="#2563eb" pointer-events="none"/>`
    )
    .join('');

  const hoverCols = relFast
    .map((v, i) => {
      const cx = xAt(i);
      const hx = Math.max(padL, cx - stepW / 2);
      const hw = Math.min(W - padR - hx, stepW);
      const rh = (relHolds[i] ?? meanH) / meanH;
      const tipTitle = `전이 구간 #${i + 1} / ${m}${stitches[i] ? ' (✂ 봉합 지점)' : ''}`;
      const tipLines = [
        `고속 원시 DD: ${Math.round(rawFastMs[i])} ms`,
        `저속 가정 DD: ${Math.round(rawSlowMs[i])} ms`,
        `*정규화 Rel_DD: ${v.toFixed(2)}x (저속 ${relSlow[i].toFixed(2)}x) · Rel_H: ${rh.toFixed(2)}x`,
      ].join('||');
      return `<rect class="viz-hit viz-col-hit" x="${hx.toFixed(
        1
      )}" y="32" width="${Math.max(8, hw).toFixed(
        1
      )}" height="204" rx="4" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(
        tipLines
      )}"/>`;
    })
    .join('');

  return `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" aria-label="원시 시간 대비 속도 불변 정규화 파형 비교">
    <!-- Top Panel: Raw Milliseconds -->
    <rect x="${padL}" y="32" width="${plotW}" height="80" rx="8" fill="#fafafa" stroke="rgba(24,24,27,0.06)"/>
    <text x="${padL}" y="22" fill="#18181b" font-size="11.5" font-weight="700">① 정규화 전: 원시 키 간격 DD (ms) — 피곤해서 천천히 치면 숫자가 크게 벌어짐</text>
    <text x="${W - padR}" y="22" text-anchor="end" fill="#71717a" font-size="10.5">실선 ${capture.cpm}타/분 vs 점선 ${Math.round(
      capture.cpm * 0.62
    )}타/분</text>
    <text x="${padL - 8}" y="44" text-anchor="end" fill="#9ca3af" font-size="9.5" font-family="var(--font-mono)">${Math.round(
      maxMs
    )}ms</text>
    <text x="${padL - 8}" y="110" text-anchor="end" fill="#9ca3af" font-size="9.5" font-family="var(--font-mono)">0ms</text>

    <polyline fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="5 4" points="${poly(
      rawSlowMs,
      yRaw
    )}"/>
    <polyline fill="none" stroke="#18181b" stroke-width="2.2" points="${poly(rawFastMs, yRaw)}"/>

    <!-- Bottom Panel: Tempo-Invariant Relative Rhythm -->
    <rect x="${padL}" y="158" width="${plotW}" height="78" rx="8" fill="#f8fafc" stroke="rgba(37,99,235,0.16)"/>
    <text x="${padL}" y="147" fill="#1d4ed8" font-size="11.5" font-weight="700">② 정규화 후: 상대 리듬 비율 (DD / 평균DD) — 속도가 변해도 개인 고유 박자 파형 ${cosMatch}% 일치</text>
    <line x1="${padL}" y1="${yRelOne}" x2="${W - padR}" y2="${yRelOne}" stroke="rgba(37,99,235,0.22)" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="${padL - 8}" y="${Number(yRelOne) + 3}" text-anchor="end" fill="#2563eb" font-size="9.5" font-weight="700" font-family="var(--font-mono)">1.0x</text>

    ${stitchMarkers.join('')}
    <polyline fill="none" stroke="#059669" stroke-width="2.6" stroke-dasharray="5 3" opacity="0.85" points="${poly(
      relSlow,
      yRel
    )}"/>
    <polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="${poly(relFast, yRel)}"/>
    ${fastDots}
    ${hoverCols}
  </svg>`;
}

/**
 * Compute partial centroid, spherical stability %, and signature dimension indices
 * for Step 2 (Profile Formation Inspector).
 */
export function analyzeProfileConvergence(template, stepCount) {
  const exemplars = Array.isArray(template?.exemplars) && template.exemplars.length > 0
    ? template.exemplars
    : template?.centroid
      ? [template.centroid]
      : [];

  const totalAvailable = exemplars.length;
  const k = Math.max(1, Math.min(stepCount || totalAvailable, totalAvailable));
  const activeExemplars = exemplars.slice(0, k);
  const dim = activeExemplars[0]?.length || 64;

  // L2-normalized centroid of the first k exemplars
  const sum = new Array(dim).fill(0);
  for (const ex of activeExemplars) {
    for (let d = 0; d < dim; d++) sum[d] += ex[d];
  }
  const partialCentroid = normalizeVec(sum);

  // Mean pairwise cosine correlation rho across all available exemplars of this person
  let pairSum = 0;
  let pairCnt = 0;
  for (let i = 0; i < exemplars.length; i++) {
    for (let j = i + 1; j < exemplars.length; j++) {
      pairSum += dot(exemplars[i], exemplars[j]);
      pairCnt += 1;
    }
  }
  const rho = pairCnt > 0 ? Math.max(0.35, Math.min(0.92, pairSum / pairCnt)) : 0.64;

  // Expected cosine alignment with asymptotic true user center on unit hypersphere:
  // E[cos(c_k, mu)] = sqrt((k * rho) / (1 + (k - 1) * rho))
  const expectedCos = Math.sqrt((k * rho) / (1 + (k - 1) * rho));
  const stabilityPct = Math.round(expectedCos * 1000) / 10;

  // Find Top-8 most prominent & sign-consistent signature dimensions
  const scoredDims = [];
  for (let d = 0; d < dim; d++) {
    const sign = Math.sign(partialCentroid[d]);
    let sameSign = 0;
    for (const ex of activeExemplars) {
      if (Math.sign(ex[d]) === sign) sameSign += 1;
    }
    const consistency = sameSign / k;
    scoredDims.push({
      d,
      score: Math.abs(partialCentroid[d]) * (0.5 + 0.5 * consistency),
    });
  }
  scoredDims.sort((a, b) => b.score - a.score);
  const topDims = new Set(scoredDims.slice(0, 8).map((it) => it.d));

  return {
    exemplars,
    totalAvailable,
    activeCount: k,
    partialCentroid,
    stabilityPct,
    pairwiseCosPct: Math.round(rho * 1000) / 10,
    topDims,
  };
}

/**
 * STEP 2: Aligned 64-D Biometric Barcode Stack -> Centroid Convergence SVG
 */
export function renderProfileBarcodeSvg(template, { color = '#2563eb', stepCount = 3, unitLabel = '문장' } = {}) {
  const info = analyzeProfileConvergence(template, stepCount);
  const { exemplars, activeCount, partialCentroid, topDims } = info;
  if (exemplars.length === 0) {
    return { svg: `<div class="viz-empty">선택한 프로필에 임베딩 데이터가 없습니다.</div>`, info };
  }

  // Display up to 4 individual sample strips so height stays compact and scannable
  const displayRows = exemplars.slice(0, Math.min(4, exemplars.length));
  const dim = partialCentroid.length;

  const W = 710;
  const padL = 106;
  const padR = 20;
  const stripW = W - padL - padR;
  const colStep = stripW / dim;
  const barW = Math.max(4.5, colStep - 2.2);

  const rowH = 30;
  const rowGap = 8;
  const padT = 26;
  const mergeGap = 38;
  const centroidH = 42;
  const H = padT + displayRows.length * (rowH + rowGap) + mergeGap + centroidH + 22;

  const parts = [];

  // Column header ticks & Top-8 invariant signature column highlights
  for (let d = 0; d < dim; d++) {
    const x = padL + d * colStep;
    if (topDims.has(d)) {
      parts.push(
        `<rect x="${(x - 1).toFixed(1)}" y="${padT - 6}" width="${(barW + 2).toFixed(
          1
        )}" height="${H - padT - 8}" rx="3" fill="${color}" opacity="0.07"/>`
      );
    }
    if (d % 16 === 0 || d === 63) {
      parts.push(
        `<text x="${(x + barW / 2).toFixed(1)}" y="15" text-anchor="middle" fill="#9ca3af" font-size="9.5" font-family="var(--font-mono)">D${
          d + 1
        }</text>`
      );
    }
  }

  const drawBarcodeRow = (vec, yTop, height, { active, isCentroid, label, subLabel }) => {
    const midY = yTop + height / 2;
    const rowOpacity = active ? 1 : 0.23;

    parts.push(
      `<g opacity="${rowOpacity}">`,
      `<text x="10" y="${(midY + (subLabel ? -2 : 4)).toFixed(
        1
      )}" fill="${isCentroid ? '#18181b' : '#3f3f46'}" font-size="${
        isCentroid ? '11.5' : '11'
      }" font-weight="${isCentroid ? '800' : '650'}">${esc(label)}</text>`
    );
    if (subLabel) {
      parts.push(
        `<text x="10" y="${(midY + 11).toFixed(1)}" fill="#71717a" font-size="9.5">${esc(
          subLabel
        )}</text>`
      );
    }

    parts.push(
      `<rect x="${padL - 4}" y="${yTop}" width="${stripW + 8}" height="${height}" rx="7" fill="${
        isCentroid ? '#f8fafc' : '#fafafa'
      }" stroke="${isCentroid ? color : 'rgba(24,24,27,0.08)'}" stroke-width="${
        isCentroid ? '1.5' : '1'
      }"/>`,
      `<line x1="${padL - 2}" y1="${midY.toFixed(1)}" x2="${
        padL + stripW + 2
      }" y2="${midY.toFixed(1)}" stroke="rgba(24,24,27,0.08)" stroke-width="1"/>`
    );

    const maxScale = 0.28; // Typical L2-normalized 64-D component range [-0.28, +0.28]
    for (let d = 0; d < dim; d++) {
      const val = vec[d] || 0;
      const mag = Math.min(1, Math.abs(val) / maxScale);
      const halfH = Math.max(1.8, mag * (height * 0.44));
      const x = padL + d * colStep;
      const y = val >= 0 ? midY - halfH : midY;
      const barColor = val >= 0 ? color : '#475569';
      const op = (0.28 + 0.72 * mag).toFixed(2);
      parts.push(
        `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(
          1
        )}" height="${halfH.toFixed(1)}" rx="1.6" fill="${barColor}" opacity="${op}" pointer-events="none"/>`
      );
    }
    parts.push(`</g>`);
  };

  displayRows.forEach((vec, idx) => {
    const yTop = padT + idx * (rowH + rowGap);
    const isIncluded = idx < activeCount;
    drawBarcodeRow(vec, yTop, rowH, {
      active: isIncluded,
      isCentroid: false,
      label: `${unitLabel} #${idx + 1}`,
      subLabel: isIncluded ? '수렴 반영됨' : '미반영 (대기)',
    });
  });

  // Merge divider banner
  const mergeY = padT + displayRows.length * (rowH + rowGap) + 14;
  parts.push(
    `<line x1="${padL}" y1="${mergeY}" x2="${W - padR}" y2="${mergeY}" stroke="rgba(24,24,27,0.12)" stroke-dasharray="4 3"/>`,
    `<rect x="${(padL + stripW / 2 - 142).toFixed(1)}" y="${mergeY - 10}" width="284" height="20" rx="10" fill="#ffffff" stroke="rgba(24,24,27,0.14)"/>`,
    `<text x="${(padL + stripW / 2).toFixed(
      1
    )}" y="${mergeY + 3.5}" text-anchor="middle" fill="#18181b" font-size="10.2" font-weight="700">↓ 글자 배열 노이즈 상쇄 · 고유 리듬 평균화 (L2 Centroid)</text>`
  );

  const centroidY = mergeY + 20;
  drawBarcodeRow(partialCentroid, centroidY, centroidH, {
    active: true,
    isCentroid: true,
    label: '대표 프로필 지문',
    subLabel: `${activeCount}개 누적 완성`,
  });

  // Full-height column hover strips across all sentences + Centroid for each of the 64 dimensions!
  const fmtSigned = (n) => `${n >= 0 ? '+' : ''}${n.toFixed(4)}`;
  for (let d = 0; d < dim; d++) {
    const x = padL + d * colStep;
    const isSig = topDims.has(d);
    const tipTitle = `임베딩 차원 D${d + 1} / 64${isSig ? ' · ★ 고유 불변 차원' : ''}`;
    const lines = displayRows.map((vec, idx) => {
      const tag = idx < activeCount ? '' : ' (미반영)';
      return `${unitLabel} #${idx + 1}: ${fmtSigned(vec[d] || 0)}${tag}`;
    });
    lines.push(`*대표 Centroid: ${fmtSigned(partialCentroid[d] || 0)}`);
    parts.push(
      `<rect class="viz-hit viz-col-hit" x="${(x - 1).toFixed(1)}" y="${
        padT - 6
      }" width="${colStep.toFixed(1)}" height="${
        H - padT - 8
      }" rx="3" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(lines.join('||'))}"/>`
    );
  }

  const svg = `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" aria-label="64차원 생체 지문 바코드 누적 및 대표 프로필 수렴 과정">
    ${parts.join('')}
  </svg>`;

  return { svg, info };
}

/**
 * Deterministic 2D PCA basis (u1, u2) on 64-D vectors via centered covariance power iteration.
 */
function compute2DPcaBasis(vectors) {
  const dim = vectors[0]?.length || 64;
  const n = vectors.length;
  const mean = new Array(dim).fill(0);
  for (const v of vectors) {
    for (let d = 0; d < dim; d++) mean[d] += v[d] / n;
  }
  const centered = vectors.map((v) => v.map((x, d) => x - mean[d]));

  const powerIter = (deflateVec = null) => {
    // Deterministic non-zero seed vector
    let u = new Array(dim).fill(0).map((_, d) => Math.cos((d + 1) * 0.73) + 0.2 * Math.sin(d * 1.9));
    u = normalizeVec(u);

    for (let iter = 0; iter < 28; iter++) {
      const next = new Array(dim).fill(0);
      for (const row of centered) {
        const proj = dot(row, u);
        for (let d = 0; d < dim; d++) next[d] += row[d] * proj;
      }
      if (deflateVec) {
        const overlap = dot(next, deflateVec);
        for (let d = 0; d < dim; d++) next[d] -= overlap * deflateVec[d];
      }
      const norm = Math.hypot(...next);
      if (norm < 1e-8) break;
      u = next.map((x) => x / norm);
    }

    // Canonical sign so chart orientation never flips arbitrarily
    let maxIdx = 0;
    for (let d = 1; d < dim; d++) {
      if (Math.abs(u[d]) > Math.abs(u[maxIdx])) maxIdx = d;
    }
    if (u[maxIdx] < 0) u = u.map((x) => -x);
    return u;
  };

  const u1 = powerIter(null);
  const u2 = powerIter(u1);
  return { mean, u1, u2 };
}

/**
 * STEP 3: Live Multi-User 2D Hypersphere Constellation Map SVG
 */
export function renderHypersphereMapSvg(entries, queryPoint = null) {
  // entries: [{ id, name, color, isSample, centroid, exemplars }]
  if (!entries || entries.length === 0) {
    return `<div class="viz-empty">표시할 참가자 프로필이 없습니다.</div>`;
  }

  // Fit 2D PCA specifically on person centroids first to maximize between-user separation!
  const centroidVecs = entries.map((e) => e.centroid).filter(Boolean);
  const allVecs = [];
  for (const ent of entries) {
    if (ent.centroid) allVecs.push(ent.centroid);
    if (Array.isArray(ent.exemplars)) {
      for (const ex of ent.exemplars) allVecs.push(ex);
    }
  }
  if (queryPoint?.embedding) allVecs.push(queryPoint.embedding);

  const basisSource = centroidVecs.length >= 3 ? centroidVecs : allVecs;
  const { mean, u1, u2 } = compute2DPcaBasis(basisSource);
  const projectRaw = (v) => {
    const shifted = v.map((x, d) => x - mean[d]);
    return { x: dot(shifted, u1), y: dot(shifted, u2) };
  };

  const rawCoords = allVecs.map(projectRaw);
  let maxSpanX = 0.14;
  let maxSpanY = 0.14;
  for (const p of rawCoords) {
    maxSpanX = Math.max(maxSpanX, Math.abs(p.x));
    maxSpanY = Math.max(maxSpanY, Math.abs(p.y));
  }
  maxSpanX *= 1.28;
  maxSpanY *= 1.32;

  const W = 710;
  const H = 316;
  const cx = W / 2;
  const cy = H / 2 + 4;
  const rxScale = (W - 160) / 2;
  const ryScale = (H - 96) / 2;

  const clampScreen = (pt) => ({
    x: Math.max(58, Math.min(W - 58, pt.x)),
    y: Math.max(44, Math.min(H - 46, pt.y)),
  });

  const toScreen = (v) => {
    const p = projectRaw(v);
    return clampScreen({
      x: cx + (p.x / maxSpanX) * rxScale,
      y: cy - (p.y / maxSpanY) * ryScale,
    });
  };

  const parts = [
    `<rect x="12" y="12" width="${W - 24}" height="${H - 24}" rx="14" fill="#fafafa" stroke="rgba(24,24,27,0.07)"/>`,
    // Subtle hypersphere polar rings & axes
    `<ellipse cx="${cx}" cy="${cy}" rx="${(rxScale * 0.45).toFixed(1)}" ry="${(
      ryScale * 0.45
    ).toFixed(1)}" fill="none" stroke="rgba(24,24,27,0.05)" stroke-dasharray="3 3"/>`,
    `<ellipse cx="${cx}" cy="${cy}" rx="${(rxScale * 0.85).toFixed(1)}" ry="${(
      ryScale * 0.85
    ).toFixed(1)}" fill="none" stroke="rgba(24,24,27,0.05)" stroke-dasharray="3 3"/>`,
    `<line x1="28" y1="${cy}" x2="${W - 28}" y2="${cy}" stroke="rgba(24,24,27,0.06)"/>`,
    `<line x1="${cx}" y1="26" x2="${cx}" y2="${H - 24}" stroke="rgba(24,24,27,0.06)"/>`,
    `<text x="${W - 26}" y="${cy - 6}" text-anchor="end" fill="#9ca3af" font-size="9.5" font-family="var(--font-mono)">PC1 (주성분 축 1)</text>`,
    `<text x="${cx + 8}" y="28" fill="#9ca3af" font-size="9.5" font-family="var(--font-mono)">PC2 (주성분 축 2)</text>`,
  ];

  const projectedEntries = entries.map((ent) => {
    const cPos = toScreen(ent.centroid);
    const exPos = (ent.exemplars || []).map((ex, i) => ({
      ...toScreen(ex),
      idx: i + 1,
    }));
    return { ...ent, cPos, exPos, radius: 24 };
  });

  // Gentle collision-avoidance relaxation between centroids (shifts exemplars rigidly with centroid)
  const minCentroidDist = 66;
  for (let pass = 0; pass < 16; pass++) {
    for (let i = 0; i < projectedEntries.length; i++) {
      for (let j = i + 1; j < projectedEntries.length; j++) {
        const a = projectedEntries[i];
        const b = projectedEntries[j];
        let dx = b.cPos.x - a.cPos.x;
        let dy = b.cPos.y - a.cPos.y;
        let dist = Math.hypot(dx, dy);
        if (dist < minCentroidDist) {
          if (dist < 1e-3) {
            dx = 1;
            dy = 0;
            dist = 1;
          }
          const push = ((minCentroidDist - dist) / 2) * 0.65;
          const ux = (dx / dist) * push;
          const uy = (dy / dist) * push;
          const nextA = clampScreen({ x: a.cPos.x - ux, y: a.cPos.y - uy });
          const nextB = clampScreen({ x: b.cPos.x + ux, y: b.cPos.y + uy });
          const dAx = nextA.x - a.cPos.x;
          const dAy = nextA.y - a.cPos.y;
          const dBx = nextB.x - b.cPos.x;
          const dBy = nextB.y - b.cPos.y;
          a.cPos = nextA;
          b.cPos = nextB;
          a.exPos = a.exPos.map((ep) => clampScreen({ ...ep, x: ep.x + dAx, y: ep.y + dAy }));
          b.exPos = b.exPos.map((ep) => clampScreen({ ...ep, x: ep.x + dBx, y: ep.y + dBy }));
        }
      }
    }
  }

  for (const ent of projectedEntries) {
    let radius = 24;
    for (const ep of ent.exPos) {
      const dist = Math.hypot(ep.x - ent.cPos.x, ep.y - ent.cPos.y);
      if (dist + 9 > radius) radius = Math.min(68, dist + 9);
    }
    ent.radius = radius;
  }

  // Layer 1: Territory halos & spokes
  for (const ent of projectedEntries) {
    parts.push(
      `<circle cx="${ent.cPos.x.toFixed(1)}" cy="${ent.cPos.y.toFixed(1)}" r="${ent.radius.toFixed(
        1
      )}" fill="${ent.color}" fill-opacity="0.07" stroke="${
        ent.color
      }" stroke-opacity="0.26" stroke-dasharray="4 3"/>`
    );
    for (const ep of ent.exPos) {
      parts.push(
        `<line x1="${ent.cPos.x.toFixed(1)}" y1="${ent.cPos.y.toFixed(1)}" x2="${ep.x.toFixed(
          1
        )}" y2="${ep.y.toFixed(1)}" stroke="${ent.color}" stroke-opacity="0.28" stroke-width="1.1"/>`,
        `<circle cx="${ep.x.toFixed(1)}" cy="${ep.y.toFixed(
          1
        )}" r="4.2" fill="#ffffff" stroke="${ent.color}" stroke-width="2">
          <title>${esc(ent.name)} · 개별 샘플 #${ep.idx}</title>
        </circle>`
      );
    }
  }

  // Layer 2: Live Query Connector (if any)
  let qScreen = null;
  let targetEnt = null;
  if (queryPoint?.embedding) {
    qScreen = toScreen(queryPoint.embedding);
    targetEnt =
      projectedEntries.find((e) => e.id === queryPoint.winnerId) || projectedEntries[0];
    if (targetEnt) {
      // Iteratively keep query star >= 54px away from ALL profile avatars so it never covers any node
      for (let pass = 0; pass < 10; pass++) {
        for (const ent of projectedEntries) {
          let vx = qScreen.x - ent.cPos.x;
          let vy = qScreen.y - ent.cPos.y;
          let vd = Math.hypot(vx, vy);
          if (vd < 54) {
            if (vd < 1e-3) {
              vx = 1;
              vy = -0.4;
              vd = Math.hypot(vx, vy);
            }
            const push = (54 - vd) * 0.85;
            qScreen = clampScreen({
              x: qScreen.x + (vx / vd) * push,
              y: qScreen.y + (vy / vd) * push,
            });
          }
        }
      }

      const lineCol = queryPoint.identified ? '#2563eb' : '#e11d48';
      parts.push(
        `<line x1="${qScreen.x.toFixed(1)}" y1="${qScreen.y.toFixed(
          1
        )}" x2="${targetEnt.cPos.x.toFixed(1)}" y2="${targetEnt.cPos.y.toFixed(
          1
        )}" stroke="${lineCol}" stroke-width="2" stroke-dasharray="5 4"/>`
      );
    }
  }

  // Layer 3: Centroid Avatar Badges & Labels
  for (const ent of projectedEntries) {
    const initial = esc(ent.name.trim().charAt(0) || '?');
    const shortName = esc(ent.name.replace(/\s*\(샘플.*\)$/, ''));
    const labelY = ent.cPos.y > H - 52 ? ent.cPos.y - 16 : ent.cPos.y + 23;
    parts.push(
      `<circle cx="${ent.cPos.x.toFixed(1)}" cy="${ent.cPos.y.toFixed(
        1
      )}" r="11" fill="${ent.color}" stroke="#ffffff" stroke-width="2.2">
        <title>${esc(ent.name)} 대표 프로필 중심점 (Centroid)</title>
      </circle>`,
      `<text x="${ent.cPos.x.toFixed(1)}" y="${(ent.cPos.y + 3.8).toFixed(
        1
      )}" text-anchor="middle" fill="#ffffff" font-size="10.5" font-weight="800">${initial}</text>`,
      `<rect x="${(ent.cPos.x - 24).toFixed(1)}" y="${(labelY - 10.5).toFixed(
        1
      )}" width="48" height="14" rx="4" fill="#fafafa" fill-opacity="0.85"/>`,
      `<text x="${ent.cPos.x.toFixed(1)}" y="${labelY.toFixed(
        1
      )}" text-anchor="middle" fill="#18181b" font-size="11" font-weight="700">${shortName}</text>`
    );
  }

  // Layer 4: Live Query Star Marker on top
  if (qScreen) {
    const qCol = queryPoint.identified ? '#18181b' : '#e11d48';
    const placeBelow = targetEnt ? qScreen.y >= targetEnt.cPos.y : false;
    const qLabelY = placeBelow
      ? Math.min(H - 18, qScreen.y + 24)
      : Math.max(26, qScreen.y - 16);
    const simSuffix = queryPoint.similarityPct ? ` (${queryPoint.similarityPct}%)` : '';
    const fullQueryLabel = `${queryPoint.label || '방금 입력한 타건'}${simSuffix}`;

    parts.push(
      `<circle cx="${qScreen.x.toFixed(1)}" cy="${qScreen.y.toFixed(
        1
      )}" r="12" fill="${qCol}" stroke="#ffffff" stroke-width="2.5"/>`,
      `<text x="${qScreen.x.toFixed(1)}" y="${(qScreen.y + 4).toFixed(
        1
      )}" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="800">★</text>`,
      `<rect x="${(qScreen.x - 66).toFixed(1)}" y="${(qLabelY - 11).toFixed(
        1
      )}" width="132" height="15" rx="5" fill="#fafafa" fill-opacity="0.9"/>`,
      `<text x="${qScreen.x.toFixed(1)}" y="${qLabelY.toFixed(
        1
      )}" text-anchor="middle" fill="${qCol}" font-size="10.2" font-weight="800">${esc(
        fullQueryLabel
      )}</text>`
    );
  }

  // Layer 5: Topmost generous hit circles so hover tooltips work effortlessly on every node!
  for (const ent of projectedEntries) {
    (ent.exPos || []).forEach((ep, idx) => {
      const exVec = ent.exemplars?.[idx];
      const cosToCenter = exVec && ent.centroid ? (dot(exVec, ent.centroid) * 100).toFixed(1) : '—';
      const tipTitle = `${ent.name} · 개별 샘플 #${ep.idx}`;
      const tipLines = [
        `대표 중심점(Centroid)과 코사인 일치율: ${cosToCenter}%`,
        `2D 투영 좌표: (${((ep.x - cx) / rxScale).toFixed(2)}, ${((cy - ep.y) / ryScale).toFixed(2)})`,
      ].join('||');
      parts.push(
        `<circle class="viz-node-hit" cx="${ep.x.toFixed(1)}" cy="${ep.y.toFixed(
          1
        )}" r="10" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(tipLines)}"/>`
      );
    });
  }

  for (const ent of projectedEntries) {
    const tmpl = ent.template || {};
    const tipTitle = `${ent.name} — 대표 프로필 중심`;
    const tipLines = [
      `누적 샘플 수: ${tmpl.count || ent.exemplars?.length || 1}개`,
      `평균 타속: ${tmpl.cpm || 0} 타/분 · 홀드: ${tmpl.holdMs || 0} ms`,
      `*연타 롤오버 비율: ${tmpl.rolloverPct || 0}%`,
    ].join('||');
    parts.push(
      `<circle class="viz-node-hit" cx="${ent.cPos.x.toFixed(1)}" cy="${ent.cPos.y.toFixed(
        1
      )}" r="18" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(tipLines)}"/>`
    );
  }

  if (qScreen && queryPoint) {
    const tipTitle = `★ ${queryPoint.label || '방금 입력한 타건'}`;
    const tipLines = [
      `가장 가까운 프로필: ${queryPoint.winnerName || '—'}`,
      `*AS-Norm 생체 일치도: ${queryPoint.similarityPct || '—'}% (${
        queryPoint.identified ? '식별 통과' : '임계치 미달/외부인'
      })`,
    ].join('||');
    parts.push(
      `<circle class="viz-node-hit" cx="${qScreen.x.toFixed(1)}" cy="${qScreen.y.toFixed(
        1
      )}" r="18" data-tip-title="${esc(tipTitle)}" data-tip-lines="${esc(tipLines)}"/>`
    );
  }

  return `<svg viewBox="0 0 ${W} ${H}" class="viz-svg" aria-label="참가자 64차원 초구면 생체 지문 2D 투영 지도">
    ${parts.join('')}
  </svg>`;
}
