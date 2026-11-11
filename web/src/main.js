import {
  KeyPrintEngine,
  synthesizeBotStroke,
  validateAndBuildStroke,
} from './biometrics.js';

const MODIFIER_KEYS = new Set([
  'Shift',
  'Control',
  'Alt',
  'Meta',
  'CapsLock',
  'Tab',
  'Escape',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'ArrowDown',
]);

const PHYSICAL_TYPING_CODE_RE =
  /^(Key[A-Z]|Digit[0-9]|Space|Minus|Equal|BracketLeft|BracketRight|Semicolon|Quote|Backquote|Backslash|Comma|Period|Slash)$/;

class LiveKeystrokeRecorder {
  constructor(inputEl, { onProgress, onSubmitAttempt } = {}) {
    this.inputEl = inputEl;
    this.onProgress = onProgress;
    this.onSubmitAttempt = onSubmitAttempt;
    this.reset();
    this.bind();
  }

  reset() {
    this.records = [];
    this.openKeys = new Map();
    this.hadRepeat = false;
    this.hadBackspace = false;
    this.hadPaste = false;
    if (this.onProgress) this.onProgress(0, '', 0);
  }

  bind() {
    this.inputEl.addEventListener('paste', (e) => {
      e.preventDefault();
      this.hadPaste = true;
      this.onSubmitAttempt({
        records: [],
        typedText: this.inputEl.value,
        hadPaste: true,
        hadRepeat: false,
        hadBackspace: false,
      });
      this.inputEl.value = '';
      this.reset();
    });

    this.inputEl.addEventListener('input', () => {
      if (this.onProgress) {
        const cpm = this.estimateLiveCpm();
        this.onProgress(this.records.length, this.inputEl.value, cpm);
      }
    });

    this.inputEl.addEventListener('keydown', (e) => {
      if (MODIFIER_KEYS.has(e.key)) return;

      if (e.repeat) {
        this.hadRepeat = true;
        return;
      }

      if (e.key === 'Backspace' || e.key === 'Delete') {
        this.hadBackspace = true;
        return;
      }

      const now = performance.now();

      if (e.key === 'Enter') {
        e.preventDefault();
        const finalize = () => {
          const enterRec = { code: 'Enter', label: '⏎', downTime: now, upTime: now + 72.0 };
          const finalizedRecords = [...this.records, enterRec];
          const payload = {
            records: finalizedRecords,
            typedText: this.inputEl.value,
            hadPaste: this.hadPaste,
            hadRepeat: this.hadRepeat,
            hadBackspace: this.hadBackspace,
          };
          this.inputEl.value = '';
          this.reset();
          this.onSubmitAttempt(payload);
        };
        if (e.isComposing) setTimeout(finalize, 18);
        else finalize();
        return;
      }

      // Support both ASCII/symbol keys AND Korean Hangul IME physical jamo keystrokes
      if (e.key.length === 1 || e.key === 'Process' || PHYSICAL_TYPING_CODE_RE.test(e.code)) {
        if (this.inputEl.value.length === 0 && this.records.length > 0 && e.key !== 'Process') {
          this.reset();
        }
        const rec = {
          code: e.code,
          label: e.key === ' ' ? '␣' : e.key === 'Process' ? e.code.replace('Key', '') : e.key,
          downTime: now,
          upTime: null,
        };
        this.records.push(rec);
        this.openKeys.set(e.code, rec);
        if (this.onProgress) {
          this.onProgress(this.records.length, this.inputEl.value, this.estimateLiveCpm());
        }
      }
    });

    this.inputEl.addEventListener('keyup', (e) => {
      const rec = this.openKeys.get(e.code);
      if (rec && rec.upTime === null) {
        rec.upTime = performance.now();
        this.openKeys.delete(e.code);
      }
    });
  }

  estimateLiveCpm() {
    if (this.records.length < 2) return 0;
    const spanSec = (this.records[this.records.length - 1].downTime - this.records[0].downTime) / 1000.0;
    if (spanSec < 0.05) return 0;
    return Math.min(1200, Math.round(((this.records.length - 1) / spanSec) * 60));
  }
}

// Application State
const engine = new KeyPrintEngine();
const state = {
  activeMode: 'guard', // 'guard' | 'arena'
  targetPassword: '.tie5Roanl',
  enrolledItems: [],
  template: null,
  activePreset: null,
  threshold: 0.56,
  calibratedThreshold: 0.56,
  strictThreshold: 0.68,
  openSetIdThreshold: 0.52,

  // Mode 2: 1:N Typing-Practice Arena State
  arenaSentence: 'we believe that we can c',
  arenaStagingStrokes: [],
  arenaGallery: [], // [{ id, name, badge, enrolledItems, template, blindStroke, crossStroke }]
  arenaActivePresetGallery: null,
  arenaAutoIndex: 1,
};

const el = {
  announcer: document.getElementById('sr-announcer'),
  engineStatus: document.getElementById('engine-status'),
  latency: document.getElementById('inference-latency'),
  tabGuard: document.getElementById('tab-mode-guard'),
  tabArena: document.getElementById('tab-mode-arena'),
  guardView: document.getElementById('guard-view'),
  arenaView: document.getElementById('arena-view'),

  // Mode 1 DOM
  targetPw: document.getElementById('target-password'),
  resetBtn: document.getElementById('reset-enroll-btn'),
  presetContainer: document.getElementById('benchmark-preset-buttons'),
  enrollInput: document.getElementById('enroll-input'),
  enrollCount: document.getElementById('enroll-count'),
  liveCounter: document.getElementById('live-key-counter'),
  enrollDots: document.querySelectorAll('#enroll-dots .dot'),
  enrollFeedback: document.getElementById('enroll-feedback'),
  enrollFeedbackText: document.getElementById('enroll-feedback-text'),
  templateCard: document.getElementById('template-summary-card'),
  templateOwner: document.getElementById('template-owner-label'),
  templateCohesion: document.getElementById('template-cohesion'),
  templateWpm: document.getElementById('template-wpm'),
  templateHold: document.getElementById('template-hold'),
  templateRollover: document.getElementById('template-rollover'),
  thresholdSlider: document.getElementById('threshold-slider'),
  thresholdDisplay: document.getElementById('threshold-display'),
  verifyInput: document.getElementById('verify-input'),
  simGenuineBtn: document.getElementById('sim-genuine-btn'),
  simGenuineSub: document.getElementById('sim-genuine-sub'),
  simHumanBtn: document.getElementById('sim-human-hacker-btn'),
  simMacroBtn: document.getElementById('sim-macro-bot-btn'),
  simEagleBtn: document.getElementById('sim-eagle-btn'),
  verdictCard: document.getElementById('verdict-card'),
  verdictBadge: document.getElementById('verdict-badge'),
  verdictTitle: document.getElementById('verdict-title'),
  verdictDesc: document.getElementById('verdict-desc'),
  verdictScore: document.getElementById('verdict-score'),
  gaugeMarker: document.getElementById('gauge-threshold-marker'),
  canvas: document.getElementById('rhythm-canvas'),
  digraphBars: document.getElementById('digraph-bars'),

  // Mode 2 DOM
  arenaSentenceInput: document.getElementById('arena-sentence-input'),
  arenaClearGalleryBtn: document.getElementById('arena-clear-gallery-btn'),
  arenaPresetGalleryBtns: document.getElementById('arena-preset-gallery-btns'),
  arenaUserName: document.getElementById('arena-user-name'),
  arenaRepCount: document.getElementById('arena-rep-count'),
  arenaLiveCpm: document.getElementById('arena-live-cpm'),
  arenaLiveAcc: document.getElementById('arena-live-acc'),
  arenaEnrollPrompt: document.getElementById('arena-enroll-prompt'),
  arenaEnrollInput: document.getElementById('arena-enroll-input'),
  arenaEnrollDots: document.querySelectorAll('#arena-enroll-dots .dot'),
  arenaEnrollFeedback: document.getElementById('arena-enroll-feedback'),
  arenaEnrollFeedbackText: document.getElementById('arena-enroll-feedback-text'),
  arenaGalleryCount: document.getElementById('arena-gallery-count'),
  arenaGalleryGrid: document.getElementById('arena-gallery-grid'),
  arenaBlindPrompt: document.getElementById('arena-blind-prompt'),
  arenaBlindInput: document.getElementById('arena-blind-input'),
  arenaBlindCpm: document.getElementById('arena-blind-cpm'),
  arenaSimRandomBtn: document.getElementById('arena-sim-random-btn'),
  arenaSimCrossBtn: document.getElementById('arena-sim-cross-btn'),
  arenaSimOutsiderBtn: document.getElementById('arena-sim-outsider-btn'),
  arenaVerdictCard: document.getElementById('arena-verdict-card'),
  arenaVerdictBadge: document.getElementById('arena-verdict-badge'),
  arenaVerdictTitle: document.getElementById('arena-verdict-title'),
  arenaVerdictDesc: document.getElementById('arena-verdict-desc'),
  arenaVerdictScore: document.getElementById('arena-verdict-score'),
  arenaThresholdTag: document.getElementById('arena-threshold-tag'),
  arenaRankingList: document.getElementById('arena-ranking-list'),
  arenaCanvas: document.getElementById('arena-rhythm-canvas'),

  // Shared Audit Log
  auditTbody: document.getElementById('audit-tbody'),
  clearLogBtn: document.getElementById('clear-log-btn'),
};

function announce(msg) {
  el.announcer.textContent = '';
  setTimeout(() => {
    el.announcer.textContent = msg;
  }, 60);
}

function switchMode(mode) {
  state.activeMode = mode;
  const isGuard = mode === 'guard';
  el.tabGuard.classList.toggle('active', isGuard);
  el.tabGuard.setAttribute('aria-selected', String(isGuard));
  el.tabArena.classList.toggle('active', !isGuard);
  el.tabArena.setAttribute('aria-selected', String(!isGuard));
  el.guardView.hidden = !isGuard;
  el.arenaView.hidden = isGuard;
}

function getDigraphLabels(pw) {
  const chars = [...pw, '⏎'];
  const labels = [];
  for (let i = 0; i < chars.length - 1; i++) {
    const a = chars[i] === ' ' ? '␣' : chars[i];
    const b = chars[i + 1] === ' ' ? '␣' : chars[i + 1];
    labels.push(`${a}→${b}`);
  }
  return labels;
}

function renderTypingPrompt(containerEl, targetText, typedSoFar = '') {
  containerEl.innerHTML = '';
  const targetChars = [...targetText];
  const typedChars = [...typedSoFar];
  let correctCount = 0;

  targetChars.forEach((ch, idx) => {
    const span = document.createElement('span');
    span.className = 'tp-char';
    span.textContent = ch === ' ' ? '␣' : ch;
    if (idx < typedChars.length) {
      if (typedChars[idx] === ch) {
        span.classList.add('correct');
        correctCount++;
      } else {
        span.classList.add('wrong');
      }
    } else if (idx === typedChars.length) {
      span.classList.add('current');
    }
    containerEl.appendChild(span);
  });

  if (typedChars.length === 0) return 100;
  const denom = Math.max(1, Math.min(targetChars.length, typedChars.length));
  return Math.round((correctCount / denom) * 100);
}

function setBanner(bannerEl, textEl, kind, icon, text) {
  bannerEl.className = `feedback-banner ${kind}`;
  bannerEl.querySelector('.fb-icon').textContent = icon;
  textEl.textContent = text;
}

// ============================================================================
// MODE 1: 1:1 PASSWORD INTRUSION GUARD
// ============================================================================
function updateThresholdUI(val) {
  state.threshold = Number(val);
  el.thresholdSlider.value = state.threshold.toFixed(2);
  el.thresholdDisplay.textContent = state.threshold.toFixed(2);
  const pct = Math.min(98, Math.max(2, Math.round(state.threshold * 100)));
  el.gaugeMarker.style.left = `${pct}%`;
}

function resetEnrollment(customPw = null) {
  if (customPw !== null) {
    state.targetPassword = customPw;
    el.targetPw.value = customPw;
  } else {
    state.targetPassword = el.targetPw.value.trim();
  }
  state.enrolledItems = [];
  state.template = null;
  state.activePreset = null;

  document.querySelectorAll('.preset-user-btn').forEach((b) => b.classList.remove('active'));
  el.enrollCount.textContent = '0';
  el.enrollDots.forEach((d) => d.classList.remove('filled'));
  el.templateCard.hidden = true;
  el.enrollInput.placeholder = `위 암호(${state.targetPassword})를 오타 없이 치고 Enter를 누르세요`;

  el.verifyInput.disabled = true;
  el.verifyInput.placeholder = '먼저 STEP 1에서 타건 지문을 등록하거나 프리셋을 선택하세요';
  el.simGenuineBtn.disabled = true;
  el.simHumanBtn.disabled = true;
  el.simMacroBtn.disabled = true;
  el.simEagleBtn.disabled = true;

  setBanner(el.enrollFeedback, el.enrollFeedbackText, 'neutral', '⌨️', `비밀번호 "${state.targetPassword}"를 평소 리듬대로 5회 입력해 주세요.`);
  drawOscilloscope(el.canvas, null, null, state.targetPassword);
  el.digraphBars.innerHTML = '<p class="empty-placeholder">검증을 실행하면 어느 키 구간에서 리듬 차이가 발생했는지 표시됩니다.</p>';

  el.verdictCard.className = 'verdict-card state-idle';
  el.verdictCard.style.setProperty('--score-pct', '0%');
  el.verdictBadge.textContent = '대기 중';
  el.verdictTitle.textContent = '검증 대기 중';
  el.verdictDesc.textContent = '비밀번호를 타이핑하거나 시뮬레이터 버튼을 눌러 추론을 실행하세요.';
  el.verdictScore.textContent = '--%';
}

function summarizeStrokeMetrics(template, enrolledItems) {
  const avgStepSec = template.meanDD.reduce((a, b) => a + b, 0) / template.meanDD.length;
  const cpm = Math.round(60.0 / Math.max(0.05, avgStepSec));
  const avgHoldMs = Math.round(
    (template.meanHold.reduce((a, b) => a + b, 0) / template.meanHold.length) * 1000
  );
  const rollCount = enrolledItems.reduce(
    (acc, it) => acc + it.cleanStroke.uds.filter((u) => u < 0).length,
    0
  );
  const totalTransitions = enrolledItems.length * template.meanUD.length;
  const rollPct = Math.round((rollCount / Math.max(1, totalTransitions)) * 100);
  return { cpm, avgHoldMs, rollPct };
}

function finalizeTemplate(ownerName) {
  state.template = engine.buildTemplate(state.enrolledItems);
  el.templateCard.hidden = false;
  el.templateOwner.textContent = ownerName;
  el.templateCohesion.textContent = `${(state.template.consistency * 100).toFixed(1)}%`;

  const { cpm, avgHoldMs, rollPct } = summarizeStrokeMetrics(state.template, state.enrolledItems);
  el.templateWpm.textContent = `${cpm} 타/분`;
  el.templateHold.textContent = `${avgHoldMs} ms`;
  el.templateRollover.textContent = `${rollPct}%`;

  el.verifyInput.disabled = false;
  el.verifyInput.placeholder = `암호(${state.targetPassword})를 타이핑하고 Enter로 본인 인증 테스트!`;
  el.simGenuineBtn.disabled = false;
  el.simHumanBtn.disabled = false;
  el.simMacroBtn.disabled = false;
  el.simEagleBtn.disabled = false;
  el.simGenuineSub.textContent = state.activePreset
    ? `${state.activePreset.subject}의 미학습 타 세션 주입`
    : '내 등록 리듬 미세 변주 주입';

  drawOscilloscope(el.canvas, state.template, null, state.targetPassword);
}

async function handleEnrollmentAttempt(payload) {
  const check = validateAndBuildStroke(payload, state.targetPassword);
  if (!check.ok) {
    setBanner(el.enrollFeedback, el.enrollFeedbackText, 'warn', '⚠️', `예외 처리 [${check.code}]: ${check.message}`);
    announce(check.message);
    return;
  }

  if (state.enrolledItems.length >= 5) {
    state.enrolledItems = [];
    state.activePreset = null;
  }

  const { embedding, latencyMs, cleanStroke } = await engine.embedStroke(check.stroke);
  el.latency.textContent = `${latencyMs} ms`;
  state.enrolledItems.push({ embedding, cleanStroke });

  const count = state.enrolledItems.length;
  el.enrollCount.textContent = String(count);
  el.enrollDots.forEach((d, idx) => d.classList.toggle('filled', idx < count));

  if (count < 5) {
    setBanner(
      el.enrollFeedback,
      el.enrollFeedbackText,
      'ok',
      '✅',
      `${count}/5회 타건 캡처 완료 (추론 ${latencyMs}ms). 동일한 리듬으로 ${5 - count}회 더 입력해 주세요.`
    );
    drawOscilloscope(el.canvas, engine.buildTemplate(state.enrolledItems), cleanStroke, state.targetPassword, true);
  } else {
    finalizeTemplate(`사용자 직접 등록 프로필 ("${state.targetPassword}")`);
    setBanner(
      el.enrollFeedback,
      el.enrollFeedbackText,
      'ok',
      '🔐',
      '5회 타건 지문 등록 완료! 이제 우측 STEP 2에서 평소 리듬 vs 변칙 타법/해커 침입을 실험해 보세요.'
    );
    el.verifyInput.focus();
  }
}

async function loadBenchmarkPreset(preset, btnEl) {
  resetEnrollment(preset.password);
  state.activePreset = preset;
  document.querySelectorAll('.preset-user-btn').forEach((b) => b.classList.remove('active'));
  btnEl.classList.add('active');

  let lastLatency = 0;
  for (const s of preset.enrollment_strokes) {
    const res = await engine.embedStroke(s);
    lastLatency = res.latencyMs;
    state.enrolledItems.push({ embedding: res.embedding, cleanStroke: res.cleanStroke });
  }
  el.latency.textContent = `${lastLatency} ms`;
  el.enrollCount.textContent = '5';
  el.enrollDots.forEach((d) => d.classList.add('filled'));

  finalizeTemplate(`${preset.label} (Session 1 등록 템플릿)`);
  setBanner(
    el.enrollFeedback,
    el.enrollFeedbackText,
    'ok',
    '⚡',
    `${preset.label}의 5회 등록 타건이 ONNX로 임베딩되었습니다! 우측 공격 시뮬레이터 또는 직접 타이핑으로 테스트하세요.`
  );
}

async function evaluateVerificationStroke(stroke, scenarioLabel) {
  if (!state.template) return;

  const { embedding, latencyMs, cleanStroke } = await engine.embedStroke(stroke);
  el.latency.textContent = `${latencyMs} ms`;

  const result = engine.scoreAgainstTemplate(embedding, cleanStroke, state.template, state.threshold);
  const pct = Math.max(0, Math.min(100, result.similarity * 100));

  el.verdictCard.style.setProperty('--score-pct', `${pct.toFixed(1)}%`);
  el.verdictScore.textContent = `${pct.toFixed(1)}%`;

  if (result.accepted) {
    el.verdictCard.className = 'verdict-card state-granted';
    el.verdictBadge.textContent = '✅ ACCESS GRANTED (본인 인증 승인)';
    el.verdictTitle.textContent = '정당한 소유자 타건 지문과 일치합니다';
    el.verdictDesc.textContent = `[${scenarioLabel}] 생체 코사인 일치도 ${pct.toFixed(1)}% ≥ 보안 임계값 ${(state.threshold * 100).toFixed(0)}% (추론 ${latencyMs}ms)`;
  } else {
    el.verdictCard.className = 'verdict-card state-blocked';
    el.verdictBadge.textContent = '🚨 INTRUSION BLOCKED (해커 침입 차단)';
    el.verdictTitle.textContent = '비밀번호는 맞지만 타건 생체 지문이 불일치합니다!';
    el.verdictDesc.textContent = `[${scenarioLabel}] 생체 코사인 일치도 ${pct.toFixed(1)}% < 보안 임계값 ${(state.threshold * 100).toFixed(0)}% → 계정 탈취 시도로 판단하여 접속을 거부했습니다.`;
  }

  announce(`${el.verdictBadge.textContent}. 일치도 ${pct.toFixed(1)}퍼센트.`);
  drawOscilloscope(el.canvas, state.template, cleanStroke, state.targetPassword, result.accepted);
  renderDigraphBars(result.digraphDeviations);
  appendAuditRow(`[1:1 차단] ${scenarioLabel}`, '일치 (Pass)', `${pct.toFixed(1)}% (τ=${(state.threshold * 100).toFixed(0)}%)`, `${latencyMs} ms`, result.accepted ? 'PASS' : 'BLOCK');
}

async function handleVerifyTyping(payload) {
  const check = validateAndBuildStroke(payload, state.targetPassword);
  if (!check.ok) {
    el.verdictCard.className = 'verdict-card state-exception';
    el.verdictCard.style.setProperty('--score-pct', '0%');
    el.verdictBadge.textContent = `⚠️ 입력 예외 감지 (${check.code})`;
    el.verdictTitle.textContent = '생체 신호 검증 전 입력 예외 처리됨';
    el.verdictDesc.textContent = check.message;
    el.verdictScore.textContent = 'N/A';
    appendAuditRow(
      '[1:1 차단] 직접 입력',
      check.code === 'PASSWORD_MISMATCH' ? '불일치 (오타)' : check.code,
      '--',
      '--',
      'WARN'
    );
    announce(check.message);
    return;
  }
  await evaluateVerificationStroke(check.stroke, '실시간 키보드 직접 입력');
}

function renderDigraphBars(deviations) {
  const labels = getDigraphLabels(state.targetPassword);
  el.digraphBars.innerHTML = '';

  deviations.forEach((z, idx) => {
    const cell = document.createElement('div');
    cell.className = 'dg-cell';
    const pct = Math.min(100, Math.max(12, Math.round((z / 4.0) * 100)));
    let color = '#10b981';
    if (z >= 2.2) color = '#f43f5e';
    else if (z >= 1.25) color = '#f59e0b';

    cell.innerHTML = `
      <div class="dg-bar-wrap">
        <div class="dg-bar-fill" style="--dg-pct: ${pct}%; --dg-color: ${color};"></div>
      </div>
      <span class="dg-key">${labels[idx] || `K${idx + 1}`}</span>
      <span class="dg-val">${z.toFixed(1)}σ</span>
    `;
    el.digraphBars.appendChild(cell);
  });
}

// ============================================================================
// MODE 2: 1:N TYPING PRACTICE AUTOMATIC PERSON CLASSIFIER ("WHO IS TYPING?")
// ============================================================================
function resetArenaSentence(newPhrase = null) {
  if (newPhrase !== null) {
    state.arenaSentence = newPhrase;
    el.arenaSentenceInput.value = newPhrase;
  } else {
    state.arenaSentence = el.arenaSentenceInput.value.trim();
  }
  state.arenaStagingStrokes = [];
  state.arenaGallery = [];
  state.arenaActivePresetGallery = null;
  state.arenaAutoIndex = 1;

  document.querySelectorAll('#arena-preset-gallery-btns .preset-user-btn').forEach((b) => b.classList.remove('active'));
  el.arenaRepCount.textContent = '0';
  el.arenaEnrollDots.forEach((d) => d.classList.remove('filled'));
  el.arenaUserName.value = '참가자 A';

  renderTypingPrompt(el.arenaEnrollPrompt, state.arenaSentence, '');
  renderTypingPrompt(el.arenaBlindPrompt, state.arenaSentence, '');
  updateArenaGalleryUI();

  setBanner(
    el.arenaEnrollFeedback,
    el.arenaEnrollFeedbackText,
    'neutral',
    '⌨️',
    `연습 문장("${state.arenaSentence}")을 참가자별로 3회씩 타이핑해 등록하거나 원클릭 8인 갤러리를 로드하세요.`
  );

  el.arenaVerdictCard.className = 'verdict-card state-idle';
  el.arenaVerdictBadge.textContent = '블라인드 추론 대기';
  el.arenaVerdictTitle.textContent = '누가 타이핑했는지 대기 중입니다';
  el.arenaVerdictDesc.textContent = '문장을 1회 타이핑하거나 블라인드 시뮬레이터 버튼을 눌러 인물을 분류해 보세요.';
  el.arenaVerdictScore.textContent = '--%';
  el.arenaRankingList.innerHTML = '<p class="empty-placeholder">블라인드 타자 추론을 실행하면 등록된 모든 후보 인물의 일치도 순위가 표시됩니다.</p>';
  drawOscilloscope(el.arenaCanvas, null, null, state.arenaSentence);
}

function updateArenaGalleryUI(winnerId = null) {
  const n = state.arenaGallery.length;
  el.arenaGalleryCount.textContent = String(n);

  const ready = n >= 2;
  el.arenaBlindInput.disabled = !ready;
  el.arenaBlindInput.placeholder = ready
    ? `누군가 나와서 "${state.arenaSentence}"를 1회 타이핑하고 Enter!`
    : '먼저 2명 이상 등록하거나 원클릭 8인 갤러리를 로드하세요';
  el.arenaSimRandomBtn.disabled = !ready;
  el.arenaSimCrossBtn.disabled = !(ready && state.arenaGallery.some((g) => g.crossStroke));
  el.arenaSimOutsiderBtn.disabled = !ready;

  if (n === 0) {
    el.arenaGalleryGrid.innerHTML =
      '<p class="empty-placeholder">아직 등록된 인물이 없습니다. 위 원클릭 8인 갤러리를 불러오거나 직접 3회 타자연습을 완료해 보세요!</p>';
    return;
  }

  el.arenaGalleryGrid.innerHTML = '';
  state.arenaGallery.forEach((profile) => {
    const card = document.createElement('div');
    card.className = `gallery-user-card${winnerId === profile.id ? ' winner-highlight' : ''}`;
    const { cpm, avgHoldMs, rollPct } = summarizeStrokeMetrics(profile.template, profile.enrolledItems);

    card.innerHTML = `
      <div class="g-card-top">
        <span class="g-card-name">👤 ${profile.name}</span>
        <button type="button" class="g-card-del" title="갤러리에서 제거" aria-label="${profile.name} 삭제">✕</button>
      </div>
      <span class="g-card-meta">${cpm}타/분 · 홀드 ${avgHoldMs}ms</span>
      <span class="g-card-meta">롤오버 ${rollPct}% · 🎯 클릭 테스트</span>
    `;

    card.querySelector('.g-card-del').addEventListener('click', (ev) => {
      ev.stopPropagation();
      state.arenaGallery = state.arenaGallery.filter((g) => g.id !== profile.id);
      updateArenaGalleryUI();
    });

    card.addEventListener('click', () => {
      if (state.arenaGallery.length < 2) return;
      injectProfileBlindTest(profile, false);
    });

    el.arenaGalleryGrid.appendChild(card);
  });
}

async function handleArenaEnrollmentAttempt(payload) {
  renderTypingPrompt(el.arenaEnrollPrompt, state.arenaSentence, '');
  const check = validateAndBuildStroke(payload, state.arenaSentence);
  if (!check.ok) {
    setBanner(
      el.arenaEnrollFeedback,
      el.arenaEnrollFeedbackText,
      'warn',
      '⚠️',
      `타자연습 예외 [${check.code}]: ${check.message}`
    );
    announce(check.message);
    return;
  }

  const userName = el.arenaUserName.value.trim() || `참가자 ${state.arenaAutoIndex}`;
  const { embedding, latencyMs, cleanStroke } = await engine.embedStroke(check.stroke);
  el.latency.textContent = `${latencyMs} ms`;
  state.arenaStagingStrokes.push({ embedding, cleanStroke });

  const rep = state.arenaStagingStrokes.length;
  el.arenaRepCount.textContent = String(rep);
  el.arenaEnrollDots.forEach((d, idx) => d.classList.toggle('filled', idx < rep));

  if (rep < 3) {
    setBanner(
      el.arenaEnrollFeedback,
      el.arenaEnrollFeedbackText,
      'ok',
      '✅',
      `[${userName}] ${rep}/3회 타자연습 완료 (${latencyMs}ms). ${3 - rep}회 더 치면 인물 프로필이 생성됩니다!`
    );
  } else {
    const tmpl = engine.buildTemplate(state.arenaStagingStrokes);
    const id = `custom_${Date.now()}_${state.arenaAutoIndex}`;
    state.arenaGallery.push({
      id,
      name: userName,
      badge: '실시간 등록',
      enrolledItems: [...state.arenaStagingStrokes],
      template: tmpl,
      blindStroke: null,
      crossStroke: null,
    });
    state.arenaStagingStrokes = [];
    state.arenaAutoIndex += 1;
    const nextLetter = String.fromCharCode(65 + (state.arenaGallery.length % 26));
    el.arenaUserName.value = `참가자 ${nextLetter}`;
    el.arenaRepCount.textContent = '0';
    el.arenaEnrollDots.forEach((d) => d.classList.remove('filled'));

    updateArenaGalleryUI();
    setBanner(
      el.arenaEnrollFeedback,
      el.arenaEnrollFeedbackText,
      'ok',
      '🎉',
      `[${userName}] 갤러리 등록 완료! 다음 참가자를 이어서 등록하거나 우측 STEP B에서 블라인드 분류를 테스트하세요.`
    );
  }
}

async function loadPresetArenaGallery(galleryCfg, btnEl) {
  resetArenaSentence(galleryCfg.practice_text);
  state.arenaActivePresetGallery = galleryCfg;
  document.querySelectorAll('#arena-preset-gallery-btns .preset-user-btn').forEach((b) => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');
  document.querySelectorAll('.arena-phrase-chip').forEach((c) => {
    c.classList.toggle('active', c.dataset.phrase === galleryCfg.practice_text);
  });

  let lastLat = 0;
  for (const member of galleryCfg.members) {
    const enrolledItems = [];
    for (const s of member.enrollment_strokes) {
      const res = await engine.embedStroke(s);
      lastLat = res.latencyMs;
      enrolledItems.push({ embedding: res.embedding, cleanStroke: res.cleanStroke });
    }
    state.arenaGallery.push({
      id: member.id,
      name: member.name,
      badge: member.badge,
      enrolledItems,
      template: engine.buildTemplate(enrolledItems),
      blindStroke: member.blind_test_stroke,
      crossStroke: member.cross_sentence_stroke || null,
    });
  }
  el.latency.textContent = `${lastLat} ms`;
  updateArenaGalleryUI();

  setBanner(
    el.arenaEnrollFeedback,
    el.arenaEnrollFeedbackText,
    'ok',
    '⚡',
    `${galleryCfg.title} (${state.arenaGallery.length}명 전원) 등록 완료! 우측 블라인드 시뮬레이터 또는 인물 카드를 클릭해 보세요.`
  );
}

async function runBlindArenaClassification(stroke, scenarioLabel, expectedProfile = null, isOutsiderExpected = false) {
  if (state.arenaGallery.length === 0) return;

  const { embedding, latencyMs, cleanStroke } = await engine.embedStroke(stroke);
  el.latency.textContent = `${latencyMs} ms`;

  const idRes = engine.identifyAgainstGallery(
    embedding,
    cleanStroke,
    state.arenaGallery,
    state.openSetIdThreshold
  );
  if (!idRes) return;

  const top = idRes.winner;
  const topPct = Math.max(0, Math.min(100, top.similarity * 100));
  el.arenaVerdictScore.textContent = `${topPct.toFixed(1)}%`;

  let auditStatus = 'PASS';
  if (!idRes.identified) {
    el.arenaVerdictCard.className = 'verdict-card state-exception';
    el.arenaVerdictBadge.textContent = '🕵️ UNKNOWN TYPIST (미등록 외부인 기각)';
    el.arenaVerdictTitle.textContent = `갤러리 내 일치하는 인물이 없습니다 (1위 후보 ${top.name}: ${topPct.toFixed(1)}%)`;
    el.arenaVerdictDesc.textContent = isOutsiderExpected
      ? `[${scenarioLabel}] 최고 일치도(${topPct.toFixed(1)}%) < 기각 임계값(${(state.openSetIdThreshold * 100).toFixed(0)}%) → 미등록 외부인을 정확히 분리했습니다! ✅`
      : `[${scenarioLabel}] 모든 등록 인물의 생체 임계값(${(state.openSetIdThreshold * 100).toFixed(0)}%) 미만으로 판정되었습니다.`;
    updateArenaGalleryUI(null);
    auditStatus = isOutsiderExpected ? 'PASS' : 'WARN';
  } else {
    const matchedTruth = !expectedProfile || expectedProfile.id === top.id;
    el.arenaVerdictCard.className = matchedTruth ? 'verdict-card state-granted' : 'verdict-card state-blocked';
    const truthTag = expectedProfile
      ? matchedTruth
        ? ' · ✅ 실제 정답 일치!'
        : ` · ❌ 실제: ${expectedProfile.name}`
      : '';
    el.arenaVerdictBadge.textContent = `🎯 1:N 예측 완료 (후보 ${state.arenaGallery.length}명 중 1위${truthTag})`;
    el.arenaVerdictTitle.textContent = `타이핑한 사람: ${top.name} (상대 신뢰도 ${(top.posterior * 100).toFixed(1)}%)`;
    el.arenaVerdictDesc.textContent = `[${scenarioLabel}] 1위 일치도 ${topPct.toFixed(1)}% (2위 대비 +${(idRes.margin * 100).toFixed(1)}%p 우위 · 추론 ${latencyMs}ms)`;
    updateArenaGalleryUI(top.id);
    auditStatus = matchedTruth && !isOutsiderExpected ? 'PASS' : 'BLOCK';
  }

  renderArenaRankingList(idRes.rankings, idRes.identified);
  drawOscilloscope(el.arenaCanvas, top.template, cleanStroke, state.arenaSentence, idRes.identified);
  appendAuditRow(
    `[1:N 식별] ${scenarioLabel}`,
    '일치 (Pass)',
    idRes.identified ? `예측: ${top.name} (${topPct.toFixed(1)}%)` : `미등록 외부인 (${topPct.toFixed(1)}%)`,
    `${latencyMs} ms`,
    auditStatus
  );
}

function renderArenaRankingList(rankings, isIdentified) {
  el.arenaRankingList.innerHTML = '';
  rankings.forEach((item, idx) => {
    const pct = Math.max(4, Math.min(100, item.similarity * 100));
    const isTop = idx === 0 && isIdentified;
    const color = isTop ? '#10b981' : idx === 0 ? '#f59e0b' : '#38bdf8';

    const row = document.createElement('div');
    row.className = `rank-row${isTop ? ' is-top1' : ''}`;
    row.innerHTML = `
      <span class="rank-num">${item.rank}위</span>
      <span class="rank-name">${isTop ? '🏆 ' : ''}${item.name}</span>
      <div class="rank-bar-track">
        <div class="rank-bar-fill" style="--rk-pct: ${pct.toFixed(1)}%; --rk-color: ${color};"></div>
      </div>
      <span class="rank-score">${pct.toFixed(1)}% <small style="color:#94a3b8;">(${Math.round(item.posterior * 100)}%)</small></span>
    `;
    el.arenaRankingList.appendChild(row);
  });
}

async function injectProfileBlindTest(profile, useCrossSentence = false) {
  let stroke = useCrossSentence && profile.crossStroke ? profile.crossStroke : profile.blindStroke;
  if (!stroke) {
    // Natural biological perturbation of user's custom enrolled stroke
    const base = profile.enrolledItems[0].cleanStroke;
    const holds = base.holds.map((h) => h * (0.96 + Math.random() * 0.08));
    const dds = base.dds.map((d) => d * (0.95 + Math.random() * 0.10));
    const uds = dds.map((d, i) => d - holds[i]);
    stroke = { holds, dds, uds };
  }
  const label = useCrossSentence
    ? `타 세션·타 문장 블라인드 (${profile.name})`
    : `블라인드 타건 주입 (실제: ${profile.name})`;
  await runBlindArenaClassification(stroke, label, profile, false);
}

async function handleArenaBlindTyping(payload) {
  renderTypingPrompt(el.arenaBlindPrompt, state.arenaSentence, '');
  const check = validateAndBuildStroke(payload, state.arenaSentence);
  if (!check.ok) {
    el.arenaVerdictCard.className = 'verdict-card state-exception';
    el.arenaVerdictBadge.textContent = `⚠️ 입력 예외 감지 (${check.code})`;
    el.arenaVerdictTitle.textContent = '타자연습 문장 예외 처리됨';
    el.arenaVerdictDesc.textContent = check.message;
    el.arenaVerdictScore.textContent = 'N/A';
    return;
  }
  await runBlindArenaClassification(check.stroke, '실시간 키보드 블라인드 타이핑', null, false);
}

// ============================================================================
// SHARED OSCILLOSCOPE & AUDIT TRAIL
// ============================================================================
function drawOscilloscope(canvas, template, queryStroke, labelText, isAccepted = false) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { left: 46, right: 20, top: 20, bottom: 32 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad.top + (plotH * g) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  if (!template) {
    ctx.fillStyle = '#94a3b8';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('프로필이 등록되면 실시간 타건 생체 리듬 파형이 여기에 렌더링됩니다.', w / 2, h / 2);
    return;
  }

  const m = template.meanDD.length;
  const labels = getDigraphLabels(labelText || '');
  const allVals = [
    ...template.meanDD.map((v, i) => (v + template.stdDD[i]) * 1000),
    ...(queryStroke ? queryStroke.dds.map((v) => v * 1000) : []),
  ];
  const maxMs = Math.max(360, Math.min(1300, Math.ceil(Math.max(...allVals) * 1.15)));

  const xAt = (i) => pad.left + (m === 1 ? plotW / 2 : (plotW * i) / (m - 1));
  const yAt = (ms) => pad.top + plotH - (Math.min(maxMs, Math.max(0, ms)) / maxMs) * plotH;

  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'right';
  for (let g = 0; g <= 4; g++) {
    const msVal = Math.round((maxMs * (4 - g)) / 4);
    const y = pad.top + (plotH * g) / 4;
    ctx.fillText(`${msVal}ms`, pad.left - 6, y + 4);
  }

  ctx.textAlign = 'center';
  const stepStride = m > 14 ? Math.ceil(m / 10) : 1;
  for (let i = 0; i < m; i += stepStride) {
    ctx.fillText(labels[i] || `K${i + 1}`, xAt(i), h - 10);
  }

  ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
  ctx.beginPath();
  for (let i = 0; i < m; i++) {
    const upper = (template.meanDD[i] + template.stdDD[i]) * 1000;
    if (i === 0) ctx.moveTo(xAt(i), yAt(upper));
    else ctx.lineTo(xAt(i), yAt(upper));
  }
  for (let i = m - 1; i >= 0; i--) {
    const lower = Math.max(10, (template.meanDD[i] - template.stdDD[i]) * 1000);
    ctx.lineTo(xAt(i), yAt(lower));
  }
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.3;
  ctx.beginPath();
  for (let i = 0; i < m; i++) {
    const ms = template.meanDD[i] * 1000;
    if (i === 0) ctx.moveTo(xAt(i), yAt(ms));
    else ctx.lineTo(xAt(i), yAt(ms));
  }
  ctx.stroke();

  if (queryStroke) {
    const qLen = Math.min(m, queryStroke.dds.length);
    const qColor = isAccepted ? '#10b981' : '#f43f5e';
    ctx.strokeStyle = qColor;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(isAccepted ? [] : [6, 4]);
    ctx.beginPath();
    for (let i = 0; i < qLen; i++) {
      const ms = queryStroke.dds[i] * 1000;
      if (i === 0) ctx.moveTo(xAt(i), yAt(ms));
      else ctx.lineTo(xAt(i), yAt(ms));
    }
    ctx.stroke();
    ctx.setLineDash([]);

    for (let i = 0; i < qLen; i++) {
      const ms = queryStroke.dds[i] * 1000;
      const isRollover = queryStroke.uds[i] < 0;
      ctx.fillStyle = isRollover ? '#a855f7' : qColor;
      ctx.beginPath();
      ctx.arc(xAt(i), yAt(ms), isRollover ? 4.8 : 3.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function appendAuditRow(scenario, pwCheck, scoreStr, latencyStr, status) {
  const emptyRow = document.getElementById('audit-empty-row');
  if (emptyRow) emptyRow.remove();

  const tr = document.createElement('tr');
  const timeStr = new Date().toLocaleTimeString('ko-KR', { hour12: false });

  let statusHtml = '<span class="pill-pass">✅ 정상 승인/일치</span>';
  if (status === 'BLOCK') statusHtml = '<span class="pill-block">🚨 침입/불일치 차단</span>';
  else if (status === 'WARN') statusHtml = '<span class="pill-warn">⚠️ 예외/외부인 분리</span>';

  tr.innerHTML = `
    <td><code>${timeStr}</code></td>
    <td>${scenario}</td>
    <td>${pwCheck}</td>
    <td><code>${scoreStr}</code></td>
    <td><code>${latencyStr}</code></td>
    <td>${statusHtml}</td>
  `;
  el.auditTbody.prepend(tr);
}

async function bootstrap() {
  el.tabGuard.addEventListener('click', () => switchMode('guard'));
  el.tabArena.addEventListener('click', () => switchMode('arena'));

  try {
    const cfg = await engine.init();
    state.calibratedThreshold = cfg.calibrated_threshold;
    state.strictThreshold = cfg.strict_threshold;
    state.openSetIdThreshold = cfg.open_set_id_threshold ?? Math.max(0.42, cfg.calibrated_threshold - 0.04);
    updateThresholdUI(cfg.calibrated_threshold);
    el.arenaThresholdTag.textContent = `외부인 기각 임계값 τ=${state.openSetIdThreshold.toFixed(2)}`;

    el.engineStatus.textContent = `ONNX WASM 활성 (${engine.loadTimeMs}ms)`;
    el.engineStatus.className = 't-val status-ready';

    // Mode 1 Preset Buttons
    cfg.benchmark_presets.forEach((preset, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-user-btn';
      btn.innerHTML = `<strong>👤 피험자 ${preset.subject}</strong><br><span style="color:#94a3b8;font-size:0.72rem;">Unseen Test #${idx + 1}</span>`;
      btn.addEventListener('click', () => loadBenchmarkPreset(preset, btn));
      el.presetContainer.appendChild(btn);
    });

    // Mode 2 Preset 8-Person Gallery Buttons
    (cfg.typing_arena_galleries || []).forEach((gal) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-user-btn arena-gal-btn';
      btn.dataset.galleryId = gal.gallery_id;
      btn.innerHTML = `<strong>⚡ ${gal.title}</strong><br><span style="color:#94a3b8;font-size:0.72rem;">${gal.description}</span>`;
      btn.addEventListener('click', () => loadPresetArenaGallery(gal, btn));
      el.arenaPresetGalleryBtns.appendChild(btn);
    });

    drawOscilloscope(el.canvas, null, null, state.targetPassword);
    resetArenaSentence(state.arenaSentence);
  } catch (err) {
    el.engineStatus.textContent = '모델 로드 오류';
    el.engineStatus.className = 't-val status-error';
    setBanner(el.enrollFeedback, el.enrollFeedbackText, 'err', '❌', `ONNX 초기화 실패: ${err.message}`);
  }

  // Bind Mode 1 Recorders
  new LiveKeystrokeRecorder(el.enrollInput, {
    onProgress: (cnt) => {
      el.liveCounter.textContent = cnt > 0 ? `${cnt}키 입력 중...` : '대기 중';
    },
    onSubmitAttempt: handleEnrollmentAttempt,
  });
  new LiveKeystrokeRecorder(el.verifyInput, {
    onSubmitAttempt: handleVerifyTyping,
  });

  // Bind Mode 2 Recorders with Live Typing Tutor Prompt Highlight!
  new LiveKeystrokeRecorder(el.arenaEnrollInput, {
    onProgress: (_, typed, cpm) => {
      const acc = renderTypingPrompt(el.arenaEnrollPrompt, state.arenaSentence, typed);
      el.arenaLiveCpm.textContent = `${cpm} 타/분`;
      el.arenaLiveAcc.textContent = `정확도 ${acc}%`;
    },
    onSubmitAttempt: handleArenaEnrollmentAttempt,
  });
  new LiveKeystrokeRecorder(el.arenaBlindInput, {
    onProgress: (_, typed, cpm) => {
      renderTypingPrompt(el.arenaBlindPrompt, state.arenaSentence, typed);
      el.arenaBlindCpm.textContent = cpm > 0 ? `실시간 ${cpm} 타/분` : '실시간 측정 대기';
    },
    onSubmitAttempt: handleArenaBlindTyping,
  });

  // Mode 1 Controls
  document.querySelectorAll('.pw-chip').forEach((chip) => {
    chip.addEventListener('click', () => resetEnrollment(chip.dataset.pw));
  });
  el.targetPw.addEventListener('change', () => {
    if (el.targetPw.value.trim().length < 6) {
      setBanner(el.enrollFeedback, el.enrollFeedbackText, 'warn', '⚠️', '비밀번호는 최소 6자 이상으로 설정해 주세요.');
      return;
    }
    resetEnrollment();
  });
  el.resetBtn.addEventListener('click', () => resetEnrollment());

  el.thresholdSlider.addEventListener('input', (e) => {
    document.querySelectorAll('.th-chip').forEach((c) => c.classList.remove('active'));
    updateThresholdUI(e.target.value);
  });
  document.querySelectorAll('.th-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.th-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      if (chip.dataset.mode === 'lenient') updateThresholdUI(Math.max(0.18, state.calibratedThreshold - 0.10));
      else if (chip.dataset.mode === 'strict') updateThresholdUI(Math.min(0.90, state.calibratedThreshold + 0.10));
      else updateThresholdUI(state.calibratedThreshold);
    });
  });

  el.simGenuineBtn.addEventListener('click', async () => {
    if (!state.template) return;
    if (state.activePreset) {
      await evaluateVerificationStroke(
        state.activePreset.genuine_test_stroke,
        `본인 정상 타건 (${state.activePreset.subject} 미학습 세션 6)`
      );
    } else {
      const base = state.enrolledItems[0].cleanStroke;
      const holds = base.holds.map((h) => h * (0.96 + Math.random() * 0.08));
      const dds = base.dds.map((d) => d * (0.95 + Math.random() * 0.10));
      const uds = dds.map((d, i) => d - holds[i]);
      await evaluateVerificationStroke({ holds, dds, uds }, '본인 정상 리듬 시뮬레이션');
    }
  });

  el.simHumanBtn.addEventListener('click', async () => {
    if (!state.template) return;
    if (state.activePreset) {
      await evaluateVerificationStroke(
        state.activePreset.skilled_impostor_stroke,
        `숙련 인간 해커 (${state.activePreset.impostor_subject} 동일 암호 타이핑)`
      );
    } else {
      const numKeys = state.targetPassword.length + 1;
      await evaluateVerificationStroke(
        synthesizeBotStroke(numKeys, 'gaussian_mimic'),
        '타인 유출 암호 타이핑 공격'
      );
    }
  });

  el.simMacroBtn.addEventListener('click', async () => {
    if (!state.template) return;
    const numKeys = state.targetPassword.length + 1;
    await evaluateVerificationStroke(
      synthesizeBotStroke(numKeys, 'uniform_macro'),
      '정밀 매크로 봇 (Constant-Delay Script)'
    );
  });

  el.simEagleBtn.addEventListener('click', async () => {
    if (!state.template) return;
    const numKeys = state.targetPassword.length + 1;
    await evaluateVerificationStroke(
      synthesizeBotStroke(numKeys, 'hunt_and_peck'),
      '독수리 타법 침입자 (메모지 유출 공격)'
    );
  });

  // Mode 2 Controls
  document.querySelectorAll('.arena-phrase-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.arena-phrase-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      resetArenaSentence(chip.dataset.phrase);
    });
  });
  el.arenaSentenceInput.addEventListener('change', () => {
    if (el.arenaSentenceInput.value.trim().length < 6) return;
    document.querySelectorAll('.arena-phrase-chip').forEach((c) => c.classList.remove('active'));
    resetArenaSentence();
  });
  el.arenaClearGalleryBtn.addEventListener('click', () => resetArenaSentence());

  let rrIndex = 0;
  el.arenaSimRandomBtn.addEventListener('click', async () => {
    if (state.arenaGallery.length < 2) return;
    const target = state.arenaGallery[rrIndex % state.arenaGallery.length];
    rrIndex += 1;
    await injectProfileBlindTest(target, false);
  });

  el.arenaSimCrossBtn.addEventListener('click', async () => {
    const candidates = state.arenaGallery.filter((g) => g.crossStroke);
    if (candidates.length === 0) return;
    const target = candidates[rrIndex % candidates.length];
    rrIndex += 1;
    await injectProfileBlindTest(target, true);
  });

  el.arenaSimOutsiderBtn.addEventListener('click', async () => {
    if (state.arenaGallery.length < 2) return;
    if (state.arenaActivePresetGallery?.outsider_stroke) {
      const out = state.arenaActivePresetGallery.outsider_stroke;
      await runBlindArenaClassification(out, out.name, null, true);
    } else {
      const numKeys = state.arenaSentence.length + 1;
      await runBlindArenaClassification(
        synthesizeBotStroke(numKeys, 'hunt_and_peck'),
        '미등록 외부인 타이핑 시뮬레이션',
        null,
        true
      );
    }
  });

  el.clearLogBtn.addEventListener('click', () => {
    el.auditTbody.innerHTML =
      '<tr id="audit-empty-row"><td colspan="6" class="empty-cell">아직 실행된 인증·분류 기록이 없습니다.</td></tr>';
  });
}

bootstrap();
