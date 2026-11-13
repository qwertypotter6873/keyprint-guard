import {
  KeyPrintEngine,
  SmartKeystrokeRecorder,
  composingTailIndex,
  computeLiveAccuracy,
  describeBehavior,
  combineEmbeddings,
  mergeTemplateSample,
  validatePasswordCapture,
  validateSentenceCapture,
} from './biometrics.js';
import { mergeBehavior } from './features.js';
import {
  RoomProfileSync,
  getInitialRoomCode,
  normalizeProfileName,
  pickProfileColor,
  sanitizeRoomCode,
} from './roomSync.js';
import { SentenceDeck } from './sentences.js';
import {
  createPresetCapture,
  renderHypersphereMapSvg,
  renderPianoRollSvg,
  renderProfileBarcodeSvg,
  renderTempoInvarianceSvg,
} from './visualizer.js';

// Typing-practice protocol comes from model_config.json (arena_enroll_sentences / arena_query_sentences)
let TARGET_ARENA_SHOTS = 5;
let TARGET_BLIND_SHOTS = 2;
const TARGET_PW_SHOTS = 4;

const engine = new KeyPrintEngine();
const deck = new SentenceDeck();

const state = {
  ready: false,
  loadError: null,
  mode: 'arena', // 'arena' | 'guard' | 'principles'
  arenaSub: 'enroll', // 'enroll' | 'blind'
  guardSub: 'verify', // 'verify' | 'enroll'
  lang: 'ko', // 'ko' | 'en'
  thresholdOffset: 0, // 여유(-) / 표준(0) / 엄격(+) relative to the validation-calibrated threshold
  showPasswordPlain: true,

  // Monotonic query counter & official pre-adaptation verdict snapshot
  querySeq: 0,
  lastVerdict: null,

  // Current prompts
  arenaEnrollSentence: '',
  arenaBlindSentence: '',

  // In-progress enrollment buffers
  arenaEnrollBuffer: { name: '', lang: 'ko', items: [] },
  arenaBlindBuffer: [],
  pwEnrollBuffer: { name: '', targetPw: '', items: [] },

  // Selected target account for Mode B (Password Guard)
  selectedGuardId: null,
  profiles: [],

  // Mode C: Interactive Visual Explainer & Profile Formation State
  viz: {
    waveTab: 'piano', // 'piano' | 'tempo'
    capture: createPresetCapture('normal'),
    selectedProfileId: null,
    shotCount: 99,
    mapDomain: 'arena', // 'arena' | 'guard'
    lastQuery: null, // { queryId, embedding, embeddings, label, domain }
  },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeColor(c, fallback = '#2563eb') {
  return typeof c === 'string' && /^#[0-9a-fA-F]{6}$/.test(c) ? c : fallback;
}

function safeInt(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n) : fallback;
}

function deepClone(obj) {
  if (obj === null || obj === undefined) return obj;
  return JSON.parse(JSON.stringify(obj));
}

function notReadyMessage() {
  return state.loadError
    ? `AI 생체지문 엔진 로드 실패 (${state.loadError}). 페이지를 새로고침해 주세요.`
    : 'AI 생체지문 엔진을 준비 중입니다. 잠시 후 다시 시도해 주세요.';
}

function invalidateLastVerdictOnTyping(live) {
  if (live && live.text && live.text.length > 0 && state.lastVerdict) {
    state.lastVerdict = null;
    state.querySeq += 1;
  }
}

function renderSentencePrompt(containerEl, targetSentence, typedValue) {
  const target = String(targetSentence || '');
  const typed = String(typedValue || '');
  const parts = [];
  const composing = composingTailIndex(typed, target); // half-built Hangul syllable: not a typo

  for (let i = 0; i < target.length; i++) {
    const ch = escapeHtml(target[i]);
    let cls = 'ch';
    if (i < typed.length) {
      cls += typed[i] === target[i] ? ' is-ok' : i === composing ? ' is-comp' : ' is-err';
    } else if (i === typed.length) {
      cls += ' is-caret';
    }
    parts.push(`<span class="${cls}">${ch}</span>`);
  }
  containerEl.innerHTML = parts.join('');
}

const INPUT_MODE_LABEL = { keyboard: '키보드', touch: '터치 키보드', mixed: '혼합 입력' };

function guardThreshold(inputMode) {
  return engine.passwordThreshold(inputMode) + state.thresholdOffset;
}

function behaviorNote(res) {
  if (!res || !res.behaviorWeight) return '';
  const delta = res.similarity - res.embeddingSimilarity;
  if (Math.abs(delta) < 0.005) return ' · 행동 프로필 근거 부족(영향 없음)';
  return ` · 행동 프로필 ${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}%p 반영`;
}

function deviceMismatchNote(template, inputMode) {
  const tMode = template?.inputMode;
  if (!tMode || tMode === 'mixed' || !inputMode || inputMode === 'mixed' || tMode === inputMode) return '';
  return ` · 주의: ${INPUT_MODE_LABEL[tMode] || tMode}로 등록된 프로필을 ${INPUT_MODE_LABEL[inputMode] || inputMode}로 비교 중(정확도 저하 가능)`;
}

function behaviorSummaryText(stats) {
  const d = describeBehavior(stats);
  if (!d) return '';
  const parts = [];
  if (d.backspacePct !== null) parts.push(`백스페이스 ${d.backspacePct}%`);
  if (d.corrections > 0 && d.singleFixPct !== null) parts.push(`한 글자 수정 ${d.singleFixPct}%`);
  if (d.capsStyle) parts.push(`대문자 ${d.capsStyle}`);
  if (d.rightShiftPct !== null && d.shiftCount >= 2) parts.push(`오른쪽 Shift ${d.rightShiftPct}%`);
  return parts.join(' · ');
}

function setFeedback(el, message, kind = '') {
  if (!el) return;
  el.textContent = message || '';
  el.className = 'feedback-line' + (kind ? ` is-${kind}` : '');
}

function renderStepDots(containerEl, labelEl, completedCount, totalCount, prefixText) {
  const safeTotal = Math.max(1, safeInt(totalCount, 1));
  const safeDone = Math.max(0, safeInt(completedCount, 0));
  labelEl.textContent = `${prefixText} ${Math.min(safeDone + 1, safeTotal)} / ${safeTotal}`;
  const dots = [];
  for (let i = 0; i < safeTotal; i++) {
    let cls = 'dot';
    if (i < safeDone) cls += ' is-done';
    else if (i === safeDone) cls += ' is-current';
    dots.push(`<span class="${cls}"></span>`);
  }
  containerEl.innerHTML = dots.join('');
}

// ================= ROOM SYNC SETUP =================
const roomSync = new RoomProfileSync({
  onChange: (profiles) => {
    state.profiles = Array.isArray(profiles) ? profiles : [];
    ensureSelectedGuardAccount();
    renderAllRosters();
    if (state.ready) renderEnrollMode();
    renderModalProfiles();
    if (state.mode === 'principles') {
      renderPrinciplesView();
    }
  },
  onStatusChange: ({ status, roomCode, peerCount, profileCount, compatSkipCount }) => {
    const syncDot = $('sync-dot');
    syncDot.className =
      'sync-dot ' +
      (status === 'online' ? 'is-online' : status === 'local' ? 'is-local' : 'is-connecting');

    const peerLabel = status === 'online' ? `기기 ${safeInt(peerCount, 1)}대` : '로컬';
    $('room-pill-label').textContent = `방 #${roomCode} · ${peerLabel} · 프로필 ${safeInt(profileCount, 0)}명`;

    $('modal-room-code').textContent = `#${roomCode}`;
    $('modal-sync-badge').textContent =
      status === 'online'
        ? `실시간 연결됨 · 현재 기기 ${safeInt(peerCount, 1)}대 접속`
        : status === 'local'
          ? '로컬 브라우저 저장 모드'
          : '실시간 서버 연결 중…';

    const compatNotice = $('compat-isolation-notice');
    if (compatNotice) {
      compatNotice.hidden = !(compatSkipCount > 0);
    }
  },
  onCompatibilitySkip: () => {
    const compatNotice = $('compat-isolation-notice');
    if (compatNotice) {
      compatNotice.hidden = false;
    }
  },
});

function getArenaTemplateForProfile(profile, lang) {
  if (!profile?.arena) return null;
  if (profile.arena[lang]) return profile.arena[lang];
  if (profile.isSample) {
    return profile.arena.en || profile.arena.ko || null;
  }
  return profile.arena.ko || profile.arena.en || null;
}

function getArenaCandidates() {
  return state.profiles
    .map((p) => {
      if (!p || typeof p.name !== 'string') return null;
      const tmpl = getArenaTemplateForProfile(p, state.lang);
      return tmpl?.centroid
        ? {
            id: String(p.id || ''),
            name: p.name,
            color: safeColor(p.color, '#2563eb'),
            isSample: Boolean(p.isSample),
            template: tmpl,
          }
        : null;
    })
    .filter(Boolean);
}

function getGuardAccounts() {
  return state.profiles.filter((p) => p && typeof p.name === 'string' && p.password && p.password.centroid);
}

function ensureSelectedGuardAccount() {
  const accounts = getGuardAccounts();
  if (accounts.length === 0) {
    state.selectedGuardId = null;
    return;
  }
  if (!accounts.some((a) => a.id === state.selectedGuardId)) {
    state.selectedGuardId = accounts[0].id;
  }
}

// ================= UI RENDERING =================
function switchMainMode(mode, { syncUrl = true } = {}) {
  const validMode = ['arena', 'guard', 'principles'].includes(mode) ? mode : 'arena';
  state.mode = validMode;

  const isArena = validMode === 'arena';
  const isGuard = validMode === 'guard';
  const isPrinciples = validMode === 'principles';
  if (!isPrinciples && vizEmbedTimer) {
    clearTimeout(vizEmbedTimer);
    vizEmbedTimer = null;
  }

  $('view-arena').hidden = !isArena;
  $('view-guard').hidden = !isGuard;
  $('view-principles').hidden = !isPrinciples;

  $('nav-arena-btn').classList.toggle('is-active', isArena);
  $('nav-arena-btn').setAttribute('aria-pressed', String(isArena));
  $('nav-guard-btn').classList.toggle('is-active', isGuard);
  $('nav-guard-btn').setAttribute('aria-pressed', String(isGuard));
  $('nav-principles-btn').classList.toggle('is-active', isPrinciples);
  $('nav-principles-btn').setAttribute('aria-pressed', String(isPrinciples));

  if (syncUrl) {
    const url = new URL(window.location.href);
    if (validMode === 'arena') url.searchParams.delete('tab');
    else url.searchParams.set('tab', validMode);
    window.history.replaceState({}, '', url.toString());
  }

  if (isArena) {
    focusActiveArenaInput();
  } else if (isGuard) {
    renderGuardVerifyState();
    focusActiveGuardInput();
  } else {
    renderPrinciplesView();
  }
}

function switchArenaSub(sub) {
  state.arenaSub = sub;
  const isEnroll = sub === 'enroll';
  $('arena-enroll-panel').hidden = !isEnroll;
  $('arena-blind-panel').hidden = isEnroll;

  $('arena-sub-enroll').classList.toggle('is-active', isEnroll);
  $('arena-sub-enroll').setAttribute('aria-selected', String(isEnroll));
  $('arena-sub-blind').classList.toggle('is-active', !isEnroll);
  $('arena-sub-blind').setAttribute('aria-selected', String(!isEnroll));

  focusActiveArenaInput();
}

function switchGuardSub(sub) {
  state.guardSub = sub;
  const isVerify = sub === 'verify';
  $('guard-verify-panel').hidden = !isVerify;
  $('guard-enroll-panel').hidden = isVerify;
  $('guard-threshold-group').hidden = !isVerify;

  $('guard-sub-verify').classList.toggle('is-active', isVerify);
  $('guard-sub-verify').setAttribute('aria-selected', String(isVerify));
  $('guard-sub-enroll').classList.toggle('is-active', !isVerify);
  $('guard-sub-enroll').setAttribute('aria-selected', String(!isVerify));

  renderGuardVerifyState();
  focusActiveGuardInput();
}

function focusActiveArenaInput() {
  setTimeout(() => {
    if (state.mode !== 'arena') return;
    if (state.arenaSub === 'enroll') {
      const nameEl = $('arena-name-input');
      if (!nameEl.value.trim()) nameEl.focus();
      else $('arena-enroll-input').focus();
    } else {
      $('arena-blind-input').focus();
    }
  }, 20);
}

function focusActiveGuardInput() {
  setTimeout(() => {
    if (state.mode !== 'guard') return;
    if (state.guardSub === 'verify') {
      if (getGuardAccounts().length > 0) $('guard-verify-input').focus();
    } else {
      const nameEl = $('guard-enroll-name');
      if (!nameEl.value.trim()) nameEl.focus();
      else $('guard-enroll-input').focus();
    }
  }, 20);
}

function renderAllRosters() {
  // 1) Arena Roster
  const arenaList = $('arena-roster-list');
  const cands = getArenaCandidates();
  if (cands.length === 0) {
    arenaList.innerHTML = `<span class="roster-empty">아직 없습니다 — 아래에서 ${safeInt(
      TARGET_ARENA_SHOTS,
      5
    )}문장을 연습해 첫 프로필을 등록하세요</span>`;
  } else {
    arenaList.innerHTML = cands
      .map((c) => {
        const safeName = String(c.name ?? '');
        const initial = escapeHtml(safeName.trim().charAt(0) || '?');
        const cnt = safeInt(c.template?.count, 1);
        const col = safeColor(c.color, '#2563eb');
        const tip = c.isSample
          ? '샘플 프로필은 시연용이라 수정할 수 없습니다'
          : '클릭하면 이 프로필에 문장을 추가로 연습합니다 (본인 확인 후 반영)';
        return `<button type="button" class="user-chip" data-arena-pick="${escapeHtml(safeName)}" data-sample="${
          c.isSample ? '1' : '0'
        }" title="${escapeHtml(tip)}">
          <span class="chip-avatar" style="background:${col}">${initial}</span>
          <span>${escapeHtml(safeName)}</span>
          <span class="chip-count">${cnt}문장</span>
        </button>`;
      })
      .join('');

    arenaList.querySelectorAll('[data-arena-pick]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const pickedName = btn.getAttribute('data-arena-pick') || '';
        if (state.arenaSub !== 'enroll') switchArenaSub('enroll');
        if (btn.getAttribute('data-sample') === '1') {
          setFeedback(
            $('arena-enroll-feedback'),
            `'${pickedName}'은(는) 샘플 프로필이라 문장을 추가할 수 없습니다. 내 이름으로 새로 등록해 주세요.`,
            'error'
          );
          $('arena-name-input').focus();
          return;
        }
        $('arena-name-input').value = pickedName;
        resetArenaEnrollProgress();
        setFeedback($('arena-enroll-feedback'), '');
        $('arena-enroll-input').focus();
      });
    });
  }

  // 2) Password Guard Accounts
  const guardList = $('guard-account-list');
  const accounts = getGuardAccounts();
  if (accounts.length === 0) {
    guardList.innerHTML = `<span class="roster-empty">등록된 보호 계정이 없습니다</span>`;
  } else {
    guardList.innerHTML = accounts
      .map((acc) => {
        const safeName = String(acc.name ?? '');
        const initial = escapeHtml(safeName.trim().charAt(0) || '?');
        const col = safeColor(acc.color, '#18181b');
        const selected = acc.id === state.selectedGuardId && state.guardSub === 'verify';
        return `<button type="button" class="user-chip ${selected ? 'is-selected' : ''}" data-guard-id="${escapeHtml(
          acc.id
        )}">
          <span class="chip-avatar" style="background:${col}">${initial}</span>
          <span>${escapeHtml(safeName)}</span>
        </button>`;
      })
      .join('');

    guardList.querySelectorAll('[data-guard-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.selectedGuardId = btn.getAttribute('data-guard-id');
        $('guard-result-card').hidden = true;
        setFeedback($('guard-verify-feedback'), '');
        if (state.guardSub !== 'verify') {
          switchGuardSub('verify');
        } else {
          renderAllRosters();
          renderGuardVerifyState();
          $('guard-verify-input').focus();
        }
      });
    });
  }

  renderGuardVerifyState();
}

function renderGuardVerifyState() {
  const accounts = getGuardAccounts();
  const hasAccounts = accounts.length > 0;
  $('guard-empty-state').hidden = hasAccounts;
  $('guard-active-box').hidden = !hasAccounts;

  if (!hasAccounts) return;
  const targetAcc = accounts.find((a) => a.id === state.selectedGuardId) || accounts[0];
  state.selectedGuardId = targetAcc.id;

  $('guard-target-name').textContent = String(targetAcc.name ?? '');
  const pw = String(targetAcc.password?.target ?? '');
  $('guard-target-pw').textContent = state.showPasswordPlain ? pw : '•'.repeat(pw.length);
  $('guard-toggle-pw-btn').textContent = state.showPasswordPlain ? '가리기' : '보기';
  $('guard-verify-input').type = state.showPasswordPlain ? 'text' : 'password';
}

function renderModalProfiles() {
  const listEl = $('modal-profile-list');
  $('modal-profile-count').textContent = String(state.profiles.length);

  const hasSamples = roomSync.hasSamples();
  $('toggle-samples-btn').textContent = hasSamples
    ? '샘플 프로필 모두 제거'
    : '실측 샘플 프로필 4명 추가';

  if (state.profiles.length === 0) {
    listEl.innerHTML = `<li class="profile-row"><span class="roster-empty">현재 방에 등록된 프로필이 없습니다.</span></li>`;
    return;
  }

  listEl.innerHTML = state.profiles
    .map((p) => {
      const safeName = String(p?.name ?? '');
      const initial = escapeHtml(safeName.trim().charAt(0) || '?');
      const col = safeColor(p?.color, '#2563eb');
      const badges = [];
      if (p?.arena?.ko) badges.push(`한글 ${safeInt(p.arena.ko.count, 1)}문장`);
      if (p?.arena?.en) badges.push(`영문 ${safeInt(p.arena.en.count, 1)}문장`);
      if (p?.password) badges.push(`암호(${safeInt(p.password.count, 1)}회)`);
      const refTmpl = p?.arena?.ko || p?.arena?.en || p?.password;
      const behTxt = refTmpl ? behaviorSummaryText(refTmpl.behavior) : '';
      const modeLabel = refTmpl?.inputMode ? INPUT_MODE_LABEL[refTmpl.inputMode] || '' : '';
      const modeTxt = modeLabel && refTmpl?.inputMode !== 'keyboard' ? ` · ${modeLabel}` : '';
      const statsTxt = refTmpl
        ? `평균 ${safeInt(refTmpl.cpm, 0)} 타/분 · 홀드 ${safeInt(refTmpl.holdMs, 0)}ms · 롤오버 ${safeInt(
            refTmpl.rolloverPct,
            0
          )}%${modeTxt}${behTxt ? ` · ${behTxt}` : ''}`
        : '';

      return `<li class="profile-row">
        <div class="profile-row-left">
          <span class="chip-avatar" style="background:${col};width:26px;height:26px;font-size:0.78rem">${initial}</span>
          <div>
            <div><strong>${escapeHtml(safeName)}</strong> <span class="chip-count">${escapeHtml(
              badges.join(' · ')
            )}</span></div>
            <div class="profile-meta-line">${escapeHtml(statsTxt)}</div>
          </div>
        </div>
        <button type="button" class="danger-sm-btn" data-del-id="${escapeHtml(p?.id)}">삭제</button>
      </li>`;
    })
    .join('');

  listEl.querySelectorAll('[data-del-id]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-del-id');
      roomSync.deleteProfile(id);
    });
  });
}

// ================= SENTENCE MANAGEMENT =================
function dealArenaEnrollSentence() {
  state.arenaEnrollSentence = deck.next(state.lang);
  $('arena-enroll-input').value = '';
  arenaEnrollRecorder.reset();
  renderSentencePrompt($('arena-enroll-prompt'), state.arenaEnrollSentence, '');
  $('arena-enroll-cpm').textContent = '0';
  $('arena-enroll-acc').textContent = '100%';
  $('arena-enroll-edit-hint').textContent = '오타 시 백스페이스 수정 가능';
}

function dealArenaBlindSentence() {
  state.arenaBlindSentence = deck.next(state.lang);
  $('arena-blind-input').value = '';
  arenaBlindRecorder.reset();
  renderSentencePrompt($('arena-blind-prompt'), state.arenaBlindSentence, '');
  $('arena-blind-cpm').textContent = '0';
  $('arena-blind-acc').textContent = '100%';
  $('arena-blind-edit-hint').textContent = '백스페이스 자유 사용';
}

function resetArenaBlindProgress() {
  state.arenaBlindBuffer = [];
  renderStepDots($('arena-blind-step-dots'), $('arena-blind-step-text'), 0, TARGET_BLIND_SHOTS, '문장');
  $('arena-blind-need').textContent = `문장 ${safeInt(TARGET_BLIND_SHOTS, 2)}개`;
}

function resetArenaEnrollProgress() {
  const typedName = normalizeProfileName($('arena-name-input').value);
  state.arenaEnrollBuffer = { name: typedName, lang: state.lang, items: [] };
  renderStepDots($('arena-step-dots'), $('arena-step-text'), 0, TARGET_ARENA_SHOTS, '문장');
  renderEnrollMode();
}

/** Existing (non-sample) profile with a template in this language -> extra-practice mode. */
function extraPracticeTarget() {
  const name = normalizeProfileName($('arena-name-input').value);
  const existing = name ? roomSync.findProfileByName(name) : null;
  return existing && existing.arena?.[state.lang] ? existing : null;
}

function renderEnrollMode() {
  const target = extraPracticeTarget();
  const dots = $('arena-step-dots');
  if (target) {
    const n = safeInt(target.arena[state.lang].count, 0);
    $('arena-step-text').textContent = `추가 연습 중 · '${target.name}' ${n}문장 (최근 12문장 반영, 본인 확인 후 추가)`;
    dots.hidden = true;
  } else {
    dots.hidden = false;
  }
}

/**
 * Sample names (exact sample names, loaded `isSample` profiles, or names containing `(샘플` / `샘플·`)
 * are blocked across ALL enrollment modes so demo fixtures cannot be spoofed or overwritten.
 */
export function isSampleName(name) {
  const norm = normalizeProfileName(name).toLowerCase();
  if (!norm) return false;
  if (norm.includes('(샘플') || norm.includes('샘플·')) return true;
  const samples = [
    ...(engine.config?.mvp_sample_profiles || []),
    ...state.profiles.filter((p) => p?.isSample),
  ];
  return samples.some(
    (p) => typeof p?.name === 'string' && normalizeProfileName(p.name).toLowerCase() === norm
  );
}

function switchLanguage(lang) {
  state.lang = lang;
  $('lang-ko-btn').classList.toggle('is-active', lang === 'ko');
  $('lang-en-btn').classList.toggle('is-active', lang === 'en');

  resetArenaEnrollProgress();
  resetArenaBlindProgress();
  dealArenaEnrollSentence();
  dealArenaBlindSentence();
  $('arena-result-card').hidden = true;
  setFeedback($('arena-enroll-feedback'), '');
  setFeedback($('arena-blind-feedback'), '');
  renderAllRosters();
  if (state.mode === 'principles') {
    renderPrinciplesView();
  } else {
    focusActiveArenaInput();
  }
}

// ================= KEYSTROKE RECORDERS =================
const arenaEnrollRecorder = new SmartKeystrokeRecorder($('arena-enroll-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
    renderSentencePrompt($('arena-enroll-prompt'), state.arenaEnrollSentence, live.text);
    $('arena-enroll-cpm').textContent = String(live.cpm);
    $('arena-enroll-acc').textContent = `${computeLiveAccuracy(live.text, state.arenaEnrollSentence)}%`;
    $('arena-enroll-edit-hint').textContent =
      live.backspaceCount > 0
        ? `백스페이스 ${live.backspaceCount}회 수정 보정됨`
        : '오타 시 백스페이스 수정 가능';
  },
  onSubmit: () => handleArenaEnrollSubmit(),
});

const arenaBlindRecorder = new SmartKeystrokeRecorder($('arena-blind-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
    renderSentencePrompt($('arena-blind-prompt'), state.arenaBlindSentence, live.text);
    $('arena-blind-cpm').textContent = String(live.cpm);
    $('arena-blind-acc').textContent = `${computeLiveAccuracy(live.text, state.arenaBlindSentence)}%`;
    $('arena-blind-edit-hint').textContent =
      live.backspaceCount > 0
        ? `백스페이스 ${live.backspaceCount}회 수정 보정됨`
        : '백스페이스 자유 사용';
  },
  onSubmit: () => handleArenaBlindSubmit(),
});

const guardVerifyRecorder = new SmartKeystrokeRecorder($('guard-verify-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
  },
  onSubmit: () => handleGuardVerifySubmit(),
});

const guardEnrollRecorder = new SmartKeystrokeRecorder($('guard-enroll-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
  },
  onSubmit: () => handleGuardEnrollSubmit(),
});

// ================= SNAPSHOT HELPERS =================
function snapshotArenaEntries(candidates) {
  return candidates.map((c) => ({
    id: c.id,
    name: c.name,
    color: safeColor(c.color, '#2563eb'),
    isSample: Boolean(c.isSample),
    centroid: deepClone(c.template.centroid),
    exemplars: deepClone(c.template.exemplars || []),
    template: deepClone(c.template),
  }));
}

function snapshotGuardEntries(accounts) {
  return accounts.map((acc) => ({
    id: acc.id,
    name: acc.name,
    color: safeColor(acc.color, '#2563eb'),
    isSample: Boolean(acc.isSample),
    centroid: deepClone(acc.password.centroid),
    exemplars: deepClone(acc.password.exemplars || []),
    template: deepClone(acc.password),
  }));
}

export function getActiveVerdictForViz() {
  const v = state.lastVerdict;
  const q = state.viz.lastQuery;
  if (!v || !q) return null;
  if (v.queryId !== q.queryId) return null;
  if (v.roomCode !== roomSync.roomCode) return null;
  if (v.modelHash !== engine.modelHash || v.preprocessVersion !== engine.preprocessVersion) return null;
  const activeDomain = state.viz.mapDomain === 'guard' ? 'password' : 'arena';
  if (v.domain !== activeDomain) return null;
  if (v.domain === 'arena' && v.lang !== state.lang) return null;

  const liveIds = new Set(state.profiles.map((p) => p.id));
  if (v.domain === 'password') {
    if (!v.targetProfileId || !liveIds.has(v.targetProfileId)) return null;
    const liveProf = state.profiles.find((p) => p.id === v.targetProfileId);
    if (!liveProf?.password?.centroid) return null;
  } else {
    const reqIds = v.requiredProfileIds || [];
    if (reqIds.length === 0 || reqIds.some((id) => !liveIds.has(id))) return null;
  }
  return v;
}

// ================= ACTION HANDLERS =================
async function handleArenaEnrollSubmit() {
  if (!state.ready) {
    setFeedback($('arena-enroll-feedback'), notReadyMessage(), 'error');
    return;
  }
  const name = normalizeProfileName($('arena-name-input').value);
  if (!name) {
    setFeedback($('arena-enroll-feedback'), '참가자 이름을 먼저 입력해 주세요.', 'error');
    $('arena-name-input').focus();
    return;
  }
  if (name.length > 24) {
    setFeedback($('arena-enroll-feedback'), '참가자 이름은 24자 이하로 입력해 주세요.', 'error');
    $('arena-name-input').focus();
    return;
  }
  if (isSampleName(name)) {
    setFeedback($('arena-enroll-feedback'), '샘플 프로필 이름은 쓸 수 없습니다. 다른 이름을 입력해 주세요.', 'error');
    $('arena-name-input').focus();
    return;
  }

  if (state.arenaEnrollBuffer.name !== name || state.arenaEnrollBuffer.lang !== state.lang) {
    state.arenaEnrollBuffer = { name, lang: state.lang, items: [] };
  }

  const capture = arenaEnrollRecorder.extractCleanStroke();
  const check = validateSentenceCapture(capture, state.arenaEnrollSentence);
  if (!check.ok) {
    setFeedback($('arena-enroll-feedback'), check.message, 'error');
    if (check.code === 'PASTE_BLOCKED') {
      $('arena-enroll-input').value = '';
      arenaEnrollRecorder.reset();
    }
    return;
  }

  const { embedding } = await engine.embedStroke(capture.stroke);
  const queryId = ++state.querySeq;
  state.lastVerdict = null;
  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: `내 실시간 타건 · '${name}' 프로필 등록 문장`,
  };
  state.viz.lastQuery = {
    queryId,
    embedding,
    embeddings: { sentence: embedding },
    label: `${name} 방금 입력`,
    domain: 'arena',
  };
  state.arenaEnrollBuffer.items.push({
    embedding,
    cpm: capture.cpm,
    holdMs: capture.holdMs,
    rolloverPct: capture.rolloverPct,
    backspaceCount: capture.backspaceCount,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
  });

  const done = state.arenaEnrollBuffer.items.length;
  const existing = roomSync.findProfileByName(name);

  if (existing && existing.arena?.[state.lang]) {
    const tmpl = existing.arena[state.lang];
    const preCandidates = getArenaCandidates();
    const preSnapshot = snapshotArenaEntries(preCandidates);
    const item = state.arenaEnrollBuffer.items[state.arenaEnrollBuffer.items.length - 1];
    state.arenaEnrollBuffer.items = [];
    const mode = capture.inputMode === 'touch' ? 'touch' : 'keyboard';
    const verdict = engine.scoreTemplate(item.embedding, tmpl, {
      domain: 'sentence',
      behavior: capture.behavior,
      inputMode: mode,
      threshold: engine.arenaAddThreshold(mode),
    });

    state.lastVerdict = {
      queryId,
      roomCode: roomSync.roomCode,
      lang: state.lang,
      domain: 'arena',
      targetProfileId: existing.id,
      requiredProfileIds: [existing.id],
      modelHash: engine.modelHash,
      preprocessVersion: engine.preprocessVersion,
      accepted: verdict.accepted,
      rejectedReason: verdict.accepted ? null : 'below_threshold',
      touchReference: false,
      similarity: verdict.similarity,
      threshold: verdict.threshold,
      margin: 0,
      inputMode: mode,
      method: verdict.method,
      adaptedAfterVerdict: false,
      evaluatedTemplatesSnapshot: preSnapshot,
      candidates: [
        {
          id: existing.id,
          name: existing.name,
          color: safeColor(existing.color, '#2563eb'),
          isSample: false,
          similarity: verdict.similarity,
          threshold: verdict.threshold,
          accepted: verdict.accepted,
        },
      ],
    };

    dealArenaEnrollSentence();
    const pct = (v) => `${Math.round(v * 100)}%`;
    if (!verdict.accepted) {
      setFeedback(
        $('arena-enroll-feedback'),
        `'${name}'님의 등록 리듬과 달라 이 문장은 반영하지 않았습니다 (일치도 ${pct(verdict.similarity)}, 기준 ${pct(verdict.threshold)}). 본인이 맞다면 다음 문장을 이어서 쳐 보세요.`,
        'error'
      );
      renderEnrollMode();
      return;
    }
    const mergedTmpl = mergeTemplateSample(tmpl, item);
    roomSync.upsertProfile({
      ...existing,
      arena: {
        ...(existing.arena || {}),
        [state.lang]: mergedTmpl,
      },
    });
    state.lastVerdict.adaptedAfterVerdict = true;
    setFeedback(
      $('arena-enroll-feedback'),
      `'${name}' 프로필에 반영했습니다 (본인 확인 일치도 ${pct(verdict.similarity)} · 총 ${mergedTmpl.count}문장, 최근 12문장 사용).`,
      'success'
    );
    renderEnrollMode();
    return;
  }

  if (done < TARGET_ARENA_SHOTS) {
    renderStepDots($('arena-step-dots'), $('arena-step-text'), done, TARGET_ARENA_SHOTS, '문장');
    dealArenaEnrollSentence();
    const bsNote =
      capture.backspaceCount > 0 ? ` (백스페이스 ${capture.backspaceCount}회 자동 보정)` : '';
    setFeedback(
      $('arena-enroll-feedback'),
      `${done}/${TARGET_ARENA_SHOTS} 완료${bsNote} — 다음 문장을 이어서 타이핑해 주세요.`,
      'success'
    );
    return;
  }

  let tmpl = existing?.arena?.[state.lang] || null;
  for (const item of state.arenaEnrollBuffer.items) {
    tmpl = mergeTemplateSample(tmpl, item);
  }

  const profileObj = existing
    ? {
        ...existing,
        name,
        arena: { ...(existing.arena || {}), [state.lang]: tmpl },
      }
    : {
        id: `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
        name,
        color: pickProfileColor(name),
        isSample: false,
        password: null,
        arena: { [state.lang]: tmpl },
      };

  roomSync.upsertProfile(profileObj);

  $('arena-name-input').value = '';
  resetArenaEnrollProgress();
  dealArenaEnrollSentence();
  setFeedback(
    $('arena-enroll-feedback'),
    `'${name}' 프로필이 실시간 등록되었습니다! 다음 참가자를 등록하거나 [2. 누가 쳤을까?] 탭에서 판별해 보세요.`,
    'success'
  );
}

async function handleArenaBlindSubmit() {
  if (!state.ready) {
    setFeedback($('arena-blind-feedback'), notReadyMessage(), 'error');
    return;
  }
  const candidates = getArenaCandidates();
  if (candidates.length === 0) {
    setFeedback(
      $('arena-blind-feedback'),
      '등록된 참가자가 없습니다. [1. 프로필 등록]에서 먼저 연습하거나 우측 상단에서 샘플 프로필을 추가해 보세요.',
      'error'
    );
    return;
  }

  const capture = arenaBlindRecorder.extractCleanStroke();
  const check = validateSentenceCapture(capture, state.arenaBlindSentence);
  if (!check.ok) {
    state.lastVerdict = null;
    setFeedback($('arena-blind-feedback'), check.message, 'error');
    if (check.code === 'PASTE_BLOCKED') {
      $('arena-blind-input').value = '';
      arenaBlindRecorder.reset();
    }
    return;
  }

  const { embedding: oneEmbedding, latencyMs } = await engine.embedStroke(capture.stroke);
  state.arenaBlindBuffer.push({ embedding: oneEmbedding, capture, accuracy: check.accuracy });
  const got = state.arenaBlindBuffer.length;
  if (got < TARGET_BLIND_SHOTS) {
    renderStepDots($('arena-blind-step-dots'), $('arena-blind-step-text'), got, TARGET_BLIND_SHOTS, '문장');
    setFeedback(
      $('arena-blind-feedback'),
      `${got}/${TARGET_BLIND_SHOTS} 완료 — 한 문장 더 이어서 타이핑하면 판별합니다.`,
      'success'
    );
    dealArenaBlindSentence();
    return;
  }

  setFeedback($('arena-blind-feedback'), '');
  const items = state.arenaBlindBuffer;
  const embedding = combineEmbeddings(items.map((it) => it.embedding));
  let behavior = null;
  for (const it of items) behavior = mergeBehavior(behavior, it.capture.behavior || null);
  const modes = [...new Set(items.map((it) => it.capture.inputMode))];
  const inputMode = modes.length === 1 ? modes[0] : 'mixed';
  const effectiveMode = inputMode === 'mixed' ? 'touch' : inputMode;
  resetArenaBlindProgress();

  const evaluatedSnapshot = snapshotArenaEntries(candidates);
  const res = engine.identifyTypist(embedding, candidates, {
    behavior,
    inputMode: effectiveMode,
  });

  const queryId = ++state.querySeq;
  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: '내 실시간 타건 · 누가 쳤을까? 판별 문장',
  };
  state.viz.mapDomain = 'arena';
  state.viz.selectedProfileId = res.winner?.id || state.viz.selectedProfileId;
  state.viz.lastQuery = {
    queryId,
    embedding,
    embeddings: { sentence: embedding },
    label: '방금 판별 타건',
    domain: 'arena',
  };
  state.lastVerdict = {
    queryId,
    roomCode: roomSync.roomCode,
    lang: state.lang,
    domain: 'arena',
    targetProfileId: res.winner?.id || null,
    requiredProfileIds: candidates.map((c) => c.id),
    modelHash: engine.modelHash,
    preprocessVersion: engine.preprocessVersion,
    accepted: Boolean(res.identified),
    rejectedReason: res.identified ? null : res.rejectedByMargin ? 'margin_hold' : 'below_threshold',
    touchReference: false,
    similarity: res.winner.similarity,
    threshold: res.openSetThreshold,
    margin: res.margin,
    lead: res.lead,
    inputMode: effectiveMode,
    method: 'as_norm',
    adaptedAfterVerdict: false,
    evaluatedTemplatesSnapshot: evaluatedSnapshot,
    candidates: res.rankings.map(({ template, ...rest }) => deepClone(rest)),
  };

  const mean = (k) => Math.round(items.reduce((a, it) => a + (it.capture[k] || 0), 0) / items.length);
  renderArenaResult(res, {
    cpm: mean('cpm'),
    accuracy: Math.round(items.reduce((a, it) => a + it.accuracy, 0) / items.length),
    backspaceCount: items.reduce((a, it) => a + (it.capture.backspaceCount || 0), 0),
    cleanDigraphs: items.reduce((a, it) => a + (it.capture.cleanDigraphCount || 0), 0),
    latencyMs,
    inputMode,
    sentences: items.length,
  });

  dealArenaBlindSentence();
}

function renderArenaResult(res, stats) {
  const card = $('arena-result-card');
  card.hidden = false;

  const pct = Math.round((Number(res.winner?.similarity) || 0) * 1000) / 10;
  $('arena-result-score').textContent = `${pct.toFixed(1)}%`;

  if (res.identified) {
    card.className = 'result-card is-pass';
    $('arena-result-kicker').textContent = `후보 ${res.rankings.length}명 중 1위 일치`;
    $('arena-result-title').textContent = `지금 타이핑한 사람은 '${res.winner.name}'님입니다`;
  } else if (res.rejectedByMargin) {
    card.className = 'result-card is-block';
    $('arena-result-kicker').textContent = '판별 보류 (1·2위 점수 차 부족)';
    $('arena-result-title').textContent = `'${res.winner.name}'님과 '${res.rankings[1].name}'님의 리듬이 너무 비슷해 판별하지 않았습니다`;
  } else {
    card.className = 'result-card is-block';
    $('arena-result-kicker').textContent = '미등록 외부인 기각 (Open-Set Rejection)';
    $('arena-result-title').textContent = '등록되지 않은 새로운 사용자의 타이핑입니다';
  }

  const topList = res.rankings.slice(0, 5);
  $('arena-ranking-bars').innerHTML = topList
    .map((r, idx) => {
      const simNum = Number(r?.similarity) || 0;
      const sPct = Math.max(4, Math.min(100, safeInt(simNum * 100, 4)));
      const col = safeColor(r?.color, '#059669');
      const fillStyle =
        idx === 0 && res.identified
          ? `width:${sPct}%;background:${col}`
          : `width:${sPct}%;background:#9ca3af`;
      return `<div class="rank-item">
        <span class="rank-name">${safeInt(idx + 1)}. ${escapeHtml(r?.name)}</span>
        <div class="rank-track"><div class="rank-fill" style="${fillStyle}"></div></div>
        <span class="rank-val">${(simNum * 100).toFixed(1)}%</span>
      </div>`;
    })
    .join('');

  const editTxt =
    stats.backspaceCount > 0 ? `백스페이스 ${stats.backspaceCount}회 자동 보정` : '무수정 타이핑';
  const modeTxt =
    stats.inputMode && stats.inputMode !== 'keyboard'
      ? ` · ${INPUT_MODE_LABEL[stats.inputMode] || stats.inputMode}(누름 시간 대체)`
      : '';
  const sentTxt = stats.sentences > 1 ? `문장 ${stats.sentences}개 · ` : '';
  $('arena-result-meta').textContent = `${sentTxt}${stats.cpm} 타/분 · 정확도 ${stats.accuracy}% · ${editTxt} (유효 전이 ${stats.cleanDigraphs}구간)${modeTxt}${behaviorNote(res.winner)}${deviceMismatchNote(res.winner.template, stats.inputMode)} · 추론 ${stats.latencyMs}ms`;
}

async function handleGuardEnrollSubmit() {
  if (!state.ready) {
    setFeedback($('guard-enroll-feedback'), notReadyMessage(), 'error');
    return;
  }
  const name = normalizeProfileName($('guard-enroll-name').value);
  const targetPw = $('guard-enroll-pw-target').value;

  if (!name) {
    setFeedback($('guard-enroll-feedback'), '사용자 이름을 먼저 입력해 주세요.', 'error');
    $('guard-enroll-name').focus();
    return;
  }
  if (name.length > 24) {
    setFeedback($('guard-enroll-feedback'), '사용자 이름은 24자 이하로 입력해 주세요.', 'error');
    $('guard-enroll-name').focus();
    return;
  }
  if (isSampleName(name)) {
    setFeedback($('guard-enroll-feedback'), '샘플 프로필 이름은 쓸 수 없습니다. 다른 이름을 입력해 주세요.', 'error');
    $('guard-enroll-name').focus();
    return;
  }

  // Block unauthenticated password overwrite if this name already has a password template!
  const existingBefore = roomSync.findProfileByName(name);
  if (existingBefore?.password) {
    state.pwEnrollBuffer = { name: '', targetPw: '', items: [] };
    renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
    $('guard-enroll-input').value = '';
    guardEnrollRecorder.reset();
    setFeedback(
      $('guard-enroll-feedback'),
      '이미 비밀번호가 등록된 이름입니다. 다른 이름을 사용하거나 프로필 관리에서 기존 계정을 먼저 삭제해 주세요.',
      'error'
    );
    $('guard-enroll-name').focus();
    return;
  }

  if (
    state.pwEnrollBuffer.name !== name ||
    state.pwEnrollBuffer.targetPw !== targetPw
  ) {
    state.pwEnrollBuffer = { name, targetPw, items: [] };
  }

  const capture = guardEnrollRecorder.extractCleanStroke();
  const check = validatePasswordCapture(capture, targetPw, { minLength: engine.config?.password_min_length ?? 12 });
  if (!check.ok) {
    setFeedback($('guard-enroll-feedback'), check.message, 'error');
    $('guard-enroll-input').value = '';
    guardEnrollRecorder.reset();
    return;
  }

  const both = await engine.embedAll(capture.stroke);
  state.pwEnrollBuffer.items.push({
    embedding: both.password,
    embedding17: both.sentence,
    capitalMethods: capture.capitalMethods,
    cpm: capture.cpm,
    holdMs: capture.holdMs,
    rolloverPct: capture.rolloverPct,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
  });

  $('guard-enroll-input').value = '';
  guardEnrollRecorder.reset();

  const done = state.pwEnrollBuffer.items.length;
  if (done < TARGET_PW_SHOTS) {
    renderStepDots($('guard-step-dots'), $('guard-step-text'), done, TARGET_PW_SHOTS, '리듬 등록');
    const bsNote =
      capture.backspaceCount > 0 ? ` (백스페이스 ${capture.backspaceCount}회 보정됨)` : '';
    setFeedback(
      $('guard-enroll-feedback'),
      `${done}/${TARGET_PW_SHOTS}회 입력 완료${bsNote} — 평소 리듬대로 한 번 더 입력해 주세요.`,
      'success'
    );
    return;
  }

  // Re-check right before saving in case another tab/peer enrolled the same name concurrently
  const existing = roomSync.findProfileByName(name);
  if (existing?.password) {
    state.pwEnrollBuffer = { name: '', targetPw: '', items: [] };
    renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
    setFeedback(
      $('guard-enroll-feedback'),
      '이미 비밀번호가 등록된 이름입니다. 다른 이름을 사용하거나 프로필 관리에서 기존 계정을 먼저 삭제해 주세요.',
      'error'
    );
    $('guard-enroll-name').focus();
    return;
  }

  let pwTmpl = null;
  for (const item of state.pwEnrollBuffer.items) {
    pwTmpl = mergeTemplateSample(pwTmpl, item);
  }
  pwTmpl.target = targetPw;

  const saved = roomSync.upsertProfile(
    existing
      ? { ...existing, name, password: pwTmpl }
      : {
          id: `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
          name,
          color: pickProfileColor(name),
          isSample: false,
          password: pwTmpl,
          arena: {},
        }
  );

  if (!saved) {
    setFeedback($('guard-enroll-feedback'), '프로필 스키마 검증에 실패했습니다. 입력 값을 확인해 주세요.', 'error');
    return;
  }

  state.pwEnrollBuffer = { name: '', targetPw: '', items: [] };
  $('guard-enroll-name').value = '';
  renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
  setFeedback($('guard-enroll-feedback'), '');

  state.selectedGuardId = saved.id;
  switchGuardSub('verify');
  setFeedback(
    $('guard-verify-feedback'),
    `'${name}' 계정의 타이핑 지문이 등록되었습니다! 본인과 다른 사람이 번갈아 입력해 비교해 보세요.`,
    'success'
  );
}

async function handleGuardVerifySubmit() {
  if (!state.ready) {
    setFeedback($('guard-verify-feedback'), notReadyMessage(), 'error');
    return;
  }
  const accounts = getGuardAccounts();
  if (accounts.length === 0) return;

  const targetAcc = accounts.find((a) => a.id === state.selectedGuardId) || accounts[0];
  const expectedPw = targetAcc.password.target;

  const capture = guardVerifyRecorder.extractCleanStroke();
  const check = validatePasswordCapture(capture, expectedPw, { minLength: 1 });

  $('guard-verify-input').value = '';
  guardVerifyRecorder.reset();

  if (!check.ok) {
    state.lastVerdict = null;
    setFeedback($('guard-verify-feedback'), check.message, 'error');
    return;
  }

  setFeedback($('guard-verify-feedback'), '');
  const both = await engine.embedAll(capture.stroke);
  const embedding = both.password;
  const embedding17 = both.sentence;
  const latencyMs = both.latencyMs;

  const scoreOpts = {
    domain: 'password',
    thresholdOffset: state.thresholdOffset,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
    embedding17,
    capitalMethods: capture.capitalMethods,
  };

  // Capture exact pre-adaptation templates BEFORE any automatic mergeTemplateSample!
  const preAdaptEntries = snapshotGuardEntries(accounts);
  const res = engine.scoreTemplate(embedding, targetAcc.password, scoreOpts);
  const threshold = res.threshold;
  const touchReference = capture.inputMode !== 'keyboard' && engine.passwordTouchReferenceOnly();

  let closerOther = null;
  const candidateScores = [
    {
      id: targetAcc.id,
      name: targetAcc.name,
      color: safeColor(targetAcc.color, '#2563eb'),
      isSample: Boolean(targetAcc.isSample),
      similarity: res.similarity,
      threshold: res.threshold,
      accepted: res.accepted,
    },
  ];
  if (accounts.length > 1) {
    const others = accounts
      .filter((a) => a.id !== targetAcc.id)
      .map((a) => {
        const s = engine.scoreTemplate(embedding, a.password, scoreOpts);
        candidateScores.push({
          id: a.id,
          name: a.name,
          color: safeColor(a.color, '#2563eb'),
          isSample: Boolean(a.isSample),
          similarity: s.similarity,
          threshold: s.threshold,
          accepted: s.accepted,
        });
        return { name: a.name, sim: s.similarity };
      })
      .sort((a, b) => b.sim - a.sim);
    if (!res.accepted && others[0] && others[0].sim > res.similarity + 0.08 && others[0].sim >= 0.45) {
      closerOther = others[0];
    }
  }

  const queryId = ++state.querySeq;
  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: `내 실시간 타건 · '${targetAcc.name}' 암호 검증`,
  };
  state.viz.mapDomain = 'guard';
  state.viz.selectedProfileId = targetAcc.id;
  state.viz.lastQuery = {
    queryId,
    embedding,
    embeddings: { password: embedding, sentence: embedding17 },
    label: '방금 입력한 암호',
    domain: 'password',
  };
  state.lastVerdict = {
    queryId,
    roomCode: roomSync.roomCode,
    lang: state.lang,
    domain: 'password',
    targetProfileId: targetAcc.id,
    requiredProfileIds: [targetAcc.id],
    modelHash: engine.modelHash,
    preprocessVersion: engine.preprocessVersion,
    accepted: Boolean(res.accepted),
    rejectedReason: res.accepted
      ? touchReference
        ? 'touch_reference_only'
        : null
      : touchReference
        ? 'below_threshold_touch_reference'
        : 'below_threshold',
    touchReference,
    similarity: res.similarity,
    threshold,
    margin: 0,
    inputMode: capture.inputMode,
    method: res.method,
    adaptedAfterVerdict: false,
    evaluatedTemplatesSnapshot: preAdaptEntries,
    candidates: candidateScores,
  };

  let adapted = false;
  if (res.adapt && !targetAcc.isSample && capture.inputMode === (targetAcc.password.inputMode || 'keyboard')) {
    const merged = mergeTemplateSample(targetAcc.password, {
      embedding,
      embedding17,
      cpm: capture.cpm,
      holdMs: capture.holdMs,
      rolloverPct: capture.rolloverPct,
      behavior: capture.behavior,
      inputMode: capture.inputMode,
      capitalMethods: capture.capitalMethods,
    });
    const updatedCount = merged.count;
    roomSync.upsertProfile({ ...targetAcc, password: { ...merged, target: targetAcc.password.target } });
    adapted = true;
    state.lastVerdict.adaptedAfterVerdict = true;
    state.lastVerdict.adaptedCount = updatedCount;
  }

  const card = $('guard-result-card');
  card.hidden = false;
  const pct = (res.similarity * 100).toFixed(1);
  $('guard-result-score').textContent = `${pct}%`;

  if (res.accepted) {
    card.className = 'result-card is-pass';
    $('guard-result-kicker').textContent = 'ACCESS GRANTED · 본인 인증 승인';
    $('guard-result-title').textContent = `${targetAcc.name}님의 타이핑 리듬과 일치합니다`;
    $('guard-result-desc').textContent = `비밀번호 문자열과 손가락 전환 리듬이 모두 기준치(${Math.round(
      threshold * 100
    )}%)를 통과했습니다.`;
  } else {
    card.className = 'result-card is-block';
    $('guard-result-kicker').textContent = 'INTRUSION BLOCKED · 침입 차단';
    $('guard-result-title').textContent = '비밀번호는 맞지만 타이핑 지문이 불일치합니다';
    const hint = closerOther
      ? ` 입력된 리듬은 등록된 '${closerOther.name}'님의 타건 패턴(${(closerOther.sim * 100).toFixed(
          0
        )}%)에 더 가깝습니다.`
      : '';
    $('guard-result-desc').textContent = `유출된 비밀번호를 알고 입력했으나 ${
      targetAcc.name
    }님의 고유 홀드·전환 리듬과 달라 접속을 차단했습니다.${hint}`;
  }
  card.dataset.reference = touchReference ? 'true' : 'false';
  if (touchReference) {
    $('guard-result-kicker').textContent += ' · 모바일 참고용';
    $('guard-result-desc').textContent +=
      ' 모바일 터치 입력은 누름 시간을 알 수 없어 비밀번호 길이의 입력만으로는 다른 사람을 충분히 구분하지 못합니다. 이 결과는 참고용으로만 보세요.';
  }

  const editTxt =
    capture.backspaceCount > 0 ? `백스페이스 ${capture.backspaceCount}회 보정` : '무수정 입력';
  const holdTxt =
    capture.inputMode === 'touch'
      ? `${INPUT_MODE_LABEL.touch}(누름 시간 대체)`
      : `홀드 ${capture.holdMs}ms · 롤오버 ${capture.rolloverPct}%`;
  const adaptTxt = adapted
    ? ` · 등록 리듬 갱신(최근 ${Math.min(12, (targetAcc.password.count || 0) + 1)}회 반영)`
    : '';
  $('guard-result-meta').textContent = `${capture.cpm} 타/분 · ${holdTxt} · ${editTxt}${behaviorNote(res)}${deviceMismatchNote(targetAcc.password, capture.inputMode)}${adaptTxt} · 추론 ${latencyMs}ms`;
}

// ================= MODE C: VISUAL EXPLAINER & PROFILE FORMATION =================
let vizEmbedTimer = null;

const vizSandboxRecorder = new SmartKeystrokeRecorder($('viz-sandbox-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
    if (vizSandboxRecorder.stack.length < 2) {
      if (state.mode === 'principles') renderVizStep3();
      return;
    }
    const cap = vizSandboxRecorder.extractCleanStroke({ resolvePending: false });
    state.viz.capture = {
      ...cap,
      isPreset: false,
      sourceLabel: '내 실시간 타건 (입력 즉시 갱신 중)',
    };
    renderVizStep1();

    if (vizEmbedTimer) clearTimeout(vizEmbedTimer);
    if (state.ready && cap.cleanDigraphCount >= 4) {
      const scheduledSeq = state.querySeq;
      vizEmbedTimer = setTimeout(async () => {
        if (state.mode !== 'principles' || state.querySeq !== scheduledSeq) return;
        try {
          const embeddings = await engine.embedAll(cap.stroke);
          if (state.mode !== 'principles' || state.querySeq !== scheduledSeq) return;
          const queryId = ++state.querySeq;
          state.lastVerdict = null;
          state.viz.lastQuery = {
            queryId,
            embedding: embeddings.sentence,
            embeddings,
            label: '방금 입력한 내 타건',
            domain: state.viz.mapDomain === 'guard' ? 'password' : 'arena',
          };
          renderVizStep3();
        } catch {
          // ignore transient short-stroke inference errors
        }
      }, 150);
    } else if (state.mode === 'principles') {
      renderVizStep3();
    }
  },
  onSubmit: async () => {
    if (!state.ready) {
      $('viz-signal-badge').textContent = notReadyMessage();
      return;
    }
    const cap = vizSandboxRecorder.extractCleanStroke();
    if (cap.cleanDigraphCount < 2) return;
    state.viz.capture = {
      ...cap,
      isPreset: false,
      sourceLabel: '내 실시간 타건 분석 완료',
    };
    renderVizStep1();
    if (cap.cleanDigraphCount >= 4) {
      const embeddings = await engine.embedAll(cap.stroke);
      const queryId = ++state.querySeq;
      state.lastVerdict = null;
      state.viz.lastQuery = {
        queryId,
        embedding: embeddings.sentence,
        embeddings,
        label: '방금 입력한 내 타건',
        domain: state.viz.mapDomain === 'guard' ? 'password' : 'arena',
      };
      renderVizStep3();
    }
  },
});

function getVizProfilePool() {
  const realList = (state.profiles || []).filter((p) => p && typeof p.name === 'string');
  const sampleFallback = engine.config?.mvp_sample_profiles || [];
  if (realList.length >= 2) {
    return { list: realList, usingFallback: false };
  }
  const existingIds = new Set(realList.map((p) => p.id));
  const merged = [...realList, ...sampleFallback.filter((s) => s && !existingIds.has(s.id))];
  return {
    list: merged,
    usingFallback: merged.length > realList.length,
  };
}

function pickTemplateForViz(profile, preferDomain = 'arena') {
  if (!profile) return { tmpl: null, unitLabel: '문장' };
  if (preferDomain === 'guard' && profile.password?.centroid) {
    return { tmpl: profile.password, unitLabel: '암호' };
  }
  const arenaTmpl = getArenaTemplateForProfile(profile, state.lang);
  if (arenaTmpl?.centroid) {
    return { tmpl: arenaTmpl, unitLabel: '문장' };
  }
  if (profile.password?.centroid) {
    return { tmpl: profile.password, unitLabel: '암호' };
  }
  return { tmpl: null, unitLabel: '샘플' };
}

function renderVizStep1() {
  const cap = state.viz.capture;
  const badge = $('viz-signal-badge');
  badge.textContent = cap.sourceLabel || '예시 파형';
  badge.classList.toggle('is-live', !cap.isPreset);

  $('viz-preset-fast-btn').classList.toggle(
    'is-selected-ghost',
    Boolean(cap.isPreset && cap.presetMode === 'normal')
  );
  $('viz-preset-slow-btn').classList.toggle(
    'is-selected-ghost',
    Boolean(cap.isPreset && cap.presetMode === 'slow')
  );

  $('viz-wave-tab-piano').classList.toggle('is-active', state.viz.waveTab === 'piano');
  $('viz-wave-tab-tempo').classList.toggle('is-active', state.viz.waveTab === 'tempo');

  const step1Wrap = $('viz-step1-svg');
  step1Wrap.innerHTML =
    state.viz.waveTab === 'piano'
      ? renderPianoRollSvg(cap)
      : renderTempoInvarianceSvg(cap);
  if (state.viz.waveTab === 'piano' && !cap.isPreset) {
    step1Wrap.scrollTop = step1Wrap.scrollHeight;
  }

  const bsNote =
    cap.backspaceCount > 0 ? `백스페이스 ${safeInt(cap.backspaceCount, 0)}회 봉합` : '무수정';
  $('viz-step1-stats').textContent = `${safeInt(cap.cpm, 0)} 타/분 · 평균 홀드 ${safeInt(
    cap.holdMs,
    0
  )}ms · 롤오버 ${safeInt(cap.rolloverPct, 0)}% · ${bsNote} · 전이 ${safeInt(cap.cleanDigraphCount, 0)}구간`;
}

function renderVizStep2() {
  const { list, usingFallback } = getVizProfilePool();
  $('viz-profile-source-hint').textContent = usingFallback
    ? '실측 샘플 프리뷰 포함 (참가자 등록 시 자동 반영)'
    : `현재 방 참가자 ${list.length}명 실시간 데이터`;

  if (list.length === 0) {
    $('viz-profile-chips').innerHTML = `<span class="roster-empty">모델 프로필 준비 중…</span>`;
    $('viz-step2-svg').innerHTML = `<div class="viz-empty">프로필 데이터를 불러오는 중입니다.</div>`;
    return;
  }

  if (!list.some((p) => p.id === state.viz.selectedProfileId)) {
    const rich =
      list.find((p) => (pickTemplateForViz(p, state.viz.mapDomain).tmpl?.exemplars?.length || 0) >= 3) ||
      list[0];
    state.viz.selectedProfileId = rich.id;
  }
  const selected = list.find((p) => p.id === state.viz.selectedProfileId) || list[0];
  const { tmpl, unitLabel } = pickTemplateForViz(selected, state.viz.mapDomain);

  $('viz-profile-chips').innerHTML = list
    .map((p) => {
      const safeName = String(p.name ?? '');
      const initial = escapeHtml(safeName.trim().charAt(0) || '?');
      const col = safeColor(p.color, '#2563eb');
      const active = p.id === selected.id;
      return `<button type="button" class="user-chip ${active ? 'is-selected' : ''}" data-viz-prof="${escapeHtml(
        p.id
      )}">
        <span class="chip-avatar" style="background:${col}">${initial}</span>
        <span>${escapeHtml(safeName)}</span>
      </button>`;
    })
    .join('');

  $('viz-profile-chips').querySelectorAll('[data-viz-prof]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.viz.selectedProfileId = btn.getAttribute('data-viz-prof');
      state.viz.shotCount = 99;
      renderVizStep2();
    });
  });

  const totalEx = Math.max(1, tmpl?.exemplars?.length || 1);
  const clampedShots = Math.max(1, Math.min(state.viz.shotCount, totalEx));

  const stepOptions = [];
  for (let k = 1; k <= Math.min(4, totalEx); k++) {
    const isLast = k === Math.min(4, totalEx);
    const actualK = safeInt(isLast ? totalEx : k, 1);
    const label = isLast ? `${actualK}${unitLabel} 완성` : `${k}${unitLabel}`;
    const isAct = isLast ? clampedShots >= k : clampedShots === k;
    stepOptions.push(
      `<button type="button" class="lang-btn ${isAct ? 'is-active' : ''}" data-viz-shots="${actualK}">${escapeHtml(
        label
      )}</button>`
    );
  }
  $('viz-shot-scrubber').innerHTML = stepOptions.join('');
  $('viz-shot-scrubber').querySelectorAll('[data-viz-shots]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.viz.shotCount = safeInt(btn.getAttribute('data-viz-shots'), 1);
      renderVizStep2();
    });
  });

  const { svg, info } = renderProfileBarcodeSvg(tmpl, {
    color: safeColor(selected.color, '#2563eb'),
    stepCount: clampedShots,
    unitLabel,
  });
  $('viz-step2-svg').innerHTML = svg;

  if (info.status === 'ok') {
    const signedCos = `${info.meanPairwiseCos >= 0 ? '+' : ''}${info.meanPairwiseCos.toFixed(2)}`;
    $('viz-stat-stability').textContent = `평균 쌍별 코사인 ${signedCos}`;
    $('viz-stat-stability-sub').textContent = `일관성 ${info.stabilityPct.toFixed(1)}% · ${info.activeCount}개 누적`;
    $('viz-stat-coherence').textContent = `${info.stabilityPct.toFixed(1)}%`;
  } else {
    $('viz-stat-stability').textContent = info.stabilityText;
    $('viz-stat-stability-sub').textContent =
      info.status === 'invalid'
        ? '영벡터 또는 비유한값 감지됨'
        : '단일 샘플 — 쌍별 유사도 계산에 2개 이상 필요';
    $('viz-stat-coherence').textContent = info.stabilityText;
  }
  $('viz-stat-hold').textContent = `${safeInt(tmpl?.holdMs, 0)} ms`;
  $('viz-stat-rollover').textContent = `${safeInt(tmpl?.rolloverPct, 0)}% (${safeInt(tmpl?.cpm, 0)}타/분)`;
}

function formatVerdictBannerText(v, winnerName) {
  const simPct = (v.similarity * 100).toFixed(1);
  const thrPct = (v.threshold * 100).toFixed(1);
  if (v.domain === 'password') {
    const statusLabel = v.accepted ? '본인 인증 승인' : '침입 차단 (리듬 불일치)';
    const refTag = v.touchReference ? ' · 모바일 참고용' : '';
    return `공식 판정 스냅샷 · [${winnerName}] ${statusLabel}${refTag} — 일치도 ${simPct}% (임계값 ${thrPct}%)`;
  }
  if (v.accepted) {
    return `공식 판정 스냅샷 · '${winnerName}' 식별 통과 — 일치도 ${simPct}% (임계값 ${thrPct}%)`;
  }
  if (v.rejectedReason === 'margin_hold') {
    const leadPct = Number.isFinite(v.lead) ? `${(v.lead * 100).toFixed(1)}%p` : '부족';
    const marginPct = `${((v.margin || 0) * 100).toFixed(1)}%p`;
    return `공식 판정 스냅샷 · 판별 보류 (1·2위 격차 ${leadPct} < 기준 ${marginPct}) — 최고 점수 ${simPct}% (임계값 ${thrPct}%)`;
  }
  return `공식 판정 스냅샷 · 미등록 외부인 기각 — 최고 점수 ${simPct}% < 임계값 ${thrPct}%`;
}

function renderVizStep3() {
  const isArenaMap = state.viz.mapDomain === 'arena';
  $('viz-map-domain-arena').classList.toggle('is-active', isArenaMap);
  $('viz-map-domain-guard').classList.toggle('is-active', !isArenaMap);

  const bannerEl = $('viz-verdict-banner');
  const statusEl = $('viz-verdict-status');
  const adaptBadgeEl = $('viz-verdict-adapt-badge');

  const activeVerdict = getActiveVerdictForViz();
  const q = state.viz.lastQuery;
  const qEmb = q?.embeddings ? q.embeddings[isArenaMap ? 'sentence' : 'password'] : q?.embedding;

  let entries = [];
  let queryPoint = null;

  if (activeVerdict && qEmb) {
    // Plot the EXACT pre-adaptation templates evaluated at verdict time!
    entries = activeVerdict.evaluatedTemplatesSnapshot;
    const targetEntry =
      entries.find((e) => e.id === activeVerdict.targetProfileId) || entries[0];
    const winnerName = targetEntry?.name || '—';
    const simPct = (activeVerdict.similarity * 100).toFixed(1);
    const thrPct = (activeVerdict.threshold * 100).toFixed(1);
    const bannerText = formatVerdictBannerText(activeVerdict, winnerName);

    if (bannerEl && statusEl && adaptBadgeEl) {
      bannerEl.className = `viz-verdict-banner ${activeVerdict.accepted ? 'is-pass' : 'is-block'}`;
      statusEl.textContent = bannerText;
      adaptBadgeEl.hidden = !activeVerdict.adaptedAfterVerdict;
    }

    queryPoint = {
      embedding: qEmb,
      label: q.label || '방금 판정 타건',
      winnerId: activeVerdict.targetProfileId,
      winnerName,
      similarityPct: simPct,
      identified: activeVerdict.accepted,
      isPreview: false,
      verdictLine: `*공식 판정: ${simPct}% / 기준 ${thrPct}% (${
        activeVerdict.accepted
          ? '승인'
          : activeVerdict.rejectedReason === 'margin_hold'
            ? '판별 보류'
            : '기각/차단'
      })`,
    };
  } else {
    // Exploratory preview (before official verdict or after invalidation by room/lang/input/deletion)
    if (bannerEl && statusEl && adaptBadgeEl) {
      bannerEl.className = 'viz-verdict-banner is-preview';
      statusEl.textContent = '실시간 탐색 미리보기 (공식 인증 판정 전)';
      adaptBadgeEl.hidden = true;
    }

    const { list } = getVizProfilePool();
    entries = list
      .map((p) => {
        const tmpl = isArenaMap
          ? getArenaTemplateForProfile(p, state.lang)
          : p.password || null;
        if (!tmpl?.centroid) return null;
        return {
          id: p.id,
          name: p.name,
          color: safeColor(p.color, '#2563eb'),
          isSample: Boolean(p.isSample),
          centroid: tmpl.centroid,
          exemplars: tmpl.exemplars || [],
          template: tmpl,
        };
      })
      .filter(Boolean);

    if (qEmb && entries.length > 0 && state.ready) {
      if (isArenaMap) {
        const scored = engine.identifyTypist(qEmb, entries, {
          domain: 'sentence',
          openSetThreshold: engine.arenaThreshold('keyboard'),
        });
        if (scored?.winner) {
          queryPoint = {
            embedding: qEmb,
            label: q.label || '탐색 타건 미리보기',
            winnerId: scored.winner.id,
            winnerName: scored.winner.name,
            similarityPct: (scored.winner.similarity * 100).toFixed(1),
            identified: scored.identified,
            isPreview: true,
            verdictLine: `*탐색 유사도 미리보기: ${(scored.winner.similarity * 100).toFixed(1)}% (공식 판정 전)`,
          };
        }
      } else {
        const q17 = q?.embeddings?.sentence || null;
        const ranked = entries
          .map((ent) => {
            const res = engine.scoreTemplate(qEmb, ent.template, {
              domain: 'password',
              thresholdOffset: state.thresholdOffset,
              embedding17: q17,
            });
            return { ent, res };
          })
          .sort((a, b) => b.res.similarity - a.res.similarity);
        if (ranked[0]) {
          queryPoint = {
            embedding: qEmb,
            label: q.label || '탐색 타건 미리보기',
            winnerId: ranked[0].ent.id,
            winnerName: ranked[0].ent.name,
            similarityPct: (ranked[0].res.similarity * 100).toFixed(1),
            identified: ranked[0].res.accepted,
            isPreview: true,
            verdictLine: `*탐색 유사도 미리보기: ${(ranked[0].res.similarity * 100).toFixed(1)}% (공식 판정 전)`,
          };
        }
      }
    }
  }

  $('viz-step3-svg').innerHTML = renderHypersphereMapSvg(entries, queryPoint);
}

function renderPrinciplesView() {
  renderVizStep1();
  renderVizStep2();
  renderVizStep3();
}

async function applyVizPreset(mode) {
  $('viz-sandbox-input').value = '';
  vizSandboxRecorder.reset();
  const cap = createPresetCapture(mode);
  state.viz.capture = cap;
  state.lastVerdict = null;
  renderVizStep1();
  if (state.ready) {
    const embeddings = await engine.embedAll(cap.stroke);
    const queryId = ++state.querySeq;
    state.viz.lastQuery = {
      queryId,
      embedding: embeddings.sentence,
      embeddings,
      label: mode === 'slow' ? '예시 타건 (저속)' : '예시 타건 (고속)',
      domain: state.viz.mapDomain === 'guard' ? 'password' : 'arena',
    };
    renderVizStep3();
  }
}

// ================= EVENT WIRING & INIT =================
function bindEvents() {
  $('nav-arena-btn').addEventListener('click', () => switchMainMode('arena'));
  $('nav-guard-btn').addEventListener('click', () => switchMainMode('guard'));
  $('nav-principles-btn').addEventListener('click', () => switchMainMode('principles'));

  $('arena-inspect-viz-btn').addEventListener('click', () => {
    state.viz.mapDomain = 'arena';
    switchMainMode('principles');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  $('guard-inspect-viz-btn').addEventListener('click', () => {
    state.viz.mapDomain = 'guard';
    switchMainMode('principles');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  $('viz-wave-tab-piano').addEventListener('click', () => {
    state.viz.waveTab = 'piano';
    renderVizStep1();
  });
  $('viz-wave-tab-tempo').addEventListener('click', () => {
    state.viz.waveTab = 'tempo';
    renderVizStep1();
  });
  $('viz-preset-fast-btn').addEventListener('click', () => applyVizPreset('normal'));
  $('viz-preset-slow-btn').addEventListener('click', () => applyVizPreset('slow'));

  $('viz-map-domain-arena').addEventListener('click', () => {
    state.viz.mapDomain = 'arena';
    renderVizStep2();
    renderVizStep3();
  });
  $('viz-map-domain-guard').addEventListener('click', () => {
    state.viz.mapDomain = 'guard';
    renderVizStep2();
    renderVizStep3();
  });

  $('viz-jump-sandbox-btn').addEventListener('click', () => {
    const input = $('viz-sandbox-input');
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => input.focus(), 220);
  });

  // Delegated zero-latency hover tooltip for all SVG charts in #view-principles
  const tipEl = $('viz-tooltip');
  const vizSection = $('view-principles');
  let activeHit = null;

  const hideVizTip = () => {
    activeHit = null;
    if (tipEl) tipEl.hidden = true;
  };

  vizSection.addEventListener('pointermove', (e) => {
    const hit = e.target?.closest?.('[data-tip-title]');
    if (!hit) {
      hideVizTip();
      return;
    }
    if (hit !== activeHit) {
      activeHit = hit;
      const title = hit.getAttribute('data-tip-title') || '';
      const rawLines = (hit.getAttribute('data-tip-lines') || '')
        .split('||')
        .filter(Boolean);
      const lineHtml = rawLines
        .map((ln) => {
          const isStrong = ln.startsWith('*');
          const clean = isStrong ? ln.slice(1) : ln;
          return `<div class="viz-tip-line${isStrong ? ' is-strong' : ''}">${escapeHtml(clean)}</div>`;
        })
        .join('');
      tipEl.innerHTML = `<div class="viz-tip-title">${escapeHtml(title)}</div>${lineHtml}`;
      tipEl.hidden = false;
    }

    const pad = 14;
    const w = tipEl.offsetWidth || 220;
    const h = tipEl.offsetHeight || 80;
    let left = e.clientX + pad;
    let top = e.clientY + pad;
    if (left + w > window.innerWidth - 12) {
      left = Math.max(12, e.clientX - w - pad);
    }
    if (top + h > window.innerHeight - 12) {
      top = Math.max(12, e.clientY - h - pad);
    }
    tipEl.style.transform = `translate3d(${Math.round(left)}px, ${Math.round(top)}px, 0)`;
  });

  vizSection.addEventListener('pointerleave', hideVizTip);
  window.addEventListener('scroll', hideVizTip, { passive: true });

  $('arena-sub-enroll').addEventListener('click', () => switchArenaSub('enroll'));
  $('arena-sub-blind').addEventListener('click', () => switchArenaSub('blind'));

  $('guard-sub-verify').addEventListener('click', () => switchGuardSub('verify'));
  $('guard-sub-enroll').addEventListener('click', () => switchGuardSub('enroll'));

  $('lang-ko-btn').addEventListener('click', () => switchLanguage('ko'));
  $('lang-en-btn').addEventListener('click', () => switchLanguage('en'));

  $('arena-next-sentence-btn').addEventListener('click', () => {
    dealArenaEnrollSentence();
    $('arena-enroll-input').focus();
  });

  $('arena-blind-shuffle-btn').addEventListener('click', () => {
    dealArenaBlindSentence();
    $('arena-blind-input').focus();
  });

  $('arena-retry-btn').addEventListener('click', () => {
    $('arena-result-card').hidden = true;
    dealArenaBlindSentence();
    $('arena-blind-input').focus();
  });

  $('arena-name-input').addEventListener('input', () => {
    const val = normalizeProfileName($('arena-name-input').value);
    if (val !== state.arenaEnrollBuffer.name) {
      resetArenaEnrollProgress();
    }
  });
  $('arena-name-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!state.ready) {
        setFeedback($('arena-enroll-feedback'), notReadyMessage(), 'error');
        return;
      }
      $('arena-enroll-input').focus();
    }
  });

  $('guard-enroll-name').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!state.ready) {
        setFeedback($('guard-enroll-feedback'), notReadyMessage(), 'error');
        return;
      }
      $('guard-enroll-input').focus();
    }
  });

  $('guard-enroll-pw-target').addEventListener('input', () => {
    state.pwEnrollBuffer = {
      name: normalizeProfileName($('guard-enroll-name').value),
      targetPw: $('guard-enroll-pw-target').value,
      items: [],
    };
    renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
  });
  $('guard-enroll-pw-target').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!state.ready) {
        setFeedback($('guard-enroll-feedback'), notReadyMessage(), 'error');
        return;
      }
      $('guard-enroll-input').focus();
    }
  });

  $('guard-toggle-pw-btn').addEventListener('click', () => {
    state.showPasswordPlain = !state.showPasswordPlain;
    renderGuardVerifyState();
  });

  document.querySelectorAll('.tau-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tau-btn').forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.thresholdOffset = Number(btn.getAttribute('data-tau-offset')) || 0;
    });
  });

  $('guard-go-enroll-btn').addEventListener('click', () => switchGuardSub('enroll'));
  $('guard-load-samples-btn').addEventListener('click', () => {
    if (!state.ready) {
      setFeedback($('guard-verify-feedback'), notReadyMessage(), 'error');
      return;
    }
    if (engine.config?.mvp_sample_profiles) {
      roomSync.loadSamples(engine.config.mvp_sample_profiles);
    }
  });

  // Modal Dialog Controls
  const dialog = $('room-dialog');
  const openModal = () => {
    renderModalProfiles();
    dialog.showModal();
  };
  $('open-room-btn').addEventListener('click', openModal);
  $('arena-manage-link').addEventListener('click', openModal);
  $('guard-manage-link').addEventListener('click', openModal);
  $('close-room-btn').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });

  $('copy-invite-btn').addEventListener('click', async () => {
    const shareUrl = roomSync.getShareUrl();
    try {
      await navigator.clipboard.writeText(shareUrl);
      $('copy-invite-btn').textContent = '링크 복사 완료!';
      setTimeout(() => {
        $('copy-invite-btn').textContent = '초대 링크 복사';
      }, 1800);
    } catch {
      window.prompt('아래 초대 링크를 복사해 공유하세요:', shareUrl);
    }
  });

  $('change-room-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const raw = $('room-code-input').value;
    if (!raw.trim()) return;
    const compatNotice = $('compat-isolation-notice');
    if (compatNotice) compatNotice.hidden = true;
    try {
      roomSync.connect(sanitizeRoomCode(raw));
    } catch (syncErr) {
      console.warn('Room switch fallback to local mode:', syncErr);
    }
    $('room-code-input').value = '';
  });

  $('toggle-samples-btn').addEventListener('click', () => {
    if (roomSync.hasSamples()) {
      roomSync.removeSamples();
    } else if (engine.config?.mvp_sample_profiles) {
      roomSync.loadSamples(engine.config.mvp_sample_profiles);
    }
    renderModalProfiles();
  });
}

// Expose programmatic helper for E2E verification without adding UI clutter
window.__keyprintTestApi = {
  engine,
  roomSync,
  state,
  isSampleName,
  getActiveVerdictForViz,
};

async function bootstrap() {
  const initialTabParam = new URLSearchParams(window.location.search).get('tab');
  const initialRoom = getInitialRoomCode();
  roomSync.roomCode = initialRoom;
  $('room-pill-label').textContent = `방 #${initialRoom} · 준비 중`;
  $('modal-room-code').textContent = `#${initialRoom}`;

  bindEvents();
  dealArenaEnrollSentence();
  dealArenaBlindSentence();

  if (initialTabParam === 'principles' || initialTabParam === 'how') {
    switchMainMode('principles', { syncUrl: true });
  } else if (initialTabParam === 'guard') {
    switchMainMode('guard', { syncUrl: true });
  }

  // Initialize ONNX engine FIRST and independently of any network/WebSocket error
  try {
    await engine.init();
    ({ enroll: TARGET_ARENA_SHOTS, query: TARGET_BLIND_SHOTS } = engine.arenaProtocol());
    roomSync.setCompatibility({
      schemaVersion: engine.schemaVersion,
      preprocessVersion: engine.preprocessVersion,
      modelHash: engine.modelHash,
    });
    state.ready = true;
    state.loadError = null;
    $('engine-pill').classList.remove('is-error');
    $('engine-pill').classList.add('is-ready');
    $('engine-status-text').textContent = `ONNX 준비 완료 (${engine.loadTimeMs}ms)`;
  } catch (err) {
    state.ready = false;
    state.loadError = err?.message || '모델 로드 실패';
    $('engine-pill').classList.add('is-error');
    $('engine-status-text').textContent = '모델 로드 실패';
    console.error(err);
  }

  // Connect RoomProfileSync safely even if window.WebSocket throws synchronously
  try {
    roomSync.connect(initialRoom);
  } catch (syncErr) {
    console.warn('Room sync fallback to local mode:', syncErr);
  }

  resetArenaEnrollProgress();
  resetArenaBlindProgress();
  renderAllRosters();

  if (state.ready) {
    try {
      if (!state.viz.lastQuery && state.viz.capture?.stroke) {
        const embeddings = await engine.embedAll(state.viz.capture.stroke);
        const queryId = ++state.querySeq;
        state.viz.lastQuery = {
          queryId,
          embedding: embeddings.sentence,
          embeddings,
          label: '예시 타건 지문',
          domain: 'arena',
        };
      }
      if (state.mode === 'principles') {
        renderPrinciplesView();
      }
    } catch (vizErr) {
      console.warn(vizErr);
    }
  }
}

bootstrap();
