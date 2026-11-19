import { ActionGate } from './asyncActions.js';
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
  EvalStudyManager,
  HANGUL_LAYOUTS,
  KEYBOARD_TYPES,
  POSTURE_OPTIONS,
  SESSION_SLOTS,
  detectSystemEnvironment,
  isNon2BeolsikLayout,
  recommendDeviceOrder,
} from './evalStudy.js';
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
const submissionGate = new ActionGate();
const evalStudy = new EvalStudyManager({
  onChange: () => renderEvalStudyPanel(),
});

const state = {
  ready: false,
  loadError: null,
  evalPanelOpen: false,
  mode: 'arena', // 'arena' | 'guard' | 'principles'
  arenaSub: 'enroll', // 'enroll' | 'blind'
  guardSub: 'verify', // 'verify' | 'enroll'
  lang: 'en', // The prerecorded practice profiles use English.
  thresholdOffset: 0, // 여유(-) / 표준(0) / 엄격(+) relative to the validation-calibrated threshold
  showPasswordPlain: true,

  // Monotonic query counter & official pre-adaptation verdict snapshot
  querySeq: 0,
  lastVerdict: null,

  // Current prompts
  arenaEnrollSentence: '',
  arenaBlindSentence: '',
  pendingArenaPrompts: {
    ko: { enroll: '', blind: '' },
    en: { enroll: '', blind: '' },
  },

  // In-progress enrollment buffers
  arenaEnrollBuffer: { name: '', lang: 'en', items: [] },
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
const FIELD_HUB_BOOT =
  typeof window !== 'undefined' && window.__KEYPRINT_FIELD_HUB?.active ? window.__KEYPRINT_FIELD_HUB : null;

if (
  FIELD_HUB_BOOT &&
  typeof window !== 'undefined' &&
  (!Array.isArray(window.__KEYPRINT_MQTT_BROKERS) || window.__KEYPRINT_MQTT_BROKERS.length === 0)
) {
  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  window.__KEYPRINT_MQTT_BROKERS = [`${wsProto}//${window.location.host}/mqtt`];
}

const roomSync = new RoomProfileSync({
  onChange: (profiles) => {
    state.profiles = Array.isArray(profiles) ? profiles : [];
    if (evalStudy.hubActive) {
      evalStudy.scheduleHubPull(120);
    }
    ensureSelectedGuardAccount();
    renderAllRosters();
    renderEvalStudyPanel();
    if (state.ready) renderEnrollMode();
    renderModalProfiles();
    if (state.mode === 'principles') {
      renderPrinciplesView();
    }
  },
  onStatusChange: ({ status, shared, roomCode, peerCount, profileCount, compatSkipCount }) => {
    const syncDot = $('sync-dot');
    syncDot.className =
      'sync-dot ' +
      (status === 'online' ? 'is-online' : status === 'local' ? 'is-local' : 'is-connecting');

    const peerLabel = status === 'online' ? `기기 ${safeInt(peerCount, 1)}대` : '로컬';
    const roomKind = FIELD_HUB_BOOT ? '로컬 Wi-Fi' : '공유 방';
    $('room-pill-label').textContent = shared
      ? `${roomKind} #${roomCode} · ${peerLabel} · 프로필 ${safeInt(profileCount, 0)}명`
      : `이 브라우저만 · 프로필 ${safeInt(profileCount, 0)}명`;

    $('modal-room-code').textContent = shared ? `${roomKind} #${roomCode}` : '이 브라우저만';
    $('copy-invite-btn').disabled = !shared;
    $('use-local-btn').hidden = !shared;
    $('modal-sync-badge').textContent =
      !shared
        ? '외부 전송 없이 로컬 저장'
        : FIELD_HUB_BOOT
          ? status === 'online'
            ? `노트북 로컬 Wi-Fi 허브 동기화 중 (외부 서버 미전송) · 기기 ${safeInt(peerCount, 1)}대`
            : '노트북 로컬 허브 연결 중…'
          : status === 'online'
            ? `실시간 연결됨 · 현재 기기 ${safeInt(peerCount, 1)}대 접속`
            : status === 'local'
              ? '연결 끊김 · 변경은 로컬 보관 후 재연결 시 동기화'
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
  return profile.arena[lang] || null;
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
    ensureActiveArenaSentence();
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

  ensureActiveArenaSentence();
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
  if (state.mode !== 'arena') return;
  if (state.arenaSub === 'enroll') {
    const nameEl = $('arena-name-input');
    if (!nameEl.value.trim()) nameEl.focus();
    else $('arena-enroll-input').focus();
  } else {
    $('arena-blind-input').focus();
  }
}

function focusActiveGuardInput() {
  if (state.mode !== 'guard') return;
  if (state.guardSub === 'verify') {
    if (getGuardAccounts().length > 0) $('guard-verify-input').focus();
  } else {
    const nameEl = $('guard-enroll-name');
    if (!nameEl.value.trim()) nameEl.focus();
    else $('guard-enroll-input').focus();
  }
}

function renderAllRosters() {
  // 1) Arena Roster
  const arenaList = $('arena-roster-list');
  const cands = getArenaCandidates();
  const otherLanguage = state.profiles.filter((p) => Object.keys(p.arena || {}).length && !p.arena?.[state.lang]).length;
  $('arena-language-notice').textContent = state.lang === 'ko'
    ? '한국어 정량 평가는 공개 스마트폰 타건을 기준으로 합니다. PC 한글 입력은 체험용입니다.'
    : '영어 성능은 공개 타건 데이터로 평가했습니다. 등록할 때와 같은 언어로 비교합니다.';
  $('arena-korean-study-link').hidden = state.lang !== 'ko';
  const rosterNotes = [];
  if (otherLanguage) rosterNotes.push(`다른 언어로 등록된 ${otherLanguage}명은 비교에서 제외됩니다.`);
  if (cands.length === 1) {
    rosterNotes.push('지금은 1명의 등록 리듬과 비교합니다. 예상 작성자 1위를 보려면 같은 언어로 2명 이상 등록하세요.');
  }
  const rosterNote = $('arena-roster-note');
  rosterNote.textContent = rosterNotes.join(' ');
  rosterNote.hidden = rosterNotes.length === 0;
  if (cands.length === 0) {
    arenaList.innerHTML = `<span class="roster-empty">아래에서 ${safeInt(
      TARGET_ARENA_SHOTS,
      5
    )}문장을 연습해 첫 프로필을 등록하세요.</span>`;
  } else {
    arenaList.innerHTML = cands
      .map((c) => {
        const safeName = String(c.name ?? '');
        const initial = escapeHtml(safeName.trim().charAt(0) || '?');
        const cnt = safeInt(c.template?.count, 1);
        const col = safeColor(c.color, '#2563eb');
        const tip = c.isSample
          ? '샘플 프로필은 시연용이라 수정할 수 없습니다'
          : '이 프로필에 문장을 추가로 연습합니다 (리듬 점수가 기준을 충족하면 반영)';
        return `<button type="button" class="user-chip" data-arena-pick="${escapeHtml(safeName)}" data-sample="${
          c.isSample ? '1' : '0'
        }" title="${escapeHtml(tip)}">
          <span class="chip-avatar" style="background:${col}">${initial}</span>
          <span class="chip-name">${escapeHtml(safeName)}</span>
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
        submissionGate.invalidate();
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
          <span class="chip-avatar" style="background:${col}">${initial}</span>
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
function setArenaSentence(sub, sentence) {
  const isEnroll = sub === 'enroll';
  const slot = `arena-${sub}`;
  if (isEnroll) state.arenaEnrollSentence = sentence;
  else state.arenaBlindSentence = sentence;
  const recorder = isEnroll ? arenaEnrollRecorder : arenaBlindRecorder;
  $(`${slot}-input`).value = '';
  recorder.reset();
  renderSentencePrompt($(`${slot}-prompt`), sentence, '');
  $(`${slot}-cpm`).textContent = '0';
  $(`${slot}-acc`).textContent = '100%';
  $(`${slot}-edit-hint`).textContent = isEnroll ? '오타 시 백스페이스 수정 가능' : '백스페이스 자유 사용';
}

function siblingArenaPrompt(sub, lang = state.lang) {
  return state.pendingArenaPrompts[lang]?.[sub === 'enroll' ? 'blind' : 'enroll'] || '';
}

function dealArenaSentence(sub) {
  const lang = state.lang === 'ko' ? 'ko' : 'en';
  const previous =
    (sub === 'enroll' ? state.arenaEnrollSentence : state.arenaBlindSentence) ||
    state.pendingArenaPrompts[lang][sub];
  const active = state.mode === 'arena' && state.arenaSub === sub && !$('room-dialog').open;
  if (!active) {
    state.pendingArenaPrompts[lang][sub] = '';
    setArenaSentence(sub, '');
    return;
  }
  const exclude = [siblingArenaPrompt(sub, lang), previous].filter(Boolean);
  const nextSentence = deck.next(lang, { exclude });
  state.pendingArenaPrompts[lang][sub] = nextSentence;
  setArenaSentence(sub, nextSentence);
}

function ensureActiveArenaSentence() {
  if (state.mode !== 'arena' || $('room-dialog').open) return;
  const lang = state.lang === 'ko' ? 'ko' : 'en';
  const sub = state.arenaSub;
  const current = sub === 'enroll' ? state.arenaEnrollSentence : state.arenaBlindSentence;
  if (current) return;
  let pending = state.pendingArenaPrompts[lang][sub];
  if (!pending) {
    const exclude = [siblingArenaPrompt(sub, lang)].filter(Boolean);
    pending = deck.next(lang, { exclude });
    state.pendingArenaPrompts[lang][sub] = pending;
  }
  setArenaSentence(sub, pending);
}

function resetArenaSentences() {
  setArenaSentence('enroll', '');
  setArenaSentence('blind', '');
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
    $('arena-step-text').textContent = `추가 연습 중 · '${target.name}' ${n}문장 (리듬 기준 충족 시 최근 12문장에 반영)`;
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

function switchLanguage(lang, { deferPrompt = false } = {}) {
  submissionGate.invalidate();
  state.querySeq += 1;
  state.lastVerdict = null;
  state.lang = lang;
  for (const language of ['ko', 'en']) {
    const button = $(`lang-${language}-btn`);
    button.classList.toggle('is-active', language === lang);
    button.setAttribute('aria-pressed', String(language === lang));
  }

  resetArenaEnrollProgress();
  resetArenaBlindProgress();
  resetArenaSentences();
  if (!deferPrompt) ensureActiveArenaSentence();
  $('arena-result-card').hidden = true;
  setFeedback($('arena-enroll-feedback'), '');
  setFeedback($('arena-blind-feedback'), '');
  renderAllRosters();
  renderEvalStudyPanel();
  if (state.mode === 'principles') {
    renderPrinciplesView();
  } else if (!deferPrompt) {
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
  onSubmit: () => runSubmission('arena-enroll', handleArenaEnrollSubmit),
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
  onSubmit: () => runSubmission('arena-blind', handleArenaBlindSubmit),
});

const guardVerifyRecorder = new SmartKeystrokeRecorder($('guard-verify-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
  },
  onSubmit: () => runSubmission('guard-verify', handleGuardVerifySubmit),
});

const guardEnrollRecorder = new SmartKeystrokeRecorder($('guard-enroll-input'), {
  onLiveUpdate: (live) => {
    invalidateLastVerdictOnTyping(live);
  },
  onSubmit: () => runSubmission('guard-enroll', handleGuardEnrollSubmit),
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

async function runSubmission(slot, action) {
  const feedback = $(`${slot}-feedback`);
  if (submissionGate.pending) {
    setFeedback(feedback, '입력을 분석 중입니다. 완료 후 다시 제출해 주세요.');
    return;
  }
  const input = $(`${slot}-input`);
  try {
    if (input) input.readOnly = true;
    await submissionGate.run(async (isCurrent) => {
      try { await action(isCurrent); }
      catch (err) {
        if (isCurrent()) setFeedback(feedback, `분석을 완료하지 못했습니다. 다시 입력해 주세요. (${err?.message || '추론 오류'})`, 'error');
      }
    });
  } finally {
    if (input) input.readOnly = false;
  }
}

function resetRoomWork() {
  submissionGate.invalidate();
  state.querySeq += 1;
  state.lastVerdict = null;
  state.viz.lastQuery = null;
  state.viz.selectedProfileId = null;
  if (vizEmbedTimer) clearTimeout(vizEmbedTimer);
  vizEmbedTimer = null;
  $('arena-name-input').value = '';
  $('guard-enroll-name').value = '';
  $('guard-enroll-pw-target').value = '';
  state.pwEnrollBuffer = { name: '', targetPw: '', items: [] };
  state.selectedGuardId = null;
  resetArenaEnrollProgress();
  resetArenaBlindProgress();
  resetArenaSentences();
  ensureActiveArenaSentence();
  for (const [slot, recorder] of [['guard-enroll', guardEnrollRecorder], ['guard-verify', guardVerifyRecorder], ['viz-sandbox', vizSandboxRecorder]]) {
    $(`${slot}-input`).value = '';
    recorder.reset();
  }
  state.viz.capture = createPresetCapture('normal');
  for (const slot of ['arena-enroll', 'arena-blind', 'guard-enroll', 'guard-verify']) setFeedback($(`${slot}-feedback`), '');
  $('arena-result-card').hidden = true;
  $('guard-result-card').hidden = true;
  renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
}

function changeWorkspace(roomCode, shared) {
  resetRoomWork();
  roomSync.connect(roomCode, { shared });
  $('share-room-consent').checked = false;
  setFeedback($('room-feedback'), shared ? '공개 공유를 시작했습니다. 이 방에서 등록하는 프로필과 시연 문자열이 브로커에 전송됩니다.' : '이 브라우저의 로컬 프로필로 돌아왔습니다. 외부 전송을 중지했습니다.');
}

function startEnglishSampleExperience() {
  if (!state.ready) {
    setFeedback($('room-feedback'), notReadyMessage(), 'error');
    return;
  }
  roomSync.loadSamples(engine.config?.mvp_sample_profiles || []);
  switchLanguage('en', { deferPrompt: true });
  $('room-dialog').close();
  switchArenaSub('blind');
  switchMainMode('arena');
  setFeedback($('arena-blind-feedback'), '다른 사람이 입력한 영어 실측 샘플과 비교합니다. 내 리듬과 비교하려면 내 프로필을 등록하세요.');
}

// ================= ACTION HANDLERS =================
async function handleArenaEnrollSubmit(isCurrent) {
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

  const enrollMotion = evalStudy._captureMotionWindow(arenaEnrollRecorder);
  const enrollPrompt = state.arenaEnrollSentence;
  const { embedding } = await engine.embedStroke(capture.stroke);
  if (!isCurrent()) return;
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
    _rawCapture: capture,
    _prompt: enrollPrompt,
    _motion: enrollMotion,
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

    dealArenaSentence('enroll');
    const scoreSummary = `리듬 점수 ${formatArenaScore(verdict.similarity)} · 기준 ${formatArenaScore(verdict.threshold)}`;
    if (!verdict.accepted) {
      setFeedback(
        $('arena-enroll-feedback'),
        `'${name}'님의 등록 리듬과 비교한 점수가 기준보다 낮아 이 문장은 반영하지 않았습니다 (${scoreSummary}). 다음 문장을 이어서 쳐 보세요.`,
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
      `'${name}' 프로필에 반영했습니다 (${scoreSummary} · 총 ${mergedTmpl.count}문장, 최근 12문장 사용).`,
      'success'
    );
    renderEnrollMode();
    return;
  }

  if (done < TARGET_ARENA_SHOTS) {
    renderStepDots($('arena-step-dots'), $('arena-step-text'), done, TARGET_ARENA_SHOTS, '문장');
    dealArenaSentence('enroll');
    const bsNote =
      capture.backspaceCount > 0 ? ` (백스페이스 ${capture.backspaceCount}회 자동 보정)` : '';
    setFeedback(
      $('arena-enroll-feedback'),
      `${done}/${TARGET_ARENA_SHOTS} 완료${bsNote} — 다음 문장을 이어서 타이핑해 주세요.`,
      'success'
    );
    return;
  }

  const enrolledItems = [...state.arenaEnrollBuffer.items];
  let tmpl = existing?.arena?.[state.lang] || null;
  for (const item of enrolledItems) {
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

  const savedArenaProfile = roomSync.upsertProfile(profileObj);
  if (savedArenaProfile) {
    await evalStudy.recordEnrollment({
      domain: 'arena',
      lang: state.lang,
      roomCode: roomSync.roomCode,
      profile: savedArenaProfile,
      template: tmpl,
      captures: enrolledItems.map((it) => it._rawCapture).filter(Boolean),
      motionSnapshots: enrolledItems.map((it) => it._motion).filter(Boolean),
      prompts: enrolledItems.map((it) => it._prompt).filter(Boolean),
      engine,
    });
  }

  $('arena-name-input').value = '';
  resetArenaEnrollProgress();
  dealArenaSentence('enroll');
  setFeedback(
    $('arena-enroll-feedback'),
    `'${name}' 프로필이 등록되었습니다. 다음 참가자를 등록하거나 [2. 누가 쳤을까?] 탭에서 리듬을 비교해 보세요.`,
    'success'
  );
}

async function handleArenaBlindSubmit(isCurrent) {
  $('arena-result-card').hidden = true;
  if (!state.ready) {
    setFeedback($('arena-blind-feedback'), notReadyMessage(), 'error');
    return;
  }
  const candidates = getArenaCandidates();
  if (candidates.length === 0) {
    setFeedback(
      $('arena-blind-feedback'),
      '현재 언어로 등록된 참가자가 없습니다. 먼저 프로필을 등록하거나 [영어 샘플로 체험]을 선택해 주세요.',
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

  const blindMotion = evalStudy._captureMotionWindow(arenaBlindRecorder);
  const blindPrompt = state.arenaBlindSentence;
  const { embedding: oneEmbedding, latencyMs } = await engine.embedStroke(capture.stroke);
  if (!isCurrent()) return;
  state.arenaBlindBuffer.push({
    embedding: oneEmbedding,
    capture,
    accuracy: check.accuracy,
    prompt: blindPrompt,
    motion: blindMotion,
  });
  const got = state.arenaBlindBuffer.length;
  if (got < TARGET_BLIND_SHOTS) {
    renderStepDots($('arena-blind-step-dots'), $('arena-blind-step-text'), got, TARGET_BLIND_SHOTS, '문장');
    setFeedback(
      $('arena-blind-feedback'),
      `${got}/${TARGET_BLIND_SHOTS} 완료. 한 문장 더 입력하면 비교 결과가 나옵니다.`,
      'success'
    );
    dealArenaSentence('blind');
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
    sourceLabel: '내 실시간 타건 · 누가 쳤을까? 비교 문장',
  };
  state.viz.mapDomain = 'arena';
  state.viz.selectedProfileId = res.winner?.id || state.viz.selectedProfileId;
  state.viz.lastQuery = {
    queryId,
    embedding,
    embeddings: { sentence: embedding },
    label: '방금 비교한 타건',
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

  await evalStudy.recordArenaQuery({
    lang: state.lang,
    roomCode: roomSync.roomCode,
    candidates,
    res,
    items,
    combinedEmbedding: embedding,
    mergedBehavior: behavior,
    inputMode: effectiveMode,
    latencyMs,
    recorder: arenaBlindRecorder,
    motionSnapshot: blindMotion,
    engine,
  });

  dealArenaSentence('blind');
}

function formatArenaScore(score, digits = 1) {
  return `${(score * 100).toFixed(digits)}점`;
}

function arenaDecisionLabel(accepted, candidateCount) {
  if (candidateCount === 1) return accepted ? '리듬 일치' : '리듬 비교 보류';
  return accepted ? '판정 수락' : '판정 보류';
}

function renderArenaResult(res, stats) {
  const singleCandidate = res.rankings.length === 1;
  $('arena-result-kicker').textContent = singleCandidate ? '1:1 비교 대상' : '예상 작성자 1위';
  $('arena-result-title').textContent = res.winner.name;
  $('arena-result-score').textContent = formatArenaScore(res.winner.similarity);

  $('arena-result-decision').classList.toggle('is-pass', res.identified);
  $('arena-result-status').textContent = arenaDecisionLabel(res.identified, res.rankings.length);
  let reason = '리듬 점수가 기준보다 낮습니다.';
  if (res.identified) {
    reason = singleCandidate ? '리듬 점수가 기준을 충족했습니다.' : '리듬 점수와 1·2위 점수 차가 기준을 충족했습니다.';
  } else if (res.rejectedByMargin) {
    reason = '1·2위 점수 차가 기준보다 작습니다.';
  }
  $('arena-result-reason').textContent = reason;

  const minimumScore = `최소 리듬 점수 ${formatArenaScore(res.openSetThreshold)}`;
  $('arena-result-criteria').textContent = singleCandidate
    ? minimumScore
    : `1·2위 점수 차 ${formatArenaScore(res.lead, 2)} (최소 ${formatArenaScore(res.margin, 2)}) · ${minimumScore}`;

  const rankingBars = $('arena-ranking-bars');
  rankingBars.hidden = singleCandidate;
  const topList = singleCandidate ? [] : res.rankings.slice(0, 5);
  rankingBars.innerHTML = topList
    .map((r, idx) => {
      const simNum = Number(r?.similarity) || 0;
      const sPct = Math.max(4, Math.min(100, safeInt(simNum * 100, 4)));
      const col = idx === 0 ? safeColor(r?.color, '#059669') : '#9ca3af';
      return `<div class="rank-item">
        <span class="rank-name">${safeInt(idx + 1)}. ${escapeHtml(r?.name)}</span>
        <div class="rank-track"><div class="rank-fill" style="width:${sPct}%;background:${col}"></div></div>
        <span class="rank-val">${formatArenaScore(simNum)}</span>
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
  $('arena-result-card').hidden = false;
}

async function handleGuardEnrollSubmit(isCurrent) {
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

  const enrollPwMotion = evalStudy._captureMotionWindow(guardEnrollRecorder);
  const both = await engine.embedAll(capture.stroke);
  if (!isCurrent()) return;
  state.pwEnrollBuffer.items.push({
    embedding: both.password,
    embedding17: both.passwordAux,
    capitalMethods: capture.capitalMethods,
    cpm: capture.cpm,
    holdMs: capture.holdMs,
    rolloverPct: capture.rolloverPct,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
    _rawCapture: capture,
    _motion: enrollPwMotion,
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

  const enrolledPwItems = [...state.pwEnrollBuffer.items];
  let pwTmpl = null;
  for (const item of enrolledPwItems) {
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

  await evalStudy.recordEnrollment({
    domain: 'password',
    roomCode: roomSync.roomCode,
    profile: saved,
    template: pwTmpl,
    captures: enrolledPwItems.map((it) => it._rawCapture).filter(Boolean),
    motionSnapshots: enrolledPwItems.map((it) => it._motion).filter(Boolean),
    passwordText: targetPw,
    engine,
  });

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

async function handleGuardVerifySubmit(isCurrent) {
  if (!state.ready) {
    setFeedback($('guard-verify-feedback'), notReadyMessage(), 'error');
    return;
  }
  let accounts = getGuardAccounts();
  if (accounts.length === 0) return;

  const targetAcc = accounts.find((a) => a.id === state.selectedGuardId) || accounts[0];
  const expectedPw = targetAcc.password.target;

  const capture = guardVerifyRecorder.extractCleanStroke();
  const verifyMotionSnapshot = evalStudy._captureMotionWindow(guardVerifyRecorder);
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
  if (!isCurrent()) return;
  accounts = getGuardAccounts();
  if (state.selectedGuardId !== targetAcc.id || !accounts.includes(targetAcc)) {
    state.lastVerdict = null;
    $('guard-result-card').hidden = true;
    setFeedback($('guard-verify-feedback'), '분석 중 프로필이 변경되었습니다. 사용할 프로필을 선택하고 다시 입력해 주세요.', 'error');
    return;
  }
  const embedding = both.password;
  const embedding17 = both.passwordAux;
  const latencyMs = both.latencyMs;

  const scoreOpts = {
    domain: 'password',
    thresholdOffset: state.thresholdOffset,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
    embedding17,
    capitalMethods: capture.capitalMethods,
    lockAdaptation: evalStudy.shouldLockAdaptation(),
    useStringCalibration: evalStudy.stringCalibrationOverride(),
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
    embeddings: both,
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
    $('guard-result-kicker').textContent = '리듬 일치 · 시연 기준 통과';
    $('guard-result-title').textContent = `${targetAcc.name}님의 타이핑 리듬과 일치합니다`;
    $('guard-result-desc').textContent = `시연 문자열이 일치했고 타이핑 리듬 점수가 기준(${Math.round(
      threshold * 100
    )}%) 이상입니다.`;
  } else {
    card.className = 'result-card is-block';
    $('guard-result-kicker').textContent = '확인 보류 · 리듬 기준 미달';
    $('guard-result-title').textContent = '비밀번호는 맞지만 등록 리듬과의 일치를 확인할 수 없습니다';
    const hint = closerOther
      ? ` 입력된 리듬은 등록된 '${closerOther.name}'님의 타건 패턴(${(closerOther.sim * 100).toFixed(
          0
        )}%)에 더 가깝습니다.`
      : '';
    $('guard-result-desc').textContent = `${targetAcc.name}님의 등록 리듬과 충분히 일치하지 않습니다. 본인이라도 입력 속도·기기·환경이 달라지면 기준에 미달할 수 있습니다.${hint}`;
  }
  card.dataset.reference = touchReference ? 'true' : 'false';
  if (touchReference) {
    const touchOnly = capture.inputMode === 'touch';
    $('guard-result-kicker').textContent += touchOnly ? ' · 모바일 참고용' : ' · 혼합 입력 참고용';
    $('guard-result-desc').textContent += touchOnly
      ? ' 터치 입력에서는 키를 누르고 있던 시간을 측정할 수 없어 이 결과를 참고용으로 표시합니다.'
      : ' 이 입력에는 누름 시간을 측정할 수 없는 키가 포함되어 이 결과를 참고용으로 표시합니다.';
  }

  const editTxt =
    capture.backspaceCount > 0 ? `백스페이스 ${capture.backspaceCount}회 보정` : '무수정 입력';
  const holdTxt =
    capture.inputMode === 'touch'
      ? `${INPUT_MODE_LABEL.touch}(누름 시간 대체)`
      : `홀드 ${capture.holdMs}ms · 롤오버 ${capture.rolloverPct}%`;
  const adaptTxt = adapted
    ? ` · 등록 리듬 갱신(최근 ${Math.min(12, (targetAcc.password.count || 0) + 1)}회 반영)`
    : evalStudy.shouldLockAdaptation()
      ? ' · 평가 잠금(템플릿 고정)'
      : '';
  $('guard-result-meta').textContent = `${capture.cpm} 타/분 · ${holdTxt} · ${editTxt}${behaviorNote(res)}${deviceMismatchNote(targetAcc.password, capture.inputMode)}${adaptTxt} · 추론 ${latencyMs}ms`;

  const calibCompareEl = $('guard-calib-compare');
  if (calibCompareEl) {
    if (res.lengthBaseline && res.stringCalibrated && (evalStudy.settings.enabled || res.stringCalibrationActive)) {
      const lb = res.lengthBaseline;
      const sc = res.stringCalibrated;
      const activeTag = res.stringCalibrationActive ? '문자열 보정 적용 중' : '길이 보정 적용 중';
      calibCompareEl.textContent = `[${activeTag}] 길이 기준 ${(lb.similarity * 100).toFixed(1)}% (z=${lb.z.toFixed(
        2
      )}) vs 문자열 보정 ${(sc.similarity * 100).toFixed(1)}% (z=${sc.z.toFixed(2)}, 지지도 w=${sc.shrinkWeight.toFixed(
        2
      )}, ${sc.latencyMs}ms)`;
      calibCompareEl.hidden = false;
    } else {
      calibCompareEl.hidden = true;
    }
  }

  await evalStudy.recordGuardQuery({
    roomCode: roomSync.roomCode,
    targetAccount: targetAcc,
    passwordText: expectedPw,
    capture,
    bothEmbeddings: both,
    res,
    thresholdOffset: state.thresholdOffset,
    adapted,
    latencyMs,
    recorder: guardVerifyRecorder,
    motionSnapshot: verifyMotionSnapshot,
    engine,
  });
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
    const scheduledSeq = state.querySeq;
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
      let embeddings;
      try { embeddings = await engine.embedAll(cap.stroke); }
      catch { showVizError(); return; }
      if (state.querySeq !== scheduledSeq) return;
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
  if (v.domain === 'password') {
    const simPct = (v.similarity * 100).toFixed(1);
    const thrPct = (v.threshold * 100).toFixed(1);
    const statusLabel = v.accepted ? '리듬 일치 · 시연 기준 통과' : '확인 보류 · 리듬 기준 미달';
    const refTag = !v.touchReference ? '' : v.inputMode === 'touch' ? ' · 모바일 참고용' : ' · 혼합 입력 참고용';
    return `공식 판정 스냅샷 · [${winnerName}] ${statusLabel}${refTag} — 일치도 ${simPct}% (임계값 ${thrPct}%)`;
  }

  const candidateCount = v.candidates.length;
  const subject = candidateCount === 1 ? '1:1 비교 대상' : '예상 작성자 1위';
  const decision = arenaDecisionLabel(v.accepted, candidateCount);
  const score = `리듬 점수 ${formatArenaScore(v.similarity)} (기준 ${formatArenaScore(v.threshold)})`;
  const gap = v.rejectedReason === 'margin_hold'
    ? ` · 1·2위 점수 차 ${formatArenaScore(v.lead, 2)} (최소 ${formatArenaScore(v.margin, 2)})`
    : '';
  return `공식 판정 스냅샷 · ${subject} ${winnerName} · ${decision} — ${score}${gap}`;
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
      verdictLine: isArenaMap
        ? `*${arenaDecisionLabel(activeVerdict.accepted, activeVerdict.candidates.length)} · 리듬 점수 ${formatArenaScore(activeVerdict.similarity)} / 기준 ${formatArenaScore(activeVerdict.threshold)}`
        : `*공식 판정: ${simPct}% / 기준 ${thrPct}% (${
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
      statusEl.textContent = '실시간 탐색 미리보기 (비교 결과 확정 전)';
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
            verdictLine: `*탐색 리듬 점수 ${formatArenaScore(scored.winner.similarity)} (미리보기)`,
          };
        }
      } else {
        const q17 = q?.embeddings?.passwordAux || null;
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

function showVizError() { $('viz-signal-badge').textContent = '분석하지 못했습니다. 잠시 후 다시 입력해 주세요.'; }

async function applyVizPreset(mode) {
  $('viz-sandbox-input').value = '';
  vizSandboxRecorder.reset();
  const scheduledSeq = ++state.querySeq;
  const cap = createPresetCapture(mode);
  state.viz.capture = cap;
  state.lastVerdict = null;
  renderVizStep1();
  if (state.ready) {
    const embeddings = await engine.embedAll(cap.stroke);
    if (state.querySeq !== scheduledSeq) return;
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
  $('viz-preset-fast-btn').addEventListener('click', () => applyVizPreset('normal').catch(showVizError));
  $('viz-preset-slow-btn').addEventListener('click', () => applyVizPreset('slow').catch(showVizError));

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

  for (const lang of ['ko', 'en']) {
    $(`lang-${lang}-btn`).addEventListener('click', () => {
      if (state.lang !== lang) switchLanguage(lang);
    });
  }

  $('arena-next-sentence-btn').addEventListener('click', () => {
    submissionGate.invalidate();
    dealArenaSentence('enroll');
    $('arena-enroll-input').focus();
  });

  $('arena-blind-shuffle-btn').addEventListener('click', () => {
    submissionGate.invalidate();
    dealArenaSentence('blind');
    $('arena-blind-input').focus();
  });

  $('arena-retry-btn').addEventListener('click', () => {
    $('arena-result-card').hidden = true;
    dealArenaSentence('blind');
    $('arena-blind-input').focus();
  });

  $('arena-name-input').addEventListener('input', () => {
    submissionGate.invalidate();
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

  $('guard-enroll-name').addEventListener('input', () => submissionGate.invalidate());
  $('guard-enroll-pw-target').addEventListener('input', () => {
    submissionGate.invalidate();
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
  dialog.addEventListener('close', ensureActiveArenaSentence);
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
    if (!state.ready) {
      setFeedback($('room-feedback'), notReadyMessage(), 'error');
      return;
    }
    const compatNotice = $('compat-isolation-notice');
    if (compatNotice) compatNotice.hidden = true;
    if (!$('share-room-consent').checked) {
      setFeedback($('room-feedback'), '공개 브로커로 전송되는 항목을 확인하고 공유 동의에 체크해 주세요.', 'error');
      return;
    }
    changeWorkspace(sanitizeRoomCode(raw), true);
  });

  $('use-local-btn').addEventListener('click', () => changeWorkspace('LOCAL', false));
  $('arena-sample-start-btn').addEventListener('click', startEnglishSampleExperience);
  $('sample-start-btn').addEventListener('click', startEnglishSampleExperience);

  $('toggle-samples-btn').addEventListener('click', () => {
    if (roomSync.hasSamples()) {
      roomSync.removeSamples();
    } else if (engine.config?.mvp_sample_profiles) {
      roomSync.loadSamples(engine.config.mvp_sample_profiles);
    }
    renderModalProfiles();
  });

  bindEvalStudyEvents();
}

// ================= EVALUATION & CROSS-DEVICE FIELD LOGGER UI =================
function populateEvalStaticSelects() {
  const fillOptions = (sel, items) => {
    if (!sel) return;
    sel.innerHTML = items
      .map((it) => `<option value="${escapeHtml(it.value)}">${escapeHtml(it.label)}</option>`)
      .join('');
  };
  fillOptions($('eval-keyboard-type'), KEYBOARD_TYPES);
  fillOptions($('eval-hangul-layout'), HANGUL_LAYOUTS);
  fillOptions($('eval-posture'), POSTURE_OPTIONS);
  fillOptions($('eval-session-slot'), SESSION_SLOTS);
  populateGuardPresetSelect();
}

function populateGuardPresetSelect() {
  const sel = $('guard-exp-preset-select');
  if (!sel) return;
  const list = evalStudy.publicStrings || [];
  const core = list.filter((x) => x.tier === 'core');
  const ext = list.filter((x) => x.tier !== 'core');
  const renderOpt = (item) =>
    `<option value="${escapeHtml(item.text)}">[${escapeHtml(item.id)}] ${escapeHtml(item.text)} — ${escapeHtml(
      item.label
    )}</option>`;
  sel.innerHTML =
    `<option value="">직접 입력 유지 (사용자 지정 문자열)</option>` +
    (core.length ? `<optgroup label="핵심 6종 (빠른 4기기 교차 실측 권장)">${core.map(renderOpt).join('')}</optgroup>` : '') +
    (ext.length ? `<optgroup label="확장 18종 (문자열 코호트 보정 실험)">${ext.map(renderOpt).join('')}</optgroup>` : '');
}

function renderEvalStudyPanel() {
  const panel = $('eval-study-panel');
  const toggleBtn = $('toggle-eval-btn');
  if (!panel || !toggleBtn) return;

  const s = evalStudy.settings;
  panel.hidden = !state.evalPanelOpen;
  toggleBtn.setAttribute('aria-expanded', String(state.evalPanelOpen));
  toggleBtn.classList.toggle('is-recording', Boolean(s.enabled));
  const dot = $('eval-dot');
  if (dot) dot.className = `sync-dot ${s.enabled ? 'is-online' : 'is-local'}`;

  const summary = evalStudy.computeLiveSummary();
  const countBadge = $('eval-pill-count');
  if (countBadge) {
    countBadge.textContent = String(summary.totalRecords);
    countBadge.hidden = summary.totalRecords === 0 && !s.enabled;
  }

  $('eval-enabled-chk').checked = Boolean(s.enabled);
  if (document.activeElement !== $('eval-participant-id')) {
    $('eval-participant-id').value = s.participantId || 'P01';
  }
  $('eval-keyboard-type').value = s.keyboardType;
  $('eval-hangul-layout').value = s.hangulLayout;
  $('eval-posture').value = s.posture;
  $('eval-session-slot').value = s.sessionSlot;
  $('eval-lock-adapt-chk').checked = Boolean(s.lockAdaptation);
  $('eval-raw-consent-chk').checked = Boolean(s.rawConsent);
  $('eval-string-calib-mode').value = s.stringCalibMode || 'auto';

  // Populate ground-truth writer options from live profiles
  const gtSelect = $('eval-gt-role');
  const prevGt = s.groundTruthRole || 'unspecified';
  const profOpts = state.profiles
    .filter((p) => p && typeof p.name === 'string')
    .map(
      (p) =>
        `<option value="${escapeHtml(`enrolled:${p.id}`)}">등록 작성자: ${escapeHtml(p.name)}${
          p.isSample ? ' (샘플)' : ''
        }</option>`
    )
    .join('');
  gtSelect.innerHTML =
    `<option value="unspecified">미지정 (일반 체험)</option>` +
    `<option value="outsider">미등록 외부인 (타인·침입 테스트)</option>` +
    profOpts;
  const validGtValues = new Set(Array.from(gtSelect.options).map((o) => o.value));
  gtSelect.value = validGtValues.has(prevGt) ? prevGt : 'unspecified';
  if (gtSelect.value !== prevGt) {
    evalStudy.settings.groundTruthRole = gtSelect.value;
  }

  const orderRec = recommendDeviceOrder(s.participantId);
  $('eval-order-banner').textContent = `권장 기기 순서 (${s.participantId || 'P01'} · 라틴 방진 ${orderRec.cohortRow}열): ${
    orderRec.formatted
  }`;

  const layoutWarn = state.lang === 'ko' && isNon2BeolsikLayout(s.hangulLayout);
  $('eval-layout-warning').hidden = !layoutWarn;
  $('eval-gt-warning').hidden = !(s.enabled && gtSelect.value === 'unspecified');

  const motion = evalStudy.motion.getLiveStatus();
  const motionEl = $('eval-motion-status');
  const isTouchEnv = navigator.maxTouchPoints > 0 || s.keyboardType.startsWith('touch_');
  const insecureHttp = typeof window !== 'undefined' && window.isSecureContext === false;
  if (motion.permissionState === 'unsupported') {
    motionEl.textContent =
      isTouchEnv && insecureHttp
        ? 'HTTP 접속 중 · HTTPS(:8443)로 이동 시 모션 활성'
        : '모션 센서 미지원 기기 (PC 브라우저)';
    motionEl.classList.remove('is-active');
  } else if (motion.permissionState === 'denied') {
    motionEl.textContent = '모션 센서 권한 거부됨';
    motionEl.classList.remove('is-active');
  } else if (motion.permissionState === 'active') {
    const hzTxt = motion.effectiveHz ? `${motion.effectiveHz}Hz (${motion.medianIntervalMs}ms)` : `${motion.sampleCount}샘플`;
    motionEl.textContent = `모션 측정 중 · ${hzTxt}`;
    motionEl.classList.add('is-active');
  } else if (motion.permissionState === 'granted') {
    motionEl.textContent = '모션 권한 허용됨 · 이벤트 대기';
    motionEl.classList.add('is-active');
  } else {
    motionEl.textContent = '모션 센서 대기 (스마트폰 클릭 시 권한 요청)';
    motionEl.classList.remove('is-active');
  }

  const sys = detectSystemEnvironment();
  $('eval-auto-env-meta').textContent = `자동 감지: ${sys.os} · ${sys.browser} · 타이머 해상도 ~${
    sys.timerResolutionUs ?? '?'
  }µs · 모델 ${engine.modelHash || '준비중'} (${engine.preprocessVersion})`;

  const hubBox = $('eval-lan-hub-box');
  if (hubBox) {
    if (evalStudy.hubActive && evalStudy.hubInfo) {
      const info = evalStudy.hubInfo;
      hubBox.hidden = false;
      $('eval-lan-hub-title').textContent = `로컬 Wi-Fi 다기기 실측 허브 연결됨 (#${roomSync.roomCode})`;
      $('eval-lan-hub-sub').textContent = `노트북·외장 키보드·스마트폰 프로필이 동일 공유기 내에서 실시간 공유되며, 모든 평가 기록은 노트북의 ${
        info.logFileRel || 'logs/field_study/field_study_merged.jsonl'
      }에 자동 통합 저장됩니다 (현재 ${info.savedRecordCount ?? summary.totalRecords}건 · 외부 서버 미전송).`;

      const qrDetails = $('eval-lan-qr-details');
      if (qrDetails && !qrDetails.dataset.autoToggled) {
        qrDetails.dataset.autoToggled = '1';
        if (sys.os === 'iOS' || sys.os === 'Android') {
          qrDetails.open = false;
        }
      }

      const qrHttpsEl = $('eval-qr-https');
      if (qrHttpsEl && info.qrSvgHttps && qrHttpsEl.dataset.renderedUrl !== info.httpsUrl) {
        qrHttpsEl.innerHTML = info.qrSvgHttps;
        qrHttpsEl.dataset.renderedUrl = info.httpsUrl;
      }
      const qrHttpEl = $('eval-qr-http');
      if (qrHttpEl && info.qrSvgHttp && qrHttpEl.dataset.renderedUrl !== info.httpUrl) {
        qrHttpEl.innerHTML = info.qrSvgHttp;
        qrHttpEl.dataset.renderedUrl = info.httpUrl;
      }

      const linkHttps = $('eval-link-https');
      if (linkHttps && info.httpsUrl) {
        linkHttps.href = info.httpsUrl;
        linkHttps.textContent = info.httpsUrl;
      }
      const linkHttp = $('eval-link-http');
      if (linkHttp && info.httpUrl) {
        linkHttp.href = info.httpUrl;
        linkHttp.textContent = info.httpUrl;
      }
      const caLink = $('eval-hub-ca-link');
      if (caLink) {
        caLink.hidden = !info.hasCaCert;
      }
    } else {
      hubBox.hidden = true;
    }
  }

  const statParts = [
    `누적 ${summary.totalRecords}건 (등록 ${summary.enrollCount} · 문장판별 ${summary.arenaQueryCount} · 암호판별 ${summary.pwQueryCount})`,
  ];
  if (summary.arenaTop1Pct !== null) statParts.push(`문장 1위 ${summary.arenaTop1Pct}%`);
  if (summary.arenaAcceptPct !== null) statParts.push(`문장 수락 ${summary.arenaAcceptPct}%`);
  if (summary.arenaOutsiderFarPct !== null) statParts.push(`외부인 오수락 ${summary.arenaOutsiderFarPct}%`);
  if (summary.pwGenuineAcceptPct !== null) statParts.push(`암호 본인수락 ${summary.pwGenuineAcceptPct}%`);
  if (summary.pwImpostorFarPct !== null) statParts.push(`암호 타인오수락 ${summary.pwImpostorFarPct}%`);
  $('eval-live-stats').textContent = statParts.join(' · ');

  $('eval-export-jsonl-btn').disabled = summary.totalRecords === 0;
  $('eval-clear-logs-btn').disabled = summary.totalRecords === 0;
}

function bindEvalStudyEvents() {
  populateEvalStaticSelects();

  $('eval-config-form').addEventListener('submit', (e) => e.preventDefault());

  $('toggle-eval-btn').addEventListener('click', () => {
    state.evalPanelOpen = !state.evalPanelOpen;
    renderEvalStudyPanel();
  });

  $('close-eval-panel-btn').addEventListener('click', () => {
    state.evalPanelOpen = false;
    renderEvalStudyPanel();
  });

  $('eval-enabled-chk').addEventListener('change', (e) => {
    evalStudy.updateSettings({ enabled: e.target.checked });
  });

  $('eval-participant-id').addEventListener('input', (e) => {
    const clean = e.target.value.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 24);
    if (clean !== e.target.value) e.target.value = clean;
    evalStudy.updateSettings({ participantId: clean || 'P01' });
  });

  $('eval-gt-role').addEventListener('change', (e) => {
    evalStudy.updateSettings({ groundTruthRole: e.target.value });
  });

  $('eval-keyboard-type').addEventListener('change', (e) => {
    const kb = e.target.value;
    const patch = { keyboardType: kb };
    if (kb.startsWith('touch_') && evalStudy.settings.hangulLayout === 'pc_2beolsik') {
      patch.hangulLayout = 'mobile_qwerty_2beolsik';
      if (evalStudy.settings.posture === 'desk_keyboard') patch.posture = 'hand_two_thumbs';
    } else if (!kb.startsWith('touch_') && evalStudy.settings.hangulLayout.startsWith('mobile_')) {
      patch.hangulLayout = 'pc_2beolsik';
      if (evalStudy.settings.posture.startsWith('hand_') || evalStudy.settings.posture === 'desk_phone_flat') {
        patch.posture = 'desk_keyboard';
      }
    }
    evalStudy.updateSettings(patch);
  });

  $('eval-hangul-layout').addEventListener('change', (e) => {
    evalStudy.updateSettings({ hangulLayout: e.target.value });
  });

  $('eval-posture').addEventListener('change', (e) => {
    evalStudy.updateSettings({ posture: e.target.value });
  });

  $('eval-session-slot').addEventListener('change', (e) => {
    evalStudy.updateSettings({ sessionSlot: e.target.value });
  });

  $('eval-lock-adapt-chk').addEventListener('change', (e) => {
    evalStudy.updateSettings({ lockAdaptation: e.target.checked });
  });

  $('eval-raw-consent-chk').addEventListener('change', (e) => {
    evalStudy.updateSettings({ rawConsent: e.target.checked });
  });

  $('eval-string-calib-mode').addEventListener('change', (e) => {
    evalStudy.updateSettings({ stringCalibMode: e.target.value });
  });

  $('eval-motion-btn').addEventListener('click', async () => {
    const status = await evalStudy.motion.requestAndStart();
    if (status.permissionState === 'unsupported') {
      const httpsUrl = evalStudy.hubInfo?.urls?.https;
      if (typeof window !== 'undefined' && window.isSecureContext === false && httpsUrl) {
        setFeedback(
          $('eval-action-feedback'),
          `iOS Safari 등 모바일 브라우저는 HTTPS에서만 모션 센서를 허용합니다. ${httpsUrl} 로 이동합니다...`
        );
        window.location.href = httpsUrl;
        return;
      }
      setFeedback($('eval-action-feedback'), '현재 브라우저는 DeviceMotion 이벤트를 제공하지 않습니다(스마트폰 HTTPS에서 사용).');
    } else if (status.permissionState === 'denied') {
      setFeedback($('eval-action-feedback'), '모션 센서 권한이 거부되었습니다. 브라우저 사이트 설정에서 동작 및 방향 접근을 허용해 주세요.', 'error');
    } else {
      setFeedback($('eval-action-feedback'), '스마트폰 모션 센서 수집을 시작했습니다. 기기를 움직이면 실제 샘플링 주기(Hz)가 표시됩니다.', 'success');
    }
    renderEvalStudyPanel();
  });

  $('eval-export-jsonl-btn').addEventListener('click', () => {
    const content = evalStudy.toJsonl();
    if (!content) return;
    const blob = new Blob([content], { type: 'application/x-ndjson;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const pid = (evalStudy.settings.participantId || 'P01').replace(/[^A-Za-z0-9_-]/g, '');
    const dateStr = detectSystemEnvironment().localDate;
    const kb = evalStudy.settings.keyboardType || 'device';
    a.href = url;
    a.download = `keyprint-eval-${pid}-${kb}-${dateStr}.jsonl`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setFeedback($('eval-action-feedback'), `총 ${evalStudy.records.length}건의 기록을 JSONL 파일로 내보냈습니다.`, 'success');
  });

  $('eval-import-jsonl-input').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    let totalAdded = 0;
    let totalSkipped = 0;
    try {
      for (const f of files) {
        const txt = await f.text();
        const r = await evalStudy.importJsonlText(txt);
        totalAdded += r.added;
        totalSkipped += r.skipped;
      }
      setFeedback(
        $('eval-action-feedback'),
        `파일 ${files.length}개 병합 완료: 신규 ${totalAdded}건 추가 (중복·무효 ${totalSkipped}건 건너뜀, 총 ${evalStudy.records.length}건).`,
        'success'
      );
    } catch (err) {
      setFeedback($('eval-action-feedback'), `JSONL 병합 오류: ${err?.message || '형식 확인 필요'}`, 'error');
    } finally {
      e.target.value = '';
    }
  });

  $('eval-clear-logs-btn').addEventListener('click', async () => {
    await evalStudy.clearAllLogs();
    setFeedback($('eval-action-feedback'), '브라우저에 저장된 실측 평가 기록을 비웠습니다.');
  });

  const hubSyncBtn = $('eval-hub-sync-btn');
  if (hubSyncBtn) {
    hubSyncBtn.addEventListener('click', async () => {
      const res = await evalStudy.syncWithHub();
      if (res.ok) {
        setFeedback(
          $('eval-action-feedback'),
          `노트북 로컬 허브 동기화 완료: 총 ${res.hubTotal}건 (${evalStudy.hubInfo?.logFileRel || 'field_study_merged.jsonl'})`,
          'success'
        );
      } else {
        setFeedback($('eval-action-feedback'), '로컬 허브 서버에 연결하지 못했습니다.', 'error');
      }
    });
  }

  $('guard-exp-preset-select').addEventListener('change', (e) => {
    const chosen = e.target.value;
    if (!chosen) return;
    const targetInput = $('guard-enroll-pw-target');
    targetInput.value = chosen;
    targetInput.dispatchEvent(new Event('input', { bubbles: true }));
    $('guard-enroll-input').focus();
  });
}

// Expose programmatic helper for E2E verification without adding UI clutter
window.__keyprintTestApi = {
  engine,
  roomSync,
  evalStudy,
  state,
  isSampleName,
  getActiveVerdictForViz,
};

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const initialTabParam = params.get('tab');
  const initialRoom = getInitialRoomCode();
  $('room-code-input').value = initialRoom;
  $('room-pill-label').textContent = '이 브라우저만 · 준비 중';
  $('modal-room-code').textContent = '이 브라우저만';

  const hadSavedEvalSettings = evalStudy.hasStoredSettings();
  await evalStudy.init();
  if (params.get('eval') === '1' || FIELD_HUB_BOOT) {
    state.evalPanelOpen = true;
    if (params.get('eval') === '1' || !hadSavedEvalSettings) {
      evalStudy.updateSettings({ enabled: true });
    }
  }

  bindEvents();
  renderEvalStudyPanel();
  if (params.get('lang') === 'ko') switchLanguage('ko', { deferPrompt: true });
  if (params.has('room') && !FIELD_HUB_BOOT) {
    $('room-dialog').showModal();
    setFeedback($('room-feedback'), `#${initialRoom} 초대 링크입니다. 공개 공유에 동의하고 입장해야 연결됩니다.`);
  }
  if (initialTabParam === 'principles' || initialTabParam === 'how') {
    switchMainMode('principles', { syncUrl: true });
  } else if (initialTabParam === 'guard') {
    switchMainMode('guard', { syncUrl: true });
  } else {
    ensureActiveArenaSentence();
  }

  // Initialize ONNX engine FIRST and independently of any network/WebSocket error
  try {
    await engine.init();
    if (engine.calibrationStudy?.public_experiment_strings) {
      evalStudy.setPublicStrings(engine.calibrationStudy.public_experiment_strings);
      populateGuardPresetSelect();
    }
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
    renderEvalStudyPanel();
  } catch (err) {
    state.ready = false;
    state.loadError = err?.message || '모델 로드 실패';
    $('engine-pill').classList.add('is-error');
    $('engine-status-text').textContent = '모델 로드 실패';
    console.error(err);
  }

  // Connect RoomProfileSync safely even if window.WebSocket throws synchronously
  try {
    if (FIELD_HUB_BOOT) {
      const hubRoom = sanitizeRoomCode(params.get('room') || FIELD_HUB_BOOT.defaultRoom || 'LOCALSTUDY');
      $('room-code-input').value = hubRoom;
      roomSync.connect(hubRoom, { shared: true });
    } else {
      roomSync.connect('LOCAL');
    }
  } catch (syncErr) {
    console.warn('Room sync fallback to local mode:', syncErr);
  }

  if (FIELD_HUB_BOOT) {
    evalStudy.connectLocalHub().then(() => {
      renderEvalStudyPanel();
    });
  }

  resetArenaEnrollProgress();
  resetArenaBlindProgress();
  renderAllRosters();
  renderEvalStudyPanel();

  if (state.ready) {
    try {
      if (!state.viz.lastQuery && state.viz.capture?.stroke) {
        const scheduledSeq = state.querySeq;
        const embeddings = await engine.embedAll(state.viz.capture.stroke);
        if (state.querySeq !== scheduledSeq) return;
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
