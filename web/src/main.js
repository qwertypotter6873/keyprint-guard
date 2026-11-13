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
  mode: 'arena', // 'arena' | 'guard' | 'principles'
  arenaSub: 'enroll', // 'enroll' | 'blind'
  guardSub: 'verify', // 'verify' | 'enroll'
  lang: 'ko', // 'ko' | 'en'
  thresholdOffset: 0, // 여유(-) / 표준(0) / 엄격(+) relative to the validation-calibrated threshold
  showPasswordPlain: true,

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
    lastQuery: null, // { embedding, label, domain }
  },
};

const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
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
  return ` · 주의: ${INPUT_MODE_LABEL[tMode]}로 등록된 프로필을 ${INPUT_MODE_LABEL[inputMode]}로 비교 중(정확도 저하 가능)`;
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
  el.textContent = message || '';
  el.className = 'feedback-line' + (kind ? ` is-${kind}` : '');
}

function renderStepDots(containerEl, labelEl, completedCount, totalCount, prefixText) {
  labelEl.textContent = `${prefixText} ${Math.min(completedCount + 1, totalCount)} / ${totalCount}`;
  const dots = [];
  for (let i = 0; i < totalCount; i++) {
    let cls = 'dot';
    if (i < completedCount) cls += ' is-done';
    else if (i === completedCount) cls += ' is-current';
    dots.push(`<span class="${cls}"></span>`);
  }
  containerEl.innerHTML = dots.join('');
}

// ================= ROOM SYNC SETUP =================
const roomSync = new RoomProfileSync({
  onChange: (profiles) => {
    state.profiles = profiles;
    ensureSelectedGuardAccount();
    renderAllRosters();
    renderModalProfiles();
    if (state.mode === 'principles') {
      renderPrinciplesView();
    }
  },
  onStatusChange: ({ status, roomCode, peerCount, profileCount }) => {
    const syncDot = $('sync-dot');
    syncDot.className =
      'sync-dot ' +
      (status === 'online' ? 'is-online' : status === 'local' ? 'is-local' : 'is-connecting');

    const peerLabel = status === 'online' ? `기기 ${peerCount}대` : '로컬';
    $('room-pill-label').textContent = `방 #${roomCode} · ${peerLabel} · 프로필 ${profileCount}명`;

    $('modal-room-code').textContent = `#${roomCode}`;
    $('modal-sync-badge').textContent =
      status === 'online'
        ? `실시간 연결됨 · 현재 기기 ${peerCount}대 접속`
        : status === 'local'
          ? '로컬 브라우저 저장 모드'
          : '실시간 서버 연결 중…';
  },
});

function getArenaTemplateForProfile(profile, lang) {
  if (!profile.arena) return null;
  if (profile.arena[lang]) return profile.arena[lang];
  if (profile.isSample) {
    return profile.arena.en || profile.arena.ko || null;
  }
  return profile.arena.ko || profile.arena.en || null;
}

function getArenaCandidates() {
  return state.profiles
    .map((p) => {
      const tmpl = getArenaTemplateForProfile(p, state.lang);
      return tmpl ? { id: p.id, name: p.name, color: p.color, isSample: p.isSample, template: tmpl } : null;
    })
    .filter(Boolean);
}

function getGuardAccounts() {
  return state.profiles.filter((p) => p.password && p.password.centroid);
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
    arenaList.innerHTML = `<span class="roster-empty">아직 없습니다 — 아래에서 ${TARGET_ARENA_SHOTS}문장을 연습해 첫 프로필을 등록하세요</span>`;
  } else {
    arenaList.innerHTML = cands
      .map((c) => {
        const initial = escapeHtml(c.name.trim().charAt(0) || '?');
        const cnt = c.template.count || 1;
        return `<button type="button" class="user-chip" data-arena-pick="${escapeHtml(c.name)}" title="클릭하면 이 이름으로 추가 문장을 연습할 수 있습니다">
          <span class="chip-avatar" style="background:${c.color}">${initial}</span>
          <span>${escapeHtml(c.name)}</span>
          <span class="chip-count">${cnt}문장</span>
        </button>`;
      })
      .join('');

    arenaList.querySelectorAll('[data-arena-pick]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const pickedName = btn.getAttribute('data-arena-pick');
        if (state.arenaSub !== 'enroll') switchArenaSub('enroll');
        $('arena-name-input').value = pickedName;
        resetArenaEnrollProgress();
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
        const initial = escapeHtml(acc.name.trim().charAt(0) || '?');
        const selected = acc.id === state.selectedGuardId && state.guardSub === 'verify';
        return `<button type="button" class="user-chip ${selected ? 'is-selected' : ''}" data-guard-id="${escapeHtml(acc.id)}">
          <span class="chip-avatar" style="background:${acc.color || '#18181b'}">${initial}</span>
          <span>${escapeHtml(acc.name)}</span>
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

  $('guard-target-name').textContent = targetAcc.name;
  const pw = targetAcc.password.target || '';
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
      const initial = escapeHtml(p.name.trim().charAt(0) || '?');
      const badges = [];
      if (p.arena?.ko) badges.push(`한글 ${p.arena.ko.count}문장`);
      if (p.arena?.en) badges.push(`영문 ${p.arena.en.count}문장`);
      if (p.password) badges.push(`암호(${p.password.count}회)`);
      const refTmpl = p.arena?.ko || p.arena?.en || p.password;
      const behTxt = refTmpl ? behaviorSummaryText(refTmpl.behavior) : '';
      const modeTxt = refTmpl?.inputMode && refTmpl.inputMode !== 'keyboard' ? ` · ${INPUT_MODE_LABEL[refTmpl.inputMode]}` : '';
      const statsTxt = refTmpl
        ? `평균 ${refTmpl.cpm || 0} 타/분 · 홀드 ${refTmpl.holdMs || 0}ms · 롤오버 ${refTmpl.rolloverPct || 0}%${modeTxt}${behTxt ? ` · ${behTxt}` : ''}`
        : '';

      return `<li class="profile-row">
        <div class="profile-row-left">
          <span class="chip-avatar" style="background:${p.color || '#2563eb'};width:26px;height:26px;font-size:0.78rem">${initial}</span>
          <div>
            <div><strong>${escapeHtml(p.name)}</strong> <span class="chip-count">${badges.join(' · ')}</span></div>
            <div class="profile-meta-line">${escapeHtml(statsTxt)}</div>
          </div>
        </div>
        <button type="button" class="danger-sm-btn" data-del-id="${escapeHtml(p.id)}">삭제</button>
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
  $('arena-blind-need').textContent = `문장 ${TARGET_BLIND_SHOTS}개`;
}

function resetArenaEnrollProgress() {
  const typedName = $('arena-name-input').value.trim();
  state.arenaEnrollBuffer = { name: typedName, lang: state.lang, items: [] };
  renderStepDots($('arena-step-dots'), $('arena-step-text'), 0, TARGET_ARENA_SHOTS, '문장');
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
  focusActiveArenaInput();
}

// ================= KEYSTROKE RECORDERS =================
const arenaEnrollRecorder = new SmartKeystrokeRecorder($('arena-enroll-input'), {
  onLiveUpdate: (live) => {
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
  onSubmit: () => handleGuardVerifySubmit(),
});

const guardEnrollRecorder = new SmartKeystrokeRecorder($('guard-enroll-input'), {
  onSubmit: () => handleGuardEnrollSubmit(),
});

// ================= ACTION HANDLERS =================
async function handleArenaEnrollSubmit() {
  if (!state.ready) return;
  const name = $('arena-name-input').value.trim();
  if (!name) {
    setFeedback($('arena-enroll-feedback'), '참가자 이름을 먼저 입력해 주세요.', 'error');
    $('arena-name-input').focus();
    return;
  }

  // If name changed mid-buffer, start fresh buffer for the new name
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
  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: `내 실시간 타건 · '${name}' 프로필 등록 문장`,
  };
  state.viz.lastQuery = {
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

  // If adding to an existing profile that already has a template in this language, commit immediately per sentence!
  if (existing && existing.arena?.[state.lang]) {
    let mergedTmpl = existing.arena[state.lang];
    for (const item of state.arenaEnrollBuffer.items) {
      mergedTmpl = mergeTemplateSample(mergedTmpl, item);
    }
    roomSync.upsertProfile({
      ...existing,
      arena: {
        ...(existing.arena || {}),
        [state.lang]: mergedTmpl,
      },
    });
    state.arenaEnrollBuffer.items = [];
    dealArenaEnrollSentence();
    setFeedback(
      $('arena-enroll-feedback'),
      `'${name}' 프로필에 문장이 추가 반영되었습니다 (총 ${mergedTmpl.count}문장).`,
      'success'
    );
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

  // Completed the enrolment sentences -> save & broadcast profile!
  let tmpl = existing?.arena?.[state.lang] || null;
  for (const item of state.arenaEnrollBuffer.items) {
    tmpl = mergeTemplateSample(tmpl, item);
  }

  const profileObj = existing
    ? {
        ...existing,
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

  // Reset for the next person
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
  if (!state.ready) return;
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
    // Two sentences carry about twice the rhythm evidence: identify only after the last one (README §5.6)
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
  resetArenaBlindProgress();

  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: '내 실시간 타건 · 누가 쳤을까? 판별 문장',
  };
  state.viz.mapDomain = 'arena';
  state.viz.lastQuery = {
    embedding,
    embeddings: { sentence: embedding },
    label: '방금 판별 타건',
    domain: 'arena',
  };
  const res = engine.identifyTypist(embedding, candidates, {
    behavior,
    inputMode: inputMode === 'mixed' ? 'touch' : inputMode,
  });

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

  // Immediately prepare a fresh unseen sentence for the next round
  dealArenaBlindSentence();
}

function renderArenaResult(res, stats) {
  const card = $('arena-result-card');
  card.hidden = false;

  const pct = Math.round(res.winner.similarity * 1000) / 10;
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
      const sPct = Math.max(4, Math.min(100, Math.round(r.similarity * 100)));
      const fillStyle =
        idx === 0 && res.identified
          ? `width:${sPct}%;background:${r.color || '#059669'}`
          : `width:${sPct}%;background:#9ca3af`;
      return `<div class="rank-item">
        <span class="rank-name">${idx + 1}. ${escapeHtml(r.name)}</span>
        <div class="rank-track"><div class="rank-fill" style="${fillStyle}"></div></div>
        <span class="rank-val">${(r.similarity * 100).toFixed(1)}%</span>
      </div>`;
    })
    .join('');

  const editTxt =
    stats.backspaceCount > 0 ? `백스페이스 ${stats.backspaceCount}회 자동 보정` : '무수정 타이핑';
  const modeTxt = stats.inputMode && stats.inputMode !== 'keyboard' ? ` · ${INPUT_MODE_LABEL[stats.inputMode]}(누름 시간 대체)` : '';
  const sentTxt = stats.sentences > 1 ? `문장 ${stats.sentences}개 · ` : '';
  $('arena-result-meta').textContent = `${sentTxt}${stats.cpm} 타/분 · 정확도 ${stats.accuracy}% · ${editTxt} (유효 전이 ${stats.cleanDigraphs}구간)${modeTxt}${behaviorNote(res.winner)}${deviceMismatchNote(res.winner.template, stats.inputMode)} · 추론 ${stats.latencyMs}ms`;
}

async function handleGuardEnrollSubmit() {
  if (!state.ready) return;
  const name = $('guard-enroll-name').value.trim();
  const targetPw = $('guard-enroll-pw-target').value;

  if (!name) {
    setFeedback($('guard-enroll-feedback'), '사용자 이름을 먼저 입력해 주세요.', 'error');
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

  // Password embedding (9ch) + 17ch embedding of the same stroke: scored together (dualPasswordScore)
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

  // Merge 4 password strokes into profile
  let pwTmpl = null;
  for (const item of state.pwEnrollBuffer.items) {
    pwTmpl = mergeTemplateSample(pwTmpl, item);
  }
  pwTmpl.target = targetPw;

  const existing = roomSync.findProfileByName(name);
  const saved = roomSync.upsertProfile(
    existing
      ? { ...existing, password: pwTmpl }
      : {
          id: `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
          name,
          color: pickProfileColor(name),
          isSample: false,
          password: pwTmpl,
          arena: {},
        }
  );

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
  if (!state.ready) return;
  const accounts = getGuardAccounts();
  if (accounts.length === 0) return;

  const targetAcc = accounts.find((a) => a.id === state.selectedGuardId) || accounts[0];
  const expectedPw = targetAcc.password.target;

  const capture = guardVerifyRecorder.extractCleanStroke();
  const check = validatePasswordCapture(capture, expectedPw, { minLength: 1 });

  $('guard-verify-input').value = '';
  guardVerifyRecorder.reset();

  if (!check.ok) {
    setFeedback($('guard-verify-feedback'), check.message, 'error');
    return;
  }

  setFeedback($('guard-verify-feedback'), '');
  const both = await engine.embedAll(capture.stroke);
  const embedding = both.password;
  const embedding17 = both.sentence;
  const latencyMs = both.latencyMs;
  state.viz.capture = {
    ...capture,
    isPreset: false,
    sourceLabel: `내 실시간 타건 · '${targetAcc.name}' 암호 검증`,
  };
  state.viz.mapDomain = 'guard';
  state.viz.lastQuery = {
    embedding,
    embeddings: { password: embedding },
    label: '방금 입력한 암호',
    domain: 'guard',
  };
  const scoreOpts = {
    domain: 'password',
    thresholdOffset: state.thresholdOffset,
    behavior: capture.behavior,
    inputMode: capture.inputMode,
    embedding17,
    capitalMethods: capture.capitalMethods,
  };
  const res = engine.scoreTemplate(embedding, targetAcc.password, scoreOpts);
  const threshold = res.threshold;

  // Template adaptation: a confident accept joins the template (last 12 kept), so the owner is not
  // rejected once they get fluent at a new password (README §5.6). Sample accounts stay fixed, and a
  // template only learns from the device type it was enrolled on.
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
    roomSync.upsertProfile({ ...targetAcc, password: { ...merged, target: targetAcc.password.target } });
    adapted = true;
  }
  const touchReference = capture.inputMode !== 'keyboard' && engine.passwordTouchReferenceOnly();

  // Check if another registered user's rhythm matches closer when blocked
  let closerOther = null;
  if (!res.accepted && accounts.length > 1) {
    const others = accounts
      .filter((a) => a.id !== targetAcc.id)
      .map((a) => ({
        name: a.name,
        sim: engine.scoreTemplate(embedding, a.password, scoreOpts).similarity,
      }))
      .sort((a, b) => b.sim - a.sim);
    if (others[0] && others[0].sim > res.similarity + 0.08 && others[0].sim >= 0.45) {
      closerOther = others[0];
    }
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
    // Phones expose no key-hold timing: ~9 intervals of a password cannot separate people reliably
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
  const adaptTxt = adapted ? ` · 등록 리듬 갱신(최근 ${Math.min(12, (targetAcc.password.count || 0) + 1)}회 반영)` : '';
  $('guard-result-meta').textContent = `${capture.cpm} 타/분 · ${holdTxt} · ${editTxt}${behaviorNote(res)}${deviceMismatchNote(targetAcc.password, capture.inputMode)}${adaptTxt} · 추론 ${latencyMs}ms`;
}

// ================= MODE C: VISUAL EXPLAINER & PROFILE FORMATION =================
let vizEmbedTimer = null;

const vizSandboxRecorder = new SmartKeystrokeRecorder($('viz-sandbox-input'), {
  onLiveUpdate: () => {
    if (vizSandboxRecorder.stack.length < 2) return;
    const cap = vizSandboxRecorder.extractCleanStroke({ resolvePending: false });
    state.viz.capture = {
      ...cap,
      isPreset: false,
      sourceLabel: '내 실시간 타건 (입력 즉시 갱신 중)',
    };
    renderVizStep1();

    if (vizEmbedTimer) clearTimeout(vizEmbedTimer);
    if (state.ready && cap.cleanDigraphCount >= 4) {
      vizEmbedTimer = setTimeout(async () => {
        try {
          const embeddings = await engine.embedAll(cap.stroke);
          state.viz.lastQuery = {
            embedding: embeddings.sentence,
            embeddings,
            label: '방금 입력한 내 타건',
            domain: state.viz.mapDomain,
          };
          renderVizStep3();
        } catch {
          // ignore transient short-stroke inference errors
        }
      }, 150);
    }
  },
  onSubmit: async () => {
    const cap = vizSandboxRecorder.extractCleanStroke();
    if (cap.cleanDigraphCount < 2) return;
    state.viz.capture = {
      ...cap,
      isPreset: false,
      sourceLabel: '내 실시간 타건 분석 완료',
    };
    renderVizStep1();
    if (state.ready && cap.cleanDigraphCount >= 4) {
      const embeddings = await engine.embedAll(cap.stroke);
      state.viz.lastQuery = {
        embedding: embeddings.sentence,
        embeddings,
        label: '방금 입력한 내 타건',
        domain: state.viz.mapDomain,
      };
      renderVizStep3();
    }
  },
});

function getVizProfilePool() {
  const realList = state.profiles || [];
  const sampleFallback = engine.config?.mvp_sample_profiles || [];
  if (realList.length >= 2) {
    return { list: realList, usingFallback: false };
  }
  const existingIds = new Set(realList.map((p) => p.id));
  const merged = [...realList, ...sampleFallback.filter((s) => !existingIds.has(s.id))];
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
  const arenaTmpl = profile.arena?.ko || profile.arena?.en || null;
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
    cap.backspaceCount > 0 ? `백스페이스 ${cap.backspaceCount}회 봉합` : '무수정';
  $('viz-step1-stats').textContent = `${cap.cpm} 타/분 · 평균 홀드 ${cap.holdMs}ms · 롤오버 ${cap.rolloverPct}% · ${bsNote} · 전이 ${cap.cleanDigraphCount}구간`;
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

  // Render profile selector chips
  $('viz-profile-chips').innerHTML = list
    .map((p) => {
      const initial = escapeHtml(p.name.trim().charAt(0) || '?');
      const active = p.id === selected.id;
      return `<button type="button" class="user-chip ${active ? 'is-selected' : ''}" data-viz-prof="${escapeHtml(p.id)}">
        <span class="chip-avatar" style="background:${p.color || '#2563eb'}">${initial}</span>
        <span>${escapeHtml(p.name)}</span>
      </button>`;
    })
    .join('');

  $('viz-profile-chips').querySelectorAll('[data-viz-prof]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.viz.selectedProfileId = btn.getAttribute('data-viz-prof');
      state.viz.shotCount = 99; // default to full convergence when switching user
      renderVizStep2();
    });
  });

  const totalEx = Math.max(1, tmpl?.exemplars?.length || 1);
  const clampedShots = Math.max(1, Math.min(state.viz.shotCount, totalEx));

  // Render cumulative shot scrubber buttons (1 -> 2 -> N 완성)
  const stepOptions = [];
  for (let k = 1; k <= Math.min(4, totalEx); k++) {
    const isLast = k === Math.min(4, totalEx);
    const actualK = isLast ? totalEx : k;
    const label = isLast
      ? `${actualK}${unitLabel} 완성`
      : `${k}${unitLabel}`;
    const isAct = isLast ? clampedShots >= k : clampedShots === k;
    stepOptions.push(
      `<button type="button" class="lang-btn ${isAct ? 'is-active' : ''}" data-viz-shots="${actualK}">${label}</button>`
    );
  }
  $('viz-shot-scrubber').innerHTML = stepOptions.join('');
  $('viz-shot-scrubber').querySelectorAll('[data-viz-shots]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.viz.shotCount = Number(btn.getAttribute('data-viz-shots'));
      renderVizStep2();
    });
  });

  const { svg, info } = renderProfileBarcodeSvg(tmpl, {
    color: selected.color || '#2563eb',
    stepCount: clampedShots,
    unitLabel,
  });
  $('viz-step2-svg').innerHTML = svg;

  $('viz-stat-stability').textContent = `${info.stabilityPct.toFixed(1)}%`;
  $('viz-stat-stability-sub').textContent =
    info.activeCount === 1
      ? '단일 샘플 — 문장 고유 편차 포함'
      : `${info.activeCount}개 누적 — 문장 노이즈 상쇄됨`;
  $('viz-stat-coherence').textContent = `${info.pairwiseCosPct.toFixed(1)}%`;
  $('viz-stat-hold').textContent = `${tmpl?.holdMs || 0} ms`;
  $('viz-stat-rollover').textContent = `${tmpl?.rolloverPct || 0}% (${tmpl?.cpm || 0}타/분)`;
}

function renderVizStep3() {
  const isArenaMap = state.viz.mapDomain === 'arena';
  $('viz-map-domain-arena').classList.toggle('is-active', isArenaMap);
  $('viz-map-domain-guard').classList.toggle('is-active', !isArenaMap);

  const { list } = getVizProfilePool();
  const entries = list
    .map((p) => {
      const tmpl = isArenaMap
        ? p.arena?.ko || p.arena?.en || null
        : p.password || null;
      if (!tmpl?.centroid) return null;
      return {
        id: p.id,
        name: p.name,
        color: p.color || '#2563eb',
        isSample: Boolean(p.isSample),
        centroid: tmpl.centroid,
        exemplars: tmpl.exemplars || [],
        template: tmpl,
      };
    })
    .filter(Boolean);

  let queryPoint = null;
  // Sentence and password profiles live in different encoder spaces: only plot a query embedded in this one
  const q = state.viz.lastQuery;
  const qEmb = q?.embeddings ? q.embeddings[isArenaMap ? 'sentence' : 'password'] : q?.embedding;
  if (qEmb && entries.length > 0 && state.ready) {
    const scored = engine.identifyTypist(qEmb, entries, {
      domain: isArenaMap ? 'sentence' : 'password',
      openSetThreshold: isArenaMap ? engine.arenaThreshold() : guardThreshold('keyboard'),
    });
    if (scored?.winner) {
      queryPoint = {
        embedding: qEmb,
        label: state.viz.lastQuery.label || '방금 입력한 타건',
        winnerId: scored.winner.id,
        winnerName: scored.winner.name,
        similarityPct: (scored.winner.similarity * 100).toFixed(1),
        identified: scored.identified,
      };
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
  renderVizStep1();
  if (state.ready) {
    const embeddings = await engine.embedAll(cap.stroke);
    state.viz.lastQuery = {
      embedding: embeddings.sentence,
      embeddings,
      label: mode === 'slow' ? '예시 타건 (저속)' : '예시 타건 (고속)',
      domain: state.viz.mapDomain,
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
    const val = $('arena-name-input').value.trim();
    if (val !== state.arenaEnrollBuffer.name) {
      resetArenaEnrollProgress();
    }
  });

  $('guard-enroll-pw-target').addEventListener('input', () => {
    state.pwEnrollBuffer = {
      name: $('guard-enroll-name').value.trim(),
      targetPw: $('guard-enroll-pw-target').value,
      items: [],
    };
    renderStepDots($('guard-step-dots'), $('guard-step-text'), 0, TARGET_PW_SHOTS, '리듬 등록');
  });

  $('guard-toggle-pw-btn').addEventListener('click', () => {
    state.showPasswordPlain = !state.showPasswordPlain;
    renderGuardVerifyState();
  });

  document.querySelectorAll('.tau-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tau-btn').forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      state.thresholdOffset = Number(btn.getAttribute('data-tau-offset'));
    });
  });

  $('guard-go-enroll-btn').addEventListener('click', () => switchGuardSub('enroll'));
  $('guard-load-samples-btn').addEventListener('click', () => {
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
    roomSync.connect(sanitizeRoomCode(raw));
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
};

async function bootstrap() {
  const initialTabParam = new URLSearchParams(window.location.search).get('tab');
  bindEvents();
  dealArenaEnrollSentence();
  dealArenaBlindSentence();

  const initialRoom = getInitialRoomCode();
  roomSync.connect(initialRoom);

  if (initialTabParam === 'principles' || initialTabParam === 'how') {
    switchMainMode('principles', { syncUrl: true });
  } else if (initialTabParam === 'guard') {
    switchMainMode('guard', { syncUrl: true });
  }

  try {
    await engine.init();
    ({ enroll: TARGET_ARENA_SHOTS, query: TARGET_BLIND_SHOTS } = engine.arenaProtocol());
    resetArenaEnrollProgress();
    resetArenaBlindProgress();
    renderAllRosters();
    state.ready = true;
    $('engine-pill').classList.add('is-ready');
    $('engine-status-text').textContent = `ONNX 준비 완료 (${engine.loadTimeMs}ms)`;

    // Compute real ONNX 64-D embedding for the initial preset capture so Step 3 map shows ★ immediately
    if (!state.viz.lastQuery && state.viz.capture?.stroke) {
      const embeddings = await engine.embedAll(state.viz.capture.stroke);
      state.viz.lastQuery = {
        embedding: embeddings.sentence,
        embeddings,
        label: '예시 타건 지문',
        domain: 'arena',
      };
    }
    if (state.mode === 'principles') {
      renderPrinciplesView();
    }
  } catch (err) {
    $('engine-status-text').textContent = '모델 로드 실패';
    console.error(err);
  }
}

bootstrap();
