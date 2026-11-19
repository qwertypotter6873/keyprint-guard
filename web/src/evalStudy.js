/**
 * Cross-device & smartphone evaluation logger, environment metadata manager,
 * counterbalanced 4-device protocol helper, and DeviceMotion sensor collector.
 *
 * Works 100% client-side on static hosting (GitHub Pages): stores records in IndexedDB
 * (with in-memory fallback for headless/private contexts) and exports/merges `.jsonl` files.
 */

import { extractStringCalibrationFeatures } from './features.js';
import { sha256Hex } from './biometrics.js';

export const EVAL_SCHEMA_VERSION = 'kp-eval-log-v1';
const DB_NAME = 'keyprint_eval_study_v1';
const DB_VERSION = 1;
const SETTINGS_KEY = 'kp_eval_study_settings_v1';

export const KEYBOARD_TYPES = [
  { value: 'mechanical_linear', label: '기계식 · 리니어 (적축·은축 등)', group: 'mechanical' },
  { value: 'mechanical_tactile', label: '기계식 · 넌클릭 (갈축 등)', group: 'mechanical' },
  { value: 'mechanical_clicky', label: '기계식 · 클릭 (청축 등)', group: 'mechanical' },
  { value: 'mechanical_other', label: '기계식 · 기타 / 무접점', group: 'mechanical' },
  { value: 'pantograph_external', label: '외장 팬터그래프 (Magic Keyboard 등)', group: 'pantograph_ext' },
  { value: 'pantograph_laptop', label: '노트북 내장 팬터그래프 (MacBook·그램 등)', group: 'pantograph_laptop' },
  { value: 'membrane', label: '외장 멤브레인 키보드', group: 'membrane' },
  { value: 'touch_phone', label: '스마트폰 화면 터치 키보드', group: 'touch' },
  { value: 'touch_tablet', label: '태블릿 화면 터치 키보드', group: 'touch' },
];

export const HANGUL_LAYOUTS = [
  { value: 'pc_2beolsik', label: 'PC 표준 두벌식 / 영문 QWERTY', twoBeolsikCompatible: true },
  { value: 'mobile_qwerty_2beolsik', label: '모바일 쿼티 두벌식 / 영문 QWERTY', twoBeolsikCompatible: true },
  { value: 'mobile_cheonjiin', label: '모바일 천지인 (두벌식 분해 불일치 주의)', twoBeolsikCompatible: false },
  { value: 'mobile_naratgeul', label: '모바일 나랏글·이지 (두벌식 분해 불일치 주의)', twoBeolsikCompatible: false },
  { value: 'mobile_danmoeum', label: '모바일 단모음 (두벌식 분해 불일치 주의)', twoBeolsikCompatible: false },
  { value: 'pc_3beolsik', label: 'PC 세벌식 (두벌식 분해 불일치 주의)', twoBeolsikCompatible: false },
];

export const POSTURE_OPTIONS = [
  { value: 'desk_keyboard', label: '책상 거치 · 양손 타이핑 (PC/외장)' },
  { value: 'lap_laptop', label: '무릎/이동 · 노트북 타이핑' },
  { value: 'hand_two_thumbs', label: '스마트폰 손에 들고 양손 엄지' },
  { value: 'hand_one_finger', label: '스마트폰 한 손 파지 · 엄지/검지' },
  { value: 'desk_phone_flat', label: '스마트폰 책상 위에 평평히 두고 타건' },
];

export const SESSION_SLOTS = [
  { value: 'day1_slot1', label: '당일 실측 · 1번째 기기' },
  { value: 'day1_slot2', label: '당일 실측 · 2번째 기기' },
  { value: 'day1_slot3', label: '당일 실측 · 3번째 기기' },
  { value: 'day1_slot4', label: '당일 실측 · 4번째 기기' },
  { value: 'followup_cross_day', label: '별도 날짜 재측정 (교차 일자 검증)' },
];

// Balanced 4x4 Williams Latin Square across the 4 primary device families so practice & fatigue cancel out.
const WILLIAMS_4_ORDERS = [
  ['기계식(외장)', '외장 팬터그래프', '스마트폰 터치', '노트북 내장'],
  ['외장 팬터그래프', '노트북 내장', '기계식(외장)', '스마트폰 터치'],
  ['노트북 내장', '스마트폰 터치', '외장 팬터그래프', '기계식(외장)'],
  ['스마트폰 터치', '기계식(외장)', '노트북 내장', '외장 팬터그래프'],
];

// Fallback 24 public experiment passwords (mirrors src/string_calibration.py::PUBLIC_EXPERIMENT_STRINGS)
export const FALLBACK_PUBLIC_STRINGS = [
  { id: 'EXP12_ALT_A', length: 12, category: 'alt_hand', tier: 'core', text: 'Gachon2026pw', label: '12자 · 양손교차·혼합 (핵심)' },
  { id: 'EXP12_REP_A', length: 12, category: 'repeat_heavy', tier: 'core', text: 'CoffeeTree88', label: '12자 · 연속반복 다수 (핵심)' },
  { id: 'EXP12_ALT_B', length: 12, category: 'alt_hand', tier: 'extended', text: 'planetworkbd', label: '12자 · 양손교차 영문' },
  { id: 'EXP12_REP_B', length: 12, category: 'repeat_heavy', tier: 'extended', text: 'summerbreeze', label: '12자 · 복수 자모반복' },
  { id: 'EXP12_LEFT', length: 12, category: 'one_hand', tier: 'extended', text: 'stewardesses', label: '12자 · 왼손 편중형' },
  { id: 'EXP12_RIGHT', length: 12, category: 'one_hand', tier: 'extended', text: 'hilopolimion', label: '12자 · 오른손 편중형' },
  { id: 'EXP12_MIX_A', length: 12, category: 'mixed_alnum', tier: 'extended', text: 'SilverKey402', label: '12자 · 대소문·숫자 A' },
  { id: 'EXP12_MIX_B', length: 12, category: 'mixed_alnum', tier: 'extended', text: 'Harbor97Gate', label: '12자 · 대소문·숫자 B' },
  { id: 'EXP16_ALT_A', length: 16, category: 'alt_hand', tier: 'core', text: 'KeyPrint2026Safe', label: '16자 · 양손교차·혼합 (핵심)' },
  { id: 'EXP16_REP_A', length: 16, category: 'repeat_heavy', tier: 'core', text: 'BookkeeperRoom77', label: '16자 · 연속반복 다수 (핵심)' },
  { id: 'EXP16_ALT_B', length: 16, category: 'alt_hand', tier: 'extended', text: 'formalsignalwork', label: '16자 · 양손교차 영문' },
  { id: 'EXP16_REP_B', length: 16, category: 'repeat_heavy', tier: 'extended', text: 'greenapplecoffee', label: '16자 · 복수 자모반복' },
  { id: 'EXP16_LEFT', length: 16, category: 'one_hand', tier: 'extended', text: 'sweaterdressedat', label: '16자 · 왼손 편중형' },
  { id: 'EXP16_RIGHT', length: 16, category: 'one_hand', tier: 'extended', text: 'minimumopinionlk', label: '16자 · 오른손 편중형' },
  { id: 'EXP16_MIX_A', length: 16, category: 'mixed_alnum', tier: 'extended', text: 'AutumnBridge2048', label: '16자 · 대소문·숫자 A' },
  { id: 'EXP16_MIX_B', length: 16, category: 'mixed_alnum', tier: 'extended', text: 'QuantumShield519', label: '16자 · 대소문·숫자 B' },
  { id: 'EXP20_ALT_A', length: 20, category: 'alt_hand', tier: 'core', text: 'Gachon Rhythm2026Key', label: '20자 · 양손교차·혼합 (핵심)' },
  { id: 'EXP20_REP_A', length: 20, category: 'repeat_heavy', tier: 'core', text: 'SweetCoffeeBalloon99', label: '20자 · 연속반복 다수 (핵심)' },
  { id: 'EXP20_ALT_B', length: 20, category: 'alt_hand', tier: 'extended', text: 'modernvisualworkland', label: '20자 · 양손교차 영문' },
  { id: 'EXP20_REP_B', length: 20, category: 'repeat_heavy', tier: 'extended', text: 'sleeplesswoodlandsee', label: '20자 · 복수 자모반복' },
  { id: 'EXP20_LEFT', length: 20, category: 'one_hand', tier: 'extended', text: 'afterwardsdressedvet', label: '20자 · 왼손 편중형' },
  { id: 'EXP20_RIGHT', length: 20, category: 'one_hand', tier: 'extended', text: 'millionopinionhookup', label: '20자 · 오른손 편중형' },
  { id: 'EXP20_MIX_A', length: 20, category: 'mixed_alnum', tier: 'extended', text: 'Morninglantern735Box', label: '20자 · 대소문·숫자 A' },
  { id: 'EXP20_MIX_B', length: 20, category: 'mixed_alnum', tier: 'extended', text: 'CrystalHorizon842Map', label: '20자 · 대소문·숫자 B' },
];

export function recommendDeviceOrder(participantId) {
  const s = String(participantId || '').trim();
  const m = s.match(/(\d+)/);
  let idx = 0;
  if (m) {
    idx = Math.max(0, parseInt(m[1], 10) - 1) % 4;
  } else if (s) {
    let h = 0;
    for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    idx = h % 4;
  }
  return {
    cohortRow: idx + 1,
    order: WILLIAMS_4_ORDERS[idx],
    formatted: WILLIAMS_4_ORDERS[idx].map((d, i) => `${i + 1}) ${d}`).join(' → '),
  };
}

export function isNon2BeolsikLayout(layoutValue) {
  const item = HANGUL_LAYOUTS.find((l) => l.value === layoutValue);
  return item ? !item.twoBeolsikCompatible : false;
}

function roundNum(val, digits = 5) {
  if (!Number.isFinite(val)) return null;
  const f = 10 ** digits;
  return Math.round(val * f) / f;
}

function medianNum(arr) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

function quantileNum(arr, q) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  return s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

function rmsNum(arr) {
  if (!arr.length) return 0;
  return Math.sqrt(arr.reduce((a, b) => a + b * b, 0) / arr.length);
}

function stdNum(arr) {
  if (arr.length < 2) return 0;
  const mu = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mu) * (b - mu), 0) / arr.length);
}

function estimateTimerResolutionUs() {
  if (typeof performance === 'undefined' || typeof performance.now !== 'function') return null;
  let minStepMs = Infinity;
  let prev = performance.now();
  for (let i = 0; i < 64; i++) {
    const cur = performance.now();
    const diff = cur - prev;
    if (diff > 0 && diff < minStepMs) minStepMs = diff;
    prev = cur;
  }
  if (!Number.isFinite(minStepMs)) return 5; // sub-5us clock did not tick across tight loop
  return Math.max(1, Math.round(minStepMs * 1000));
}

export function detectSystemEnvironment() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : '';
  const touchPoints = typeof navigator !== 'undefined' ? Number(navigator.maxTouchPoints || 0) : 0;

  let os = 'Unknown';
  if (/Android/i.test(ua)) os = 'Android';
  else if (/iPhone|iPad|iPod/i.test(ua) || (/Macintosh/i.test(ua) && touchPoints > 1)) os = 'iOS';
  else if (/Mac OS X|Macintosh/i.test(ua)) os = 'macOS';
  else if (/Windows/i.test(ua)) os = 'Windows';
  else if (/Linux/i.test(ua)) os = 'Linux';

  let browser = 'Other';
  if (/SamsungBrowser\//i.test(ua)) browser = 'SamsungInternet';
  else if (/Edg\//i.test(ua)) browser = 'Edge';
  else if (/Firefox\//i.test(ua) || /FxiOS\//i.test(ua)) browser = 'Firefox';
  else if (/CriOS\//i.test(ua) || (/Chrome\//i.test(ua) && !/Edg\//i.test(ua))) browser = 'Chrome';
  else if (/Safari\//i.test(ua) && !/Chrome\//i.test(ua)) browser = 'Safari';

  const now = new Date();
  const localDate = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
  ].join('-');

  return {
    recordedAt: now.toISOString(),
    localDate,
    os,
    browser,
    maxTouchPoints: touchPoints,
    viewportWidth: typeof window !== 'undefined' ? window.innerWidth : null,
    viewportHeight: typeof window !== 'undefined' ? window.innerHeight : null,
    timerResolutionUs: estimateTimerResolutionUs(),
  };
}

/**
 * Rolling DeviceMotion collector with iOS 13+ permission handshake and window summarization.
 */
export class DeviceMotionCollector {
  constructor({ maxSamples = 2400, onStatusChange = null } = {}) {
    this.maxSamples = maxSamples;
    this.onStatusChange = onStatusChange;
    this.samples = [];
    this.permissionState = 'idle'; // 'idle' | 'unsupported' | 'prompt_required' | 'granted' | 'denied' | 'active'
    this.listening = false;
    this.lastEventPerfMs = null;
    this._boundHandler = (e) => this._onMotion(e);
    this._detectInitialSupport();
  }

  _detectInitialSupport() {
    if (typeof window === 'undefined' || typeof window.DeviceMotionEvent === 'undefined') {
      this.permissionState = 'unsupported';
      return;
    }
    if (typeof window.DeviceMotionEvent.requestPermission === 'function') {
      this.permissionState = 'prompt_required';
    } else {
      this.permissionState = 'idle';
    }
  }

  /** Must be invoked synchronously inside a user click/tap handler for iOS Safari. */
  async requestAndStart() {
    if (typeof window === 'undefined' || typeof window.DeviceMotionEvent === 'undefined') {
      this.permissionState = 'unsupported';
      this._notify();
      return this.getLiveStatus();
    }
    if (typeof window.DeviceMotionEvent.requestPermission === 'function') {
      try {
        const res = await window.DeviceMotionEvent.requestPermission();
        if (res !== 'granted') {
          this.permissionState = 'denied';
          this._notify();
          return this.getLiveStatus();
        }
        this.permissionState = 'granted';
      } catch {
        this.permissionState = 'denied';
        this._notify();
        return this.getLiveStatus();
      }
    } else {
      this.permissionState = 'granted';
    }
    this._attach();
    this._notify();
    return this.getLiveStatus();
  }

  stop() {
    if (this.listening && typeof window !== 'undefined') {
      window.removeEventListener('devicemotion', this._boundHandler);
    }
    this.listening = false;
    if (this.permissionState === 'active') this.permissionState = 'granted';
    this._notify();
  }

  _attach() {
    if (this.listening || typeof window === 'undefined') return;
    window.addEventListener('devicemotion', this._boundHandler, { passive: true });
    this.listening = true;
  }

  /** Allow programmatic injection for E2E/unit verification as well as real browser events. */
  ingestSample({
    tMs = performance.now(),
    ax = null,
    ay = null,
    az = null,
    agx = null,
    agy = null,
    agz = null,
    rAlpha = null,
    rBeta = null,
    rGamma = null,
    reportedIntervalMs = null,
  }) {
    const dt = this.lastEventPerfMs !== null && tMs > this.lastEventPerfMs ? tMs - this.lastEventPerfMs : reportedIntervalMs;
    this.lastEventPerfMs = tMs;
    this.permissionState = 'active';
    this.samples.push({
      tMs,
      ax: Number.isFinite(ax) ? ax : null,
      ay: Number.isFinite(ay) ? ay : null,
      az: Number.isFinite(az) ? az : null,
      agx: Number.isFinite(agx) ? agx : null,
      agy: Number.isFinite(agy) ? agy : null,
      agz: Number.isFinite(agz) ? agz : null,
      rAlpha: Number.isFinite(rAlpha) ? rAlpha : null,
      rBeta: Number.isFinite(rBeta) ? rBeta : null,
      rGamma: Number.isFinite(rGamma) ? rGamma : null,
      dtMs: Number.isFinite(dt) && dt > 0 ? dt : null,
    });
    if (this.samples.length > this.maxSamples) {
      this.samples.splice(0, this.samples.length - this.maxSamples);
    }
  }

  _onMotion(e) {
    const nowPerf = performance.now();
    const tMs =
      typeof e.timeStamp === 'number' && e.timeStamp > 0 && Math.abs(e.timeStamp - nowPerf) < 2000
        ? e.timeStamp
        : nowPerf;
    const acc = e.acceleration || {};
    const accG = e.accelerationIncludingGravity || {};
    const rot = e.rotationRate || {};
    const prevState = this.permissionState;
    this.ingestSample({
      tMs,
      ax: acc.x,
      ay: acc.y,
      az: acc.z,
      agx: accG.x,
      agy: accG.y,
      agz: accG.z,
      rAlpha: rot.alpha,
      rBeta: rot.beta,
      rGamma: rot.gamma,
      reportedIntervalMs: typeof e.interval === 'number' ? e.interval : null,
    });
    if (prevState !== 'active' || this.samples.length % 30 === 0) {
      this._notify();
    }
  }

  _notify() {
    if (this.onStatusChange) this.onStatusChange(this.getLiveStatus());
  }

  getLiveStatus() {
    const recent = this.samples.slice(-60);
    const dts = recent.map((s) => s.dtMs).filter((v) => v !== null && v > 0 && v < 500);
    const medMs = medianNum(dts);
    const hz = medMs && medMs > 0 ? Math.round((1000 / medMs) * 10) / 10 : 0;
    return {
      permissionState: this.permissionState,
      listening: this.listening,
      sampleCount: this.samples.length,
      medianIntervalMs: medMs ? roundNum(medMs, 2) : null,
      effectiveHz: hz || null,
    };
  }

  /** Summarize DeviceMotion over `[startMs - 80, endMs + 120]` and around keystroke downTimes. */
  summarizeWindow(startMs, endMs, strokeDownTimesMs = []) {
    const status = this.getLiveStatus();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || this.samples.length === 0) {
      return {
        permissionState: status.permissionState,
        available: false,
        sampleCount: 0,
      };
    }
    const winLo = startMs - 80;
    const winHi = endMs + 120;
    const win = this.samples.filter((s) => s.tMs >= winLo && s.tMs <= winHi);
    if (win.length < 3) {
      return {
        permissionState: status.permissionState,
        available: false,
        sampleCount: win.length,
      };
    }

    const dts = win.map((s) => s.dtMs).filter((v) => v !== null && v > 0 && v < 500);
    const medDt = medianNum(dts);
    const p90Dt = quantileNum(dts, 0.9);

    // Estimate gravity vector via window mean of accelerationIncludingGravity
    const agValid = win.filter((s) => s.agx !== null && s.agy !== null && s.agz !== null);
    let pitchDeg = null;
    let rollDeg = null;
    let gravityJitterRms = null;
    let gxMean = 0;
    let gyMean = 0;
    let gzMean = 0;
    if (agValid.length >= 3) {
      gxMean = agValid.reduce((a, s) => a + s.agx, 0) / agValid.length;
      gyMean = agValid.reduce((a, s) => a + s.agy, 0) / agValid.length;
      gzMean = agValid.reduce((a, s) => a + s.agz, 0) / agValid.length;
      const gNorm = Math.hypot(gxMean, gyMean, gzMean) || 9.81;
      pitchDeg = (Math.asin(Math.max(-1, Math.min(1, gyMean / gNorm))) * 180) / Math.PI;
      rollDeg = (Math.atan2(-gxMean, -gzMean) * 180) / Math.PI;
      gravityJitterRms = Math.hypot(
        stdNum(agValid.map((s) => s.agx)),
        stdNum(agValid.map((s) => s.agy)),
        stdNum(agValid.map((s) => s.agz))
      );
    }

    // Linear acceleration magnitude (native `acceleration` or high-pass residual from `ag - gMean`)
    const linMags = win
      .map((s) => {
        if (s.ax !== null && s.ay !== null && s.az !== null) {
          return Math.hypot(s.ax, s.ay, s.az);
        }
        if (s.agx !== null && s.agy !== null && s.agz !== null && agValid.length >= 3) {
          return Math.hypot(s.agx - gxMean, s.agy - gyMean, s.agz - gzMean);
        }
        return null;
      })
      .filter((v) => v !== null);

    const gyroMags = win
      .filter((s) => s.rAlpha !== null && s.rBeta !== null && s.rGamma !== null)
      .map((s) => Math.hypot(s.rAlpha, s.rBeta, s.rGamma));

    // Keystroke tap impulse in [-10ms, +45ms] around each stroke downTime
    const impulses = [];
    if (linMags.length >= 3 && Array.isArray(strokeDownTimesMs)) {
      for (const td of strokeDownTimesMs) {
        let peak = null;
        for (const s of win) {
          if (s.tMs >= td - 10 && s.tMs <= td + 45) {
            const m =
              s.ax !== null && s.ay !== null && s.az !== null
                ? Math.hypot(s.ax, s.ay, s.az)
                : s.agx !== null
                  ? Math.hypot(s.agx - gxMean, s.agy - gyMean, s.agz - gzMean)
                  : null;
            if (m !== null && (peak === null || m > peak)) peak = m;
          }
        }
        if (peak !== null) impulses.push(peak);
      }
    }

    const linRms = rmsNum(linMags);
    const gyroRms = rmsNum(gyroMags);
    let inferredPosture = 'unknown';
    if (gravityJitterRms !== null) {
      const flatTilt =
        pitchDeg !== null &&
        Math.abs(pitchDeg) < 12 &&
        (rollDeg === null || Math.abs(rollDeg) < 12 || Math.abs(Math.abs(rollDeg) - 180) < 12);
      inferredPosture = gravityJitterRms < 0.85 && gyroRms < 10.0 && flatTilt ? 'on_desk_flat' : 'in_hand';
    }

    return {
      permissionState: status.permissionState,
      available: true,
      sampleCount: win.length,
      medianIntervalMs: roundNum(medDt, 2),
      p90IntervalMs: roundNum(p90Dt, 2),
      effectiveHz: medDt && medDt > 0 ? roundNum(1000 / medDt, 1) : null,
      pitchDeg: roundNum(pitchDeg, 2),
      rollDeg: roundNum(rollDeg, 2),
      gravityJitterRms: roundNum(gravityJitterRms, 4),
      linearAccelRms: roundNum(linRms, 4),
      linearAccelP95: roundNum(quantileNum(linMags, 0.95), 4),
      gyroRms: roundNum(gyroRms, 3),
      gyroP95: roundNum(quantileNum(gyroMags, 0.95), 3),
      tapImpulseMean: impulses.length ? roundNum(impulses.reduce((a, b) => a + b, 0) / impulses.length, 4) : null,
      tapImpulseStd: impulses.length >= 2 ? roundNum(stdNum(impulses), 4) : null,
      inferredPosture,
    };
  }
}

/**
 * IndexedDB + in-memory evaluation study store & logger.
 */
export class EvalStudyManager {
  constructor({ onChange = null } = {}) {
    this.onChange = onChange;
    this.motion = new DeviceMotionCollector({
      onStatusChange: () => this._emit(),
    });
    this.db = null;
    this.records = [];
    this.enrollEnvs = new Map(); // `${roomCode}::${profileId}::${domain}` -> env snapshot
    this.publicStrings = FALLBACK_PUBLIC_STRINGS;
    this.hubActive = false;
    this.hubInfo = null;
    this.hubSyncError = null;
    this._pullTimer = null;
    this.settings = this._loadSettings();
  }

  hasStoredSettings() {
    try {
      const storage = typeof window !== 'undefined' ? window.localStorage : null;
      return Boolean(storage && storage.getItem(SETTINGS_KEY));
    } catch {
      return false;
    }
  }

  _defaultSettings() {
    const sys = detectSystemEnvironment();
    const isMobile = sys.os === 'iOS' || sys.os === 'Android';
    return {
      enabled: false,
      participantId: 'P01',
      groundTruthRole: 'unspecified', // 'unspecified' | 'outsider' | `enrolled:${profileId}`
      keyboardType: isMobile ? 'touch_phone' : 'mechanical_linear',
      hangulLayout: isMobile ? 'mobile_qwerty_2beolsik' : 'pc_2beolsik',
      posture: isMobile ? 'hand_two_thumbs' : 'desk_keyboard',
      sessionSlot: 'day1_slot1',
      lockAdaptation: true,
      rawConsent: false,
      stringCalibMode: 'auto', // 'auto' | 'force_on' | 'force_off'
      note: '',
    };
  }

  _loadSettings() {
    const base = this._defaultSettings();
    try {
      const storage = typeof window !== 'undefined' ? window.localStorage : null;
      const raw = storage ? storage.getItem(SETTINGS_KEY) : null;
      if (!raw) return base;
      const parsed = JSON.parse(raw);
      return { ...base, ...parsed };
    } catch {
      return base;
    }
  }

  _saveSettings() {
    try {
      const storage = typeof window !== 'undefined' ? window.localStorage : null;
      if (storage) {
        storage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));
      }
    } catch {
      // ignore storage quota errors
    }
  }

  updateSettings(patch) {
    this.settings = { ...this.settings, ...patch };
    this._saveSettings();
    this._emit();
  }

  setPublicStrings(list) {
    if (Array.isArray(list) && list.length > 0) {
      this.publicStrings = list;
      this._emit();
    }
  }

  isPublicExperimentPassword(text) {
    return this.publicStrings.some((item) => item.text === text);
  }

  findPublicPasswordEntry(text) {
    return this.publicStrings.find((item) => item.text === text) || null;
  }

  shouldLockAdaptation() {
    return Boolean(this.settings.enabled && this.settings.lockAdaptation);
  }

  stringCalibrationOverride() {
    if (!this.settings.enabled || this.settings.stringCalibMode === 'auto') return null;
    return this.settings.stringCalibMode === 'force_on';
  }

  async init() {
    if (typeof indexedDB === 'undefined') {
      this._emit();
      return;
    }
    try {
      this.db = await new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains('records')) {
            db.createObjectStore('records', { keyPath: 'recordId' });
          }
          if (!db.objectStoreNames.contains('enroll_envs')) {
            db.createObjectStore('enroll_envs', { keyPath: 'key' });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      await this._hydrateFromDb();
    } catch {
      this.db = null;
    }
    this._emit();
  }

  async _hydrateFromDb() {
    if (!this.db) return;
    const readAll = (storeName) =>
      new Promise((resolve, reject) => {
        const tx = this.db.transaction(storeName, 'readonly');
        const req = tx.objectStore(storeName).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => reject(req.error);
      });
    const [recs, envs] = await Promise.all([readAll('records'), readAll('enroll_envs')]);
    this.records = recs.sort((a, b) => String(a.recordedAt).localeCompare(String(b.recordedAt)));
    this.enrollEnvs.clear();
    for (const item of envs) {
      if (item?.key) this.enrollEnvs.set(item.key, item.env);
    }
  }

  async _putStore(storeName, value) {
    if (!this.db) return;
    await new Promise((resolve, reject) => {
      const tx = this.db.transaction(storeName, 'readwrite');
      tx.objectStore(storeName).put(value);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async clearAllLogs() {
    this.records = [];
    this.enrollEnvs.clear();
    if (this.db) {
      await new Promise((resolve, reject) => {
        const tx = this.db.transaction(['records', 'enroll_envs'], 'readwrite');
        tx.objectStore('records').clear();
        tx.objectStore('enroll_envs').clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    }
    this._emit();
  }

  /**
   * Automatically deletes local IndexedDB evaluation records and enrollment environments
   * recorded under an outdated `{ schemaVersion, preprocessVersion, modelHash }` tuple.
   */
  async purgeOutdatedRecords({ schemaVersion, preprocessVersion, modelHash } = {}) {
    if (!preprocessVersion || !modelHash) return 0;
    const isOutdatedEnv = (env) => {
      if (!env || typeof env !== 'object') return false;
      if (env.modelHash != null && env.modelHash !== modelHash) return true;
      if (env.preprocessVersion != null && env.preprocessVersion !== preprocessVersion) return true;
      if (schemaVersion != null && env.schemaVersion != null && env.schemaVersion !== schemaVersion) return true;
      return false;
    };

    const removedRecordIds = [];
    for (const rec of this.records) {
      if (rec?.recordId && (isOutdatedEnv(rec.env) || isOutdatedEnv(rec.enrollEnv))) {
        removedRecordIds.push(rec.recordId);
      }
    }

    const removedEnvKeys = [];
    const removedProfileIds = new Set();
    for (const [key, env] of this.enrollEnvs.entries()) {
      if (isOutdatedEnv(env)) {
        removedEnvKeys.push(key);
        const parts = String(key).split('::');
        if (parts.length >= 2 && parts[1]) removedProfileIds.add(parts[1]);
      }
    }

    if (removedRecordIds.length === 0 && removedEnvKeys.length === 0) return 0;

    const removedRecordSet = new Set(removedRecordIds);
    this.records = this.records.filter((rec) => !removedRecordSet.has(rec.recordId));
    for (const key of removedEnvKeys) {
      this.enrollEnvs.delete(key);
    }

    if (typeof this.settings.groundTruthRole === 'string' && this.settings.groundTruthRole.startsWith('enrolled:')) {
      const targetId = this.settings.groundTruthRole.slice('enrolled:'.length);
      const stillHasEnv = Array.from(this.enrollEnvs.keys()).some((k) => String(k).split('::')[1] === targetId);
      if (removedProfileIds.has(targetId) && !stillHasEnv) {
        this.settings = { ...this.settings, groundTruthRole: 'unspecified' };
        this._saveSettings();
      }
    }

    if (this.db) {
      try {
        await new Promise((resolve, reject) => {
          const tx = this.db.transaction(['records', 'enroll_envs'], 'readwrite');
          const recStore = tx.objectStore('records');
          const envStore = tx.objectStore('enroll_envs');
          for (const id of removedRecordIds) recStore.delete(id);
          for (const key of removedEnvKeys) envStore.delete(key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
        });
      } catch {
        // Ignore IndexedDB deletion errors in restricted contexts
      }
    }

    this._emit();
    return removedRecordIds.length + removedEnvKeys.length;
  }

  currentEnvSnapshot({ engine, lang, inputMode }) {
    const sys = detectSystemEnvironment();
    const layoutMismatch = lang === 'ko' && isNon2BeolsikLayout(this.settings.hangulLayout);
    return {
      ...sys,
      participantId: String(this.settings.participantId || '').trim() || 'UNSET',
      keyboardType: this.settings.keyboardType,
      hangulLayout: this.settings.hangulLayout,
      layoutMismatch2Beolsik: layoutMismatch,
      posture: this.settings.posture,
      sessionSlot: this.settings.sessionSlot,
      detectedInputMode: inputMode || 'keyboard',
      modelHash: engine?.modelHash || null,
      preprocessVersion: engine?.preprocessVersion || null,
      schemaVersion: engine?.schemaVersion || 4,
      note: String(this.settings.note || '').trim() || null,
    };
  }

  _envKey(roomCode, profileId, domain) {
    return `${roomCode || 'LOCAL'}::${profileId}::${domain}`;
  }

  getEnrollEnv(roomCode, profileId, domain) {
    return this.enrollEnvs.get(this._envKey(roomCode, profileId, domain)) || null;
  }

  _sanitizeRawCapture(capture, { allowKeyIdentity }) {
    if (!this.settings.rawConsent || !capture?.stroke) return null;
    const st = capture.stroke;
    return {
      cpm: capture.cpm,
      holdMs: capture.holdMs,
      rolloverPct: capture.rolloverPct,
      backspaceCount: capture.backspaceCount,
      cleanDigraphCount: capture.cleanDigraphCount,
      inputMode: capture.inputMode,
      capitalMethods: allowKeyIdentity ? capture.capitalMethods || [] : [],
      behavior: capture.behavior || null,
      holds: (st.holds || []).map((v) => (v === null ? null : roundNum(v, 5))),
      holdValid: (st.holdValid || []).map(Boolean),
      hCurr: (st.hCurr || []).map((v) => (v === null ? null : roundNum(v, 5))),
      hNext: (st.hNext || []).map((v) => (v === null ? null : roundNum(v, 5))),
      hvCurr: (st.hvCurr || []).map(Boolean),
      hvNext: (st.hvNext || []).map(Boolean),
      dds: (st.dds || []).map((v) => roundNum(v, 5)),
      uds: (st.uds || []).map((v) => (v === null ? null : roundNum(v, 5))),
      stitchFlags: (st.stitchFlags || []).map(Boolean),
      pairKeys: allowKeyIdentity ? st.pairKeys || [] : null,
      pairKeysRedacted: !allowKeyIdentity,
    };
  }

  _captureMotionWindow(recorder) {
    if (!recorder || !Array.isArray(recorder.stack) || recorder.stack.length === 0) {
      return this.motion.summarizeWindow(NaN, NaN, []);
    }
    const downTimes = recorder.stack.filter((s) => !s.synthetic).map((s) => s.downTime);
    if (downTimes.length === 0) return this.motion.summarizeWindow(NaN, NaN, []);
    const startMs = Math.min(...downTimes);
    const endMs = Math.max(
      ...recorder.stack
        .filter((s) => !s.synthetic)
        .map((s) => (s.upTime !== null && s.upTime !== undefined ? s.upTime : s.downTime))
    );
    return this.motion.summarizeWindow(startMs, endMs, downTimes);
  }

  async recordEnrollment({
    domain, // 'arena' | 'password'
    lang = null,
    roomCode,
    profile,
    template,
    captures = [],
    recorders = [],
    motionSnapshots = [],
    prompts = [],
    passwordText = null,
    engine,
  }) {
    const dominantMode = template?.inputMode || captures[0]?.inputMode || 'keyboard';
    const env = this.currentEnvSnapshot({ engine, lang, inputMode: dominantMode });
    const domKey = domain === 'password' ? 'password' : `arena_${lang || 'en'}`;
    const envKey = this._envKey(roomCode, profile.id, domKey);
    this.enrollEnvs.set(envKey, env);
    await this._putStore('enroll_envs', { key: envKey, env });

    if (!this.settings.enabled) {
      await this._pushToHub([]);
      return null;
    }

    const pubEntry = domain === 'password' ? this.findPublicPasswordEntry(passwordText) : null;
    const allowKeyIdentity = domain === 'arena' || Boolean(pubEntry);
    const pwHash =
      domain === 'password' && passwordText ? (await sha256Hex(`kp-pw:${passwordText}`)).slice(0, 12) : null;
    const strFx =
      domain === 'password' && passwordText && engine?.config?.keyboard_layout
        ? extractStringCalibrationFeatures(
            passwordText,
            engine.config.keyboard_layout,
            engine.calibrationStudy?.digraph_table || {},
            engine.calibrationStudy?.support_kappa ?? 40.0
          )
        : null;

    const motionWindows = [
      ...motionSnapshots.filter((m) => m?.available),
      ...recorders.map((r) => this._captureMotionWindow(r)).filter((m) => m?.available),
    ];
    const record = {
      schema: EVAL_SCHEMA_VERSION,
      recordId: `ev_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      eventType: 'enroll',
      recordedAt: env.recordedAt,
      domain,
      lang: domain === 'arena' ? lang : null,
      roomCode: roomCode || 'LOCAL',
      profileId: profile.id,
      profileName: profile.name,
      shotCount: template?.count || captures.length,
      env,
      templateSummary: {
        count: template?.count || 0,
        cpm: template?.cpm || 0,
        holdMs: template?.holdMs || 0,
        rolloverPct: template?.rolloverPct || 0,
        inputMode: template?.inputMode || dominantMode,
        centroid:
          this.settings.rawConsent && Array.isArray(template?.centroid)
            ? template.centroid.map((v) => roundNum(v, 5))
            : null,
        exemplars:
          this.settings.rawConsent && Array.isArray(template?.exemplars)
            ? template.exemplars.map((ex) => ex.map((v) => roundNum(v, 5)))
            : null,
        centroid17:
          this.settings.rawConsent && Array.isArray(template?.enc17?.centroid)
            ? template.enc17.centroid.map((v) => roundNum(v, 5))
            : null,
        exemplars17:
          this.settings.rawConsent && Array.isArray(template?.enc17?.exemplars)
            ? template.enc17.exemplars.map((ex) => ex.map((v) => roundNum(v, 5)))
            : null,
        behavior: this.settings.rawConsent ? template?.behavior || null : null,
      },
      passwordMeta:
        domain === 'password'
          ? {
              length: [...String(passwordText || '')].length,
              publicExperimentId: pubEntry?.id || null,
              publicCategory: pubEntry?.category || null,
              publicTier: pubEntry?.tier || null,
              publicText: pubEntry ? pubEntry.text : null,
              passwordHash12: pwHash,
              isPublicAllowlist: Boolean(pubEntry),
              stringFeatures: strFx ? strFx.features.map((v) => roundNum(v, 5)) : null,
              transitionSupport: strFx ? roundNum(strFx.support, 5) : null,
            }
          : null,
      prompts: domain === 'arena' ? prompts : null,
      deviceMotion: motionWindows.length > 0 ? motionWindows[motionWindows.length - 1] : null,
      rawCaptures: this.settings.rawConsent
        ? captures.map((c) => this._sanitizeRawCapture(c, { allowKeyIdentity }))
        : null,
    };

    this.records.push(record);
    await this._putStore('records', record);
    await this._pushToHub([record]);
    this._emit();
    return record;
  }

  async recordArenaQuery({
    lang,
    roomCode,
    candidates,
    res,
    items, // [{ embedding, capture, accuracy, prompt }]
    combinedEmbedding,
    mergedBehavior,
    inputMode,
    latencyMs,
    recorder,
    motionSnapshot = null,
    engine,
  }) {
    if (!this.settings.enabled) return null;
    const env = this.currentEnvSnapshot({ engine, lang, inputMode });
    const gt = this.settings.groundTruthRole || 'unspecified';
    let actualRole = 'unspecified';
    let actualProfileId = null;
    let actualProfileName = null;
    if (gt === 'outsider') {
      actualRole = 'outsider';
    } else if (gt.startsWith('enrolled:')) {
      actualRole = 'enrolled';
      actualProfileId = gt.slice('enrolled:'.length);
      actualProfileName = candidates.find((c) => c.id === actualProfileId)?.name || null;
    }

    const winnerId = res.winner?.id || null;
    const top1Correct =
      actualRole === 'enrolled' ? winnerId === actualProfileId : actualRole === 'outsider' ? null : null;
    const decisionCorrect =
      actualRole === 'enrolled'
        ? Boolean(res.identified && winnerId === actualProfileId)
        : actualRole === 'outsider'
          ? !res.identified
          : null;

    const domKey = `arena_${lang || 'en'}`;
    if (this.hubActive && candidates.some((c) => !this.getEnrollEnv(roomCode, c.id, domKey))) {
      await this._pullFromHubOnce();
    }
    const winnerEnrollEnv = winnerId ? this.getEnrollEnv(roomCode, winnerId, domKey) : null;
    const actualEnrollEnv = actualProfileId ? this.getEnrollEnv(roomCode, actualProfileId, domKey) : null;
    const crossDevice =
      actualEnrollEnv && actualEnrollEnv.keyboardType
        ? actualEnrollEnv.keyboardType !== env.keyboardType
        : winnerEnrollEnv && winnerEnrollEnv.keyboardType
          ? winnerEnrollEnv.keyboardType !== env.keyboardType
          : null;

    const motionSummary = motionSnapshot || this._captureMotionWindow(recorder);
    const record = {
      schema: EVAL_SCHEMA_VERSION,
      recordId: `ev_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      eventType: 'query',
      domain: 'arena',
      lang,
      roomCode: roomCode || 'LOCAL',
      recordedAt: env.recordedAt,
      queryEnv: env,
      groundTruth: {
        role: actualRole,
        profileId: actualProfileId,
        profileName: actualProfileName,
        enrollEnv: actualEnrollEnv,
      },
      crossDevice,
      candidateCount: candidates.length,
      verdict: {
        winnerId,
        winnerName: res.winner?.name || null,
        winnerEnrollEnv,
        top1Score: roundNum(res.winner?.similarity, 5),
        top1EmbeddingScore: roundNum(res.winner?.embeddingSimilarity, 5),
        top1BehaviorLlr: roundNum(res.winner?.behaviorLlr, 5),
        behaviorWeight: res.winner?.behaviorWeight ?? 0,
        leadGap: Number.isFinite(res.lead) ? roundNum(res.lead, 5) : null,
        openSetThreshold: roundNum(res.openSetThreshold, 5),
        marginThreshold: roundNum(res.margin, 5),
        accepted: Boolean(res.identified),
        rejectedByMargin: Boolean(res.rejectedByMargin),
        top1Correct,
        decisionCorrect,
        latencyMs,
      },
      rankings: res.rankings.map((r) => ({
        id: r.id,
        name: r.name,
        rank: r.rank,
        similarity: roundNum(r.similarity, 5),
        embeddingSimilarity: roundNum(r.embeddingSimilarity, 5),
        rawSimilarity: roundNum(r.rawSimilarity, 5),
        zAs: roundNum(r.zAs, 5),
        behaviorLlr: roundNum(r.behaviorLlr, 5),
        enrollEnv: this.getEnrollEnv(roomCode, r.id, domKey),
      })),
      deviceMotion: this.settings.rawConsent || motionSummary.available ? motionSummary : null,
      optInDiagnostics: this.settings.rawConsent
        ? {
            combinedEmbedding: combinedEmbedding.map((v) => roundNum(v, 5)),
            mergedBehavior,
            prompts: items.map((it) => it.prompt || null),
            rawCaptures: items.map((it) => this._sanitizeRawCapture(it.capture, { allowKeyIdentity: true })),
          }
        : null,
    };

    this.records.push(record);
    await this._putStore('records', record);
    await this._pushToHub([record]);
    this._emit();
    return record;
  }

  async recordGuardQuery({
    roomCode,
    targetAccount,
    passwordText,
    capture,
    bothEmbeddings,
    res,
    thresholdOffset,
    adapted,
    latencyMs,
    recorder,
    motionSnapshot = null,
    engine,
  }) {
    if (!this.settings.enabled) return null;
    const env = this.currentEnvSnapshot({ engine, lang: 'en', inputMode: capture.inputMode });
    if (this.hubActive && !this.getEnrollEnv(roomCode, targetAccount.id, 'password')) {
      await this._pullFromHubOnce();
    }
    const targetEnrollEnv = this.getEnrollEnv(roomCode, targetAccount.id, 'password');

    const gt = this.settings.groundTruthRole || 'unspecified';
    let actualRole = 'unspecified';
    let isGenuine = null;
    if (gt === 'outsider') {
      actualRole = 'impostor';
      isGenuine = false;
    } else if (gt.startsWith('enrolled:')) {
      const actualId = gt.slice('enrolled:'.length);
      const actualEnrollEnv =
        this.getEnrollEnv(roomCode, actualId, 'password') ||
        this.getEnrollEnv(roomCode, actualId, 'arena_ko') ||
        this.getEnrollEnv(roomCode, actualId, 'arena_en');
      const sameParticipant =
        Boolean(actualEnrollEnv?.participantId) &&
        Boolean(targetEnrollEnv?.participantId) &&
        actualEnrollEnv.participantId === targetEnrollEnv.participantId;
      isGenuine = actualId === targetAccount.id || sameParticipant;
      actualRole = isGenuine ? 'genuine' : 'impostor';
    }

    const crossDevice =
      targetEnrollEnv && targetEnrollEnv.keyboardType ? targetEnrollEnv.keyboardType !== env.keyboardType : null;

    const pubEntry = this.findPublicPasswordEntry(passwordText);
    const allowKeyIdentity = Boolean(pubEntry);
    const pwHash = passwordText ? (await sha256Hex(`kp-pw:${passwordText}`)).slice(0, 12) : null;
    const strFx =
      passwordText && engine?.config?.keyboard_layout
        ? extractStringCalibrationFeatures(
            passwordText,
            engine.config.keyboard_layout,
            engine.calibrationStudy?.digraph_table || {},
            engine.calibrationStudy?.support_kappa ?? 40.0
          )
        : null;

    const motionSummary = motionSnapshot || this._captureMotionWindow(recorder);
    const decisionCorrect = isGenuine === null ? null : Boolean(res.accepted) === isGenuine;

    const record = {
      schema: EVAL_SCHEMA_VERSION,
      recordId: `ev_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      eventType: 'query',
      domain: 'password',
      roomCode: roomCode || 'LOCAL',
      recordedAt: env.recordedAt,
      queryEnv: env,
      targetAccount: {
        id: targetAccount.id,
        name: targetAccount.name,
        templateInputMode: targetAccount.password?.inputMode || 'keyboard',
        templateShotCount: targetAccount.password?.count || 0,
        templateHoldMs: targetAccount.password?.holdMs || 0,
        enrollEnv: targetEnrollEnv,
      },
      groundTruth: {
        role: actualRole,
        isGenuine,
        selectedRoleRaw: gt,
      },
      crossDevice,
      passwordMeta: {
        length: [...String(passwordText || '')].length,
        publicExperimentId: pubEntry?.id || null,
        publicCategory: pubEntry?.category || null,
        publicTier: pubEntry?.tier || null,
        publicText: pubEntry ? pubEntry.text : null,
        passwordHash12: pwHash,
        isPublicAllowlist: Boolean(pubEntry),
        stringFeatures: strFx ? strFx.features.map((v) => roundNum(v, 5)) : null,
        transitionSupport: strFx ? roundNum(strFx.support, 5) : null,
      },
      verdict: {
        method: res.method,
        stringCalibrationActive: Boolean(res.stringCalibrationActive),
        similarity: roundNum(res.similarity, 5),
        z: roundNum(res.z, 5),
        raw9: roundNum(res.raw9, 5),
        raw17: roundNum(res.raw17, 5),
        capitalLlr: roundNum(res.capitalLlr, 5),
        threshold: roundNum(res.threshold, 5),
        thresholdOffset: roundNum(thresholdOffset, 4),
        accepted: Boolean(res.accepted),
        decisionCorrect,
        adaptedAfterVerdict: Boolean(adapted),
        lockAdaptationActive: this.shouldLockAdaptation(),
        latencyMs,
        lengthBaseline: res.lengthBaseline
          ? {
              z: roundNum(res.lengthBaseline.z, 5),
              similarity: roundNum(res.lengthBaseline.similarity, 5),
              accepted: Boolean(res.lengthBaseline.accepted),
              mu9: roundNum(res.lengthBaseline.mu9, 5),
              sd9: roundNum(res.lengthBaseline.sd9, 5),
              mu17: roundNum(res.lengthBaseline.mu17, 5),
              sd17: roundNum(res.lengthBaseline.sd17, 5),
              tau: roundNum(res.lengthBaseline.tau, 5),
            }
          : null,
        stringCalibrated: res.stringCalibrated
          ? {
              z: roundNum(res.stringCalibrated.z, 5),
              similarity: roundNum(res.stringCalibrated.similarity, 5),
              accepted: Boolean(res.stringCalibrated.accepted),
              mu9: roundNum(res.stringCalibrated.mu9, 5),
              sd9: roundNum(res.stringCalibrated.sd9, 5),
              mu17: roundNum(res.stringCalibrated.mu17, 5),
              sd17: roundNum(res.stringCalibrated.sd17, 5),
              tau: roundNum(res.stringCalibrated.tau, 5),
              shrinkWeight: roundNum(res.stringCalibrated.shrinkWeight, 5),
              support: roundNum(res.stringCalibrated.support, 5),
              latencyMs: res.stringCalibrated.latencyMs,
            }
          : null,
      },
      deviceMotion: this.settings.rawConsent || motionSummary.available ? motionSummary : null,
      optInDiagnostics: this.settings.rawConsent
        ? {
            embedding9: (bothEmbeddings?.password || []).map((v) => roundNum(v, 5)),
            embedding17: (bothEmbeddings?.passwordAux || []).map((v) => roundNum(v, 5)),
            rawCapture: this._sanitizeRawCapture(capture, { allowKeyIdentity }),
          }
        : null,
    };

    this.records.push(record);
    await this._putStore('records', record);
    await this._pushToHub([record]);
    this._emit();
    return record;
  }

  async connectLocalHub() {
    if (
      typeof window === 'undefined' ||
      !window.__KEYPRINT_FIELD_HUB?.active ||
      typeof fetch !== 'function'
    ) {
      return null;
    }
    try {
      const res = await fetch('./api/field-study/info', { cache: 'no-store' });
      if (!res.ok) return null;
      const ctype = res.headers.get('content-type') || '';
      if (!ctype.includes('application/json')) return null;
      const info = await res.json();
      if (!info?.ok || !info.hubVersion) return null;
      this.hubActive = true;
      this.hubInfo = info;
      this.hubSyncError = null;
      await this.syncWithHub();
      return this.hubInfo;
    } catch (err) {
      this.hubSyncError = err?.message || 'hub_unreachable';
      return null;
    }
  }

  scheduleHubPull(delayMs = 150) {
    if (!this.hubActive) return;
    if (this._pullTimer) clearTimeout(this._pullTimer);
    this._pullTimer = setTimeout(() => {
      this._pullTimer = null;
      this._pullFromHubOnce().then((changed) => {
        if (changed) this._emit();
      });
    }, delayMs);
  }

  async _pullFromHubOnce() {
    if (!this.hubActive || typeof fetch !== 'function') return false;
    try {
      const res = await fetch('./api/field-study/records', { cache: 'no-store' });
      if (!res.ok) return false;
      const data = await res.json();
      if (!data?.ok) return false;
      let changed = false;
      if (Array.isArray(data.enrollEnvs)) {
        for (const item of data.enrollEnvs) {
          if (item?.key && item?.env && !this.enrollEnvs.has(item.key)) {
            this.enrollEnvs.set(item.key, item.env);
            await this._putStore('enroll_envs', { key: item.key, env: item.env });
            changed = true;
          }
        }
      }
      if (data.jsonl) {
        const imported = await this._importJsonlInternal(data.jsonl);
        if (imported.added > 0) changed = true;
      }
      if (this.hubInfo && Number.isFinite(data.total)) {
        this.hubInfo.savedRecordCount = data.total;
      }
      this.hubSyncError = null;
      return changed;
    } catch (err) {
      this.hubSyncError = err?.message || 'pull_failed';
      return false;
    }
  }

  async _pushToHub(recordsToSend = []) {
    if (!this.hubActive || typeof fetch !== 'function') return null;
    try {
      const enrollEnvs = Array.from(this.enrollEnvs.entries()).map(([key, env]) => ({ key, env }));
      const res = await fetch('./api/field-study/records', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records: recordsToSend, enrollEnvs }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      if (data?.ok && this.hubInfo && Number.isFinite(data.total)) {
        this.hubInfo.savedRecordCount = data.total;
      }
      this.hubSyncError = null;
      return data;
    } catch (err) {
      this.hubSyncError = err?.message || 'push_failed';
      return null;
    }
  }

  async syncWithHub() {
    if (!this.hubActive) return { ok: false, total: this.records.length, hubTotal: 0 };
    await this._pullFromHubOnce();
    if (this.records.length > 0 || this.enrollEnvs.size > 0) {
      await this._pushToHub(this.records);
    }
    this._emit();
    return {
      ok: !this.hubSyncError,
      total: this.records.length,
      hubTotal: this.hubInfo?.savedRecordCount ?? this.records.length,
    };
  }

  toJsonl() {
    return this.records.map((r) => JSON.stringify(r)).join('\n') + (this.records.length ? '\n' : '');
  }

  async _importJsonlInternal(text) {
    const lines = String(text || '')
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    const existingIds = new Set(this.records.map((r) => r.recordId));
    let added = 0;
    let skipped = 0;
    for (const line of lines) {
      const obj = JSON.parse(line);
      if (!obj || typeof obj !== 'object' || !obj.recordId || !obj.eventType) {
        skipped += 1;
        continue;
      }
      if (existingIds.has(obj.recordId)) {
        skipped += 1;
        continue;
      }
      existingIds.add(obj.recordId);
      this.records.push(obj);
      await this._putStore('records', obj);
      if (obj.eventType === 'enroll' && obj.profileId && obj.env) {
        const domKey = obj.domain === 'password' ? 'password' : `arena_${obj.lang || 'en'}`;
        const key = this._envKey(obj.roomCode, obj.profileId, domKey);
        this.enrollEnvs.set(key, obj.env);
        await this._putStore('enroll_envs', { key, env: obj.env });
      }
      added += 1;
    }
    if (added > 0) {
      this.records.sort((a, b) => String(a.recordedAt).localeCompare(String(b.recordedAt)));
    }
    return { added, skipped, total: this.records.length };
  }

  async importJsonlText(text) {
    const res = await this._importJsonlInternal(text);
    if (res.added > 0) {
      await this._pushToHub(this.records);
    }
    this._emit();
    return res;
  }

  computeLiveSummary() {
    const enrolls = this.records.filter((r) => r.eventType === 'enroll');
    const arenaQueries = this.records.filter((r) => r.eventType === 'query' && r.domain === 'arena');
    const pwQueries = this.records.filter((r) => r.eventType === 'query' && r.domain === 'password');

    const arenaEnrolled = arenaQueries.filter((r) => r.groundTruth?.role === 'enrolled');
    const arenaOutsider = arenaQueries.filter((r) => r.groundTruth?.role === 'outsider');
    const pwGenuine = pwQueries.filter((r) => r.groundTruth?.isGenuine === true);
    const pwImpostor = pwQueries.filter((r) => r.groundTruth?.isGenuine === false);

    const rate = (arr, pred) => (arr.length ? Math.round((arr.filter(pred).length / arr.length) * 1000) / 10 : null);

    return {
      totalRecords: this.records.length,
      enrollCount: enrolls.length,
      arenaQueryCount: arenaQueries.length,
      pwQueryCount: pwQueries.length,
      devicesSeen: [...new Set(this.records.map((r) => (r.env || r.queryEnv)?.keyboardType).filter(Boolean))],
      arenaTop1Pct: rate(arenaEnrolled, (r) => r.verdict?.top1Correct === true),
      arenaAcceptPct: rate(arenaEnrolled, (r) => r.verdict?.decisionCorrect === true),
      arenaOutsiderFarPct: rate(arenaOutsider, (r) => r.verdict?.accepted === true),
      pwGenuineAcceptPct: rate(pwGenuine, (r) => r.verdict?.accepted === true),
      pwImpostorFarPct: rate(pwImpostor, (r) => r.verdict?.accepted === true),
      pwCrossDeviceCount: pwQueries.filter((r) => r.crossDevice === true).length,
      arenaCrossDeviceCount: arenaQueries.filter((r) => r.crossDevice === true).length,
    };
  }

  _emit() {
    if (this.onChange) this.onChange(this.computeLiveSummary(), this.motion.getLiveStatus());
  }
}
