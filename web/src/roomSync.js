import mqtt from 'mqtt';
import { BEHAVIOR_FEATURES } from './features.js';

const DEFAULT_BROKERS = [
  'wss://broker.emqx.io:8084/mqtt',
  'wss://broker.hivemq.com:8884/mqtt',
];

// v4: Frozen stitch/tensor preprocessing + dual-encoder hash & concrete profile schema isolation.
// Profiles and room events carry { schemaVersion, preprocessVersion, modelHash } across MQTT,
// BroadcastChannel, and localStorage so incompatible embeddings can never mix or wipe valid state.
// NOTE: `modelHash` & `preprocessVersion` are embedding-space compatibility guards, NOT cryptographic
// user ownership authentication.
export const TOPIC_NS = 'keyprint-agy-mvp-v4';
export const SCHEMA_VERSION = 4;
export const DEFAULT_PREPROCESS_VERSION = 'kp-preproc-v4';

const ID_RE = /^[a-zA-Z0-9_-]{1,64}$/;
const HEX_COLOR_RE = /^#[0-9a-fA-F]{6}$/;
const VALID_INPUT_MODES = new Set(['keyboard', 'touch', 'mixed']);
const BEHAVIOR_FEATURE_SET = new Set(BEHAVIOR_FEATURES);
const FORBIDDEN_PROTO_KEYS = ['__proto__', 'constructor', 'prototype'];

function isSafePlainObject(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  for (const k of FORBIDDEN_PROTO_KEYS) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) return false;
  }
  const proto = Object.getPrototypeOf(obj);
  return proto === Object.prototype || proto === null;
}

export function normalizeProfileName(raw) {
  if (typeof raw !== 'string') return '';
  return raw.replace(/[\x00-\x1F\x7F]/g, '').trim();
}

function sanitizeVec64(v) {
  if (!Array.isArray(v) || v.length !== 64) return null;
  const copy = new Array(64);
  let sumSq = 0;
  for (let i = 0; i < 64; i++) {
    const x = v[i];
    if (typeof x !== 'number' || !Number.isFinite(x)) return null;
    copy[i] = x;
    sumSq += x * x;
  }
  if (Math.sqrt(sumSq) <= 1e-6) return null;
  return copy;
}

function sanitizeExemplars(list) {
  if (!Array.isArray(list) || list.length < 1 || list.length > 12) return null;
  const out = [];
  for (let i = 0; i < list.length; i++) {
    const vec = sanitizeVec64(list[i]);
    if (!vec) return null;
    out.push(vec);
  }
  return out;
}

function sanitizeBehavior(beh) {
  if (beh === null || beh === undefined) return { ok: true, value: null };
  if (!isSafePlainObject(beh)) return { ok: false, reason: 'INVALID_BEHAVIOR' };
  const keys = Object.keys(beh);
  if (keys.length > BEHAVIOR_FEATURE_SET.size) return { ok: false, reason: 'INVALID_BEHAVIOR_KEYS' };
  const out = {};
  for (const k of keys) {
    if (!BEHAVIOR_FEATURE_SET.has(k)) return { ok: false, reason: 'UNKNOWN_BEHAVIOR_KEY' };
    const pair = beh[k];
    if (!Array.isArray(pair) || pair.length !== 2) return { ok: false, reason: 'INVALID_BEHAVIOR_PAIR' };
    const [sum, count] = pair;
    if (typeof sum !== 'number' || !Number.isFinite(sum)) return { ok: false, reason: 'INVALID_BEHAVIOR_SUM' };
    if (!Number.isInteger(count) || count < 0 || count > 100000) {
      return { ok: false, reason: 'INVALID_BEHAVIOR_COUNT' };
    }
    out[k] = [sum, count];
  }
  return { ok: true, value: out };
}

function sanitizeInputModeCounts(mc) {
  if (mc === null || mc === undefined) return { ok: true, value: undefined };
  if (!isSafePlainObject(mc)) return { ok: false, reason: 'INVALID_INPUT_MODE_COUNTS' };
  const keys = Object.keys(mc);
  if (keys.length > 3) return { ok: false, reason: 'INVALID_INPUT_MODE_COUNTS' };
  const out = {};
  for (const k of keys) {
    if (!VALID_INPUT_MODES.has(k)) return { ok: false, reason: 'INVALID_INPUT_MODE_KEY' };
    const v = mc[k];
    if (!Number.isInteger(v) || v < 0 || v > 100000) return { ok: false, reason: 'INVALID_INPUT_MODE_VAL' };
    out[k] = v;
  }
  return { ok: true, value: out };
}

function sanitizeNonNegNumber(val, defaultVal = 0) {
  if (val === undefined) return { ok: true, value: defaultVal };
  if (typeof val !== 'number' || !Number.isFinite(val) || val < 0) {
    return { ok: false, reason: 'INVALID_METRIC' };
  }
  return { ok: true, value: val };
}

function cloneSafeSampleStroke(strokeObj) {
  if (!isSafePlainObject(strokeObj)) return null;
  try {
    return JSON.parse(JSON.stringify(strokeObj));
  } catch {
    return null;
  }
}

function validateSentenceTemplate(tmpl, { isSample, allowUnstampedSample }) {
  if (!isSafePlainObject(tmpl)) return { ok: false, reason: 'INVALID_ARENA_TEMPLATE' };
  if ('enc17' in tmpl && tmpl.enc17 !== undefined) return { ok: false, reason: 'ARENA_FORBIDS_ENC17' };
  if ('target' in tmpl && tmpl.target !== undefined) return { ok: false, reason: 'ARENA_FORBIDS_TARGET' };
  if ('testGenuineStroke' in tmpl && tmpl.testGenuineStroke !== undefined) {
    return { ok: false, reason: 'ARENA_FORBIDS_PW_SAMPLE_STROKE' };
  }

  const centroid = sanitizeVec64(tmpl.centroid);
  if (!centroid) return { ok: false, reason: 'INVALID_ARENA_CENTROID' };
  const exemplars = sanitizeExemplars(tmpl.exemplars);
  if (!exemplars) return { ok: false, reason: 'INVALID_ARENA_EXEMPLARS' };
  if (!Number.isInteger(tmpl.count) || tmpl.count < 1 || tmpl.count > 10000) {
    return { ok: false, reason: 'INVALID_ARENA_COUNT' };
  }

  const rawMode =
    tmpl.inputMode === undefined && allowUnstampedSample && isSample ? 'keyboard' : tmpl.inputMode;
  if (typeof rawMode !== 'string' || !VALID_INPUT_MODES.has(rawMode)) {
    return { ok: false, reason: 'INVALID_ARENA_INPUT_MODE' };
  }

  const cpm = sanitizeNonNegNumber(tmpl.cpm, 0);
  if (!cpm.ok) return cpm;
  const holdMs = sanitizeNonNegNumber(tmpl.holdMs, 0);
  if (!holdMs.ok) return holdMs;
  const rolloverPct = sanitizeNonNegNumber(tmpl.rolloverPct, 0);
  if (!rolloverPct.ok) return rolloverPct;
  const beh = sanitizeBehavior(tmpl.behavior);
  if (!beh.ok) return beh;
  const mc = sanitizeInputModeCounts(tmpl.inputModeCounts);
  if (!mc.ok) return mc;

  let testQueryStroke;
  if (tmpl.testQueryStroke !== undefined) {
    if (!isSample) return { ok: false, reason: 'FORBIDDEN_SAMPLE_FIELD' };
    testQueryStroke = cloneSafeSampleStroke(tmpl.testQueryStroke);
    if (!testQueryStroke) return { ok: false, reason: 'INVALID_SAMPLE_STROKE' };
  }

  const clean = {
    centroid,
    exemplars,
    count: tmpl.count,
    inputMode: rawMode,
    cpm: cpm.value,
    holdMs: holdMs.value,
    rolloverPct: rolloverPct.value,
    behavior: beh.value,
  };
  if (mc.value !== undefined) clean.inputModeCounts = mc.value;
  if (testQueryStroke !== undefined) clean.testQueryStroke = testQueryStroke;
  return { ok: true, value: clean };
}

function validatePasswordTemplate(tmpl, { isSample, allowUnstampedSample }) {
  if (!isSafePlainObject(tmpl)) return { ok: false, reason: 'INVALID_PW_TEMPLATE' };
  if ('testQueryStroke' in tmpl && tmpl.testQueryStroke !== undefined) {
    return { ok: false, reason: 'PW_FORBIDS_ARENA_SAMPLE_STROKE' };
  }

  const centroid = sanitizeVec64(tmpl.centroid);
  if (!centroid) return { ok: false, reason: 'INVALID_PW_CENTROID' };
  const exemplars = sanitizeExemplars(tmpl.exemplars);
  if (!exemplars) return { ok: false, reason: 'INVALID_PW_EXEMPLARS' };
  if (!Number.isInteger(tmpl.count) || tmpl.count < 1 || tmpl.count > 10000) {
    return { ok: false, reason: 'INVALID_PW_COUNT' };
  }

  const rawMode =
    tmpl.inputMode === undefined && allowUnstampedSample && isSample ? 'keyboard' : tmpl.inputMode;
  if (typeof rawMode !== 'string' || !VALID_INPUT_MODES.has(rawMode)) {
    return { ok: false, reason: 'INVALID_PW_INPUT_MODE' };
  }

  if (typeof tmpl.target !== 'string') return { ok: false, reason: 'INVALID_PW_TARGET' };
  if (/[\x00-\x1F\x7F]/.test(tmpl.target)) return { ok: false, reason: 'INVALID_PW_CONTROL_CHARS' };
  const minLen = isSample ? 6 : 12;
  if (tmpl.target.length < minLen || tmpl.target.length > 128) {
    return { ok: false, reason: 'INVALID_PW_LENGTH' };
  }

  if (!isSafePlainObject(tmpl.enc17)) return { ok: false, reason: 'INVALID_PW_ENC17' };
  const enc17Centroid = sanitizeVec64(tmpl.enc17.centroid);
  if (!enc17Centroid) return { ok: false, reason: 'INVALID_PW_ENC17_CENTROID' };
  const enc17Exemplars = sanitizeExemplars(tmpl.enc17.exemplars);
  if (!enc17Exemplars) return { ok: false, reason: 'INVALID_PW_ENC17_EXEMPLARS' };

  let capsUse;
  if (tmpl.capsUse !== undefined && tmpl.capsUse !== null) {
    if (
      !Array.isArray(tmpl.capsUse) ||
      tmpl.capsUse.length !== 2 ||
      !tmpl.capsUse.every((n) => Number.isInteger(n) && n >= 0 && n <= 100000)
    ) {
      return { ok: false, reason: 'INVALID_CAPS_USE' };
    }
    capsUse = [tmpl.capsUse[0], tmpl.capsUse[1]];
  }

  const cpm = sanitizeNonNegNumber(tmpl.cpm, 0);
  if (!cpm.ok) return cpm;
  const holdMs = sanitizeNonNegNumber(tmpl.holdMs, 0);
  if (!holdMs.ok) return holdMs;
  const rolloverPct = sanitizeNonNegNumber(tmpl.rolloverPct, 0);
  if (!rolloverPct.ok) return rolloverPct;
  const beh = sanitizeBehavior(tmpl.behavior);
  if (!beh.ok) return beh;
  const mc = sanitizeInputModeCounts(tmpl.inputModeCounts);
  if (!mc.ok) return mc;

  let testGenuineStroke;
  if (tmpl.testGenuineStroke !== undefined) {
    if (!isSample) return { ok: false, reason: 'FORBIDDEN_SAMPLE_FIELD' };
    testGenuineStroke = cloneSafeSampleStroke(tmpl.testGenuineStroke);
    if (!testGenuineStroke) return { ok: false, reason: 'INVALID_SAMPLE_STROKE' };
  }

  const clean = {
    target: tmpl.target,
    enc17: { centroid: enc17Centroid, exemplars: enc17Exemplars },
    centroid,
    exemplars,
    count: tmpl.count,
    inputMode: rawMode,
    cpm: cpm.value,
    holdMs: holdMs.value,
    rolloverPct: rolloverPct.value,
    behavior: beh.value,
  };
  if (capsUse !== undefined) clean.capsUse = capsUse;
  if (mc.value !== undefined) clean.inputModeCounts = mc.value;
  if (testGenuineStroke !== undefined) clean.testGenuineStroke = testGenuineStroke;
  return { ok: true, value: clean };
}

/**
 * Validate and deep-sanitize a profile object against the concrete v4 schema and active
 * `{ schemaVersion, preprocessVersion, modelHash }` compatibility envelope.
 * Never throws on malformed/adversarial inputs.
 */
export function validateProfileSchema(
  profile,
  {
    schemaVersion = SCHEMA_VERSION,
    preprocessVersion = DEFAULT_PREPROCESS_VERSION,
    modelHash = null,
    allowUnstampedSample = false,
  } = {}
) {
  try {
    if (!isSafePlainObject(profile)) return { ok: false, reason: 'NOT_OBJECT' };

    if (typeof profile.id !== 'string' || !ID_RE.test(profile.id)) {
      return { ok: false, reason: 'INVALID_ID' };
    }
    if (typeof profile.name !== 'string') {
      return { ok: false, reason: 'INVALID_NAME_TYPE' };
    }
    const cleanName = normalizeProfileName(profile.name);
    if (cleanName.length < 1 || cleanName.length > 24) {
      return { ok: false, reason: 'INVALID_NAME_LENGTH' };
    }
    if (typeof profile.isSample !== 'boolean') {
      return { ok: false, reason: 'INVALID_IS_SAMPLE' };
    }
    if (typeof profile.color !== 'string' || !HEX_COLOR_RE.test(profile.color)) {
      return { ok: false, reason: 'INVALID_COLOR' };
    }

    const canStampSample = Boolean(allowUnstampedSample && profile.isSample === true);
    const now = Date.now();
    const createdAt = profile.createdAt === undefined && canStampSample ? now : profile.createdAt;
    const updatedAt = profile.updatedAt === undefined && canStampSample ? now : profile.updatedAt;
    if (typeof createdAt !== 'number' || !Number.isFinite(createdAt) || createdAt < 0) {
      return { ok: false, reason: 'INVALID_CREATED_AT' };
    }
    if (typeof updatedAt !== 'number' || !Number.isFinite(updatedAt) || updatedAt < 0) {
      return { ok: false, reason: 'INVALID_UPDATED_AT' };
    }

    const pSchema = profile.schemaVersion === undefined && canStampSample ? schemaVersion : profile.schemaVersion;
    const pPreproc =
      profile.preprocessVersion === undefined && canStampSample ? preprocessVersion : profile.preprocessVersion;
    const pHash = profile.modelHash === undefined && canStampSample ? modelHash : profile.modelHash;

    if (
      typeof schemaVersion !== 'number' ||
      typeof preprocessVersion !== 'string' ||
      !preprocessVersion ||
      typeof modelHash !== 'string' ||
      !modelHash ||
      pSchema !== schemaVersion ||
      pPreproc !== preprocessVersion ||
      pHash !== modelHash
    ) {
      return { ok: false, reason: 'VERSION_MISMATCH' };
    }

    if (profile.arena !== undefined && profile.arena !== null && !isSafePlainObject(profile.arena)) {
      return { ok: false, reason: 'INVALID_ARENA_CONTAINER' };
    }
    const cleanArena = {};
    if (profile.arena) {
      for (const k of Object.keys(profile.arena)) {
        if (k !== 'ko' && k !== 'en') return { ok: false, reason: 'INVALID_ARENA_LANG' };
        const slot = profile.arena[k];
        if (slot !== undefined && slot !== null) {
          const res = validateSentenceTemplate(slot, {
            isSample: profile.isSample,
            allowUnstampedSample: canStampSample,
          });
          if (!res.ok) return res;
          cleanArena[k] = res.value;
        }
      }
    }

    let cleanPassword = null;
    if (profile.password !== undefined && profile.password !== null) {
      const res = validatePasswordTemplate(profile.password, {
        isSample: profile.isSample,
        allowUnstampedSample: canStampSample,
      });
      if (!res.ok) return res;
      cleanPassword = res.value;
    }

    if (!cleanArena.ko && !cleanArena.en && !cleanPassword) {
      return { ok: false, reason: 'NO_VALID_TEMPLATE' };
    }

    return {
      ok: true,
      profile: {
        id: profile.id,
        name: cleanName,
        color: profile.color,
        isSample: profile.isSample,
        createdAt,
        updatedAt,
        schemaVersion: pSchema,
        preprocessVersion: pPreproc,
        modelHash: pHash,
        password: cleanPassword,
        arena: cleanArena,
      },
    };
  } catch {
    return { ok: false, reason: 'INTERNAL_VALIDATION_ERROR' };
  }
}

export function sanitizeRoomCode(raw) {
  const cleaned = String(raw || '')
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9가-힣_-]/g, '')
    .slice(0, 16);
  return cleaned || 'GACHON';
}

export function getInitialRoomCode() {
  if (typeof window !== 'undefined' && window.location) {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get('room');
    if (fromUrl) return sanitizeRoomCode(fromUrl);
  }
  try {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('keyprint_last_room');
      if (saved) return sanitizeRoomCode(saved);
    }
  } catch {
    // Ignore storage access errors
  }
  return 'GACHON';
}

const AVATAR_COLORS = [
  '#2563eb',
  '#059669',
  '#d97706',
  '#7c3aed',
  '#e11d48',
  '#0891b2',
  '#4f46e5',
  '#ea580c',
];

export function pickProfileColor(name) {
  let hash = 0;
  const s = String(name || '');
  for (let i = 0; i < s.length; i++) {
    hash = (hash * 31 + s.charCodeAt(i)) >>> 0;
  }
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}

function finiteTs(v) {
  const n = Number(v);
  return Number.isFinite(n) && n >= 0 ? n : Date.now();
}

/**
 * Multi-user real-time profile store synchronized across:
 * 1) Multiple devices / laptops simultaneously via MQTT-over-WSS retained topics
 * 2) Same-browser tabs via BroadcastChannel
 * 3) Offline / page reloads via localStorage
 *
 * Enforces v4 `{ schemaVersion, preprocessVersion, modelHash }` isolation and `validateProfileSchema`
 * across every incoming & persisted channel.
 */
export class RoomProfileSync {
  constructor({
    onChange,
    onStatusChange,
    onCompatibilitySkip,
    schemaVersion = SCHEMA_VERSION,
    preprocessVersion = null,
    modelHash = null,
    brokers = null,
  } = {}) {
    this.onChange = onChange;
    this.onStatusChange = onStatusChange;
    this.onCompatibilitySkip = onCompatibilitySkip;
    this.schemaVersion = schemaVersion;
    this.preprocessVersion = preprocessVersion;
    this.modelHash = modelHash;
    this.customBrokers = brokers;

    this.clientId = `kp_${Math.random().toString(36).slice(2, 10)}`;
    this.roomCode = 'GACHON';
    this.profiles = new Map(); // id -> validated profile object
    this.tombstones = new Map(); // id -> deleted timestamp
    this.peers = new Map(); // clientId -> timestamp
    this.client = null;
    this.bc = null;
    this.heartbeatTimer = null;
    this.failoverTimer = null;
    this.brokerIdx = 0;
    this.status = 'connecting'; // 'connecting' | 'online' | 'local'
    this.compatSkipCount = 0;
  }

  _brokers() {
    if (Array.isArray(this.customBrokers) && this.customBrokers.length > 0) {
      return this.customBrokers;
    }
    if (typeof window !== 'undefined' && Array.isArray(window.__KEYPRINT_MQTT_BROKERS) && window.__KEYPRINT_MQTT_BROKERS.length > 0) {
      return window.__KEYPRINT_MQTT_BROKERS;
    }
    return DEFAULT_BROKERS;
  }

  _compatEnvelope() {
    return {
      schemaVersion: this.schemaVersion,
      preprocessVersion: this.preprocessVersion,
      modelHash: this.modelHash,
    };
  }

  _isCompatibleEnvelope(obj) {
    if (!obj || typeof obj !== 'object') return false;
    if (!this.preprocessVersion || !this.modelHash) return false;
    return (
      obj.schemaVersion === this.schemaVersion &&
      obj.preprocessVersion === this.preprocessVersion &&
      obj.modelHash === this.modelHash
    );
  }

  _notifyCompatibilitySkip(source, payload) {
    this.compatSkipCount += 1;
    if (this.onCompatibilitySkip) {
      this.onCompatibilitySkip({
        source,
        count: this.compatSkipCount,
        roomCode: this.roomCode,
        incomingSchemaVersion: payload?.schemaVersion ?? null,
        incomingPreprocessVersion: payload?.preprocessVersion ?? null,
        incomingModelHash: payload?.modelHash ?? null,
      });
    }
  }

  setCompatibility({ schemaVersion = SCHEMA_VERSION, preprocessVersion, modelHash } = {}) {
    this.schemaVersion = schemaVersion;
    if (preprocessVersion) this.preprocessVersion = preprocessVersion;
    if (modelHash) this.modelHash = modelHash;

    // Re-validate any already-loaded profiles against the updated compatibility tuple
    let pruned = false;
    for (const [id, p] of this.profiles.entries()) {
      const check = validateProfileSchema(p, this._compatEnvelope());
      if (!check.ok) {
        this.profiles.delete(id);
        pruned = true;
        if (check.reason === 'VERSION_MISMATCH') {
          this._notifyCompatibilitySkip('revalidation', p);
        }
      } else {
        this.profiles.set(id, check.profile);
      }
    }
    if (pruned) {
      this._emitChange();
    }
  }

  connect(roomCode) {
    this.disconnect();
    this.roomCode = sanitizeRoomCode(roomCode);
    this.compatSkipCount = 0;

    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem('keyprint_last_room', this.roomCode);
      }
    } catch {
      // Ignore storage errors
    }

    if (typeof window !== 'undefined' && window.location && window.history?.replaceState) {
      try {
        const url = new URL(window.location.href);
        url.searchParams.set('room', this.roomCode);
        window.history.replaceState({}, '', url.toString());
      } catch {
        // Ignore URL update errors
      }
    }

    this.profiles.clear();
    this.tombstones.clear();
    this.peers.clear();
    this.peers.set(this.clientId, Date.now());

    this._loadFromLocalStorage();
    this._setupBroadcastChannel();
    this._connectMqtt(0);
    this._emitChange();
  }

  disconnect() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.failoverTimer) {
      clearTimeout(this.failoverTimer);
      this.failoverTimer = null;
    }
    if (this.bc) {
      try {
        this.bc.close();
      } catch {
        // Ignore
      }
      this.bc = null;
    }
    if (this.client) {
      try {
        const peerTopic = this._peerTopic(this.clientId);
        this.client.publish(peerTopic, '', { qos: 0, retain: true });
        this.client.end(true);
      } catch {
        // Ignore cleanup errors
      }
      this.client = null;
    }
  }

  _storageKey() {
    return `keyprint_mvp_v4_room_${this.roomCode}`;
  }

  _profileTopic(id = '+') {
    return `${TOPIC_NS}/room/${this.roomCode}/profiles/${id}`;
  }

  _eventTopic() {
    return `${TOPIC_NS}/room/${this.roomCode}/events`;
  }

  _peerTopic(cid = '+') {
    return `${TOPIC_NS}/room/${this.roomCode}/peers/${cid}`;
  }

  _loadFromLocalStorage() {
    if (!this.preprocessVersion || !this.modelHash) return;
    let raw = null;
    try {
      if (typeof localStorage !== 'undefined') {
        raw = localStorage.getItem(this._storageKey());
      }
    } catch {
      return;
    }
    if (!raw) return;

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      this._saveToLocalStorage();
      return;
    }

    if (!isSafePlainObject(parsed)) {
      this._saveToLocalStorage();
      return;
    }

    const hasTopEnvelope =
      parsed.schemaVersion !== undefined ||
      parsed.preprocessVersion !== undefined ||
      parsed.modelHash !== undefined;
    if (hasTopEnvelope && !this._isCompatibleEnvelope(parsed)) {
      this._notifyCompatibilitySkip('storage_envelope', parsed);
      this._saveToLocalStorage();
      return;
    }

    let hadInvalid = false;
    if (Array.isArray(parsed.profiles)) {
      for (const p of parsed.profiles) {
        const check = validateProfileSchema(p, this._compatEnvelope());
        if (check.ok) {
          this.profiles.set(check.profile.id, check.profile);
        } else {
          hadInvalid = true;
          if (check.reason === 'VERSION_MISMATCH') {
            this._notifyCompatibilitySkip('storage_profile', p);
          }
        }
      }
    } else if (parsed.profiles !== undefined) {
      hadInvalid = true;
    }

    if (isSafePlainObject(parsed.tombstones)) {
      for (const [k, ts] of Object.entries(parsed.tombstones)) {
        const numTs = Number(ts);
        if (ID_RE.test(k) && Number.isFinite(numTs) && numTs >= 0) {
          this.tombstones.set(k, numTs);
        } else {
          hadInvalid = true;
        }
      }
    } else if (parsed.tombstones !== undefined) {
      hadInvalid = true;
    }

    if (hadInvalid) {
      this._saveToLocalStorage();
    }
  }

  _saveToLocalStorage() {
    if (!this.preprocessVersion || !this.modelHash) return;
    try {
      if (typeof localStorage === 'undefined') return;
      const payload = {
        ...this._compatEnvelope(),
        profiles: Array.from(this.profiles.values()),
        tombstones: Object.fromEntries(this.tombstones.entries()),
      };
      localStorage.setItem(this._storageKey(), JSON.stringify(payload));
    } catch {
      // Ignore storage quota / sandbox errors
    }
  }

  _setupBroadcastChannel() {
    if (typeof BroadcastChannel === 'undefined') return;
    try {
      this.bc = new BroadcastChannel(`keyprint_bc_v4_${this.roomCode}`);
      this.bc.onmessage = (ev) => {
        const msg = ev?.data;
        if (!isSafePlainObject(msg)) return;
        if (!this._isCompatibleEnvelope(msg)) {
          this._notifyCompatibilitySkip('broadcast', msg);
          return;
        }
        if (msg.type === 'upsert' && msg.profile) {
          this._mergeIncomingProfile(msg.profile, 'broadcast');
        } else if (msg.type === 'delete' && typeof msg.id === 'string' && ID_RE.test(msg.id)) {
          this._applyDelete(msg.id, finiteTs(msg.ts));
        } else if (msg.type === 'clear_all') {
          this._applyClearAll(finiteTs(msg.ts));
        }
      };
    } catch {
      this.bc = null;
    }
  }

  _connectMqtt(brokerIndex) {
    const brokers = this._brokers();
    if (brokers.length === 0) {
      this._setStatus('local');
      return;
    }
    this.brokerIdx = brokerIndex % brokers.length;
    const brokerUrl = brokers[this.brokerIdx];
    this._setStatus('connecting');

    const myPeerTopic = this._peerTopic(this.clientId);
    let client;
    try {
      client = mqtt.connect(brokerUrl, {
        clientId: `${this.clientId}_${Math.random().toString(36).slice(2, 6)}`,
        clean: true,
        connectTimeout: 4500,
        reconnectPeriod: 4000,
        will: {
          topic: myPeerTopic,
          payload: '',
          qos: 0,
          retain: true,
        },
      });
    } catch {
      // Synchronous WebSocket throw (e.g. CSP / sandbox / blocked WS): fall back cleanly without crashing!
      if (brokerIndex + 1 < brokers.length) {
        this._connectMqtt(brokerIndex + 1);
      } else {
        this.client = null;
        this._setStatus('local');
      }
      return;
    }
    this.client = client;

    let connectedOnce = false;
    if (this.failoverTimer) clearTimeout(this.failoverTimer);
    this.failoverTimer = setTimeout(() => {
      if (!connectedOnce && this.client === client) {
        try {
          client.end(true);
        } catch {
          // ignore
        }
        if (brokerIndex + 1 < brokers.length) {
          this._connectMqtt(brokerIndex + 1);
        } else {
          this._setStatus('local');
        }
      }
    }, 4800);

    client.on('connect', () => {
      if (this.client !== client) return;
      connectedOnce = true;
      if (this.failoverTimer) {
        clearTimeout(this.failoverTimer);
        this.failoverTimer = null;
      }
      this._setStatus('online');

      client.subscribe(
        [this._profileTopic('+'), this._eventTopic(), this._peerTopic('+')],
        { qos: 1 },
        () => {
          // Re-broadcast any locally created valid non-sample profiles so room stays converged
          for (const p of this.profiles.values()) {
            if (!p.isSample && this._isCompatibleEnvelope(p)) {
              client.publish(this._profileTopic(p.id), JSON.stringify(p), { qos: 1, retain: true });
            }
          }
          this._publishPresence();
        }
      );

      if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = setInterval(() => this._publishPresence(), 18000);
    });

    client.on('message', (topic, buf) => {
      if (this.client !== client) return;
      const text = buf ? buf.toString() : '';

      if (topic.startsWith(`${TOPIC_NS}/room/${this.roomCode}/peers/`)) {
        const cid = topic.split('/').pop();
        if (!cid || !ID_RE.test(cid)) return;
        if (!text) {
          if (cid !== this.clientId) this.peers.delete(cid);
        } else {
          try {
            const data = JSON.parse(text);
            if (isSafePlainObject(data)) {
              this.peers.set(cid, finiteTs(data.ts));
            }
          } catch {
            // ignore
          }
        }
        this._emitStatus();
        return;
      }

      if (topic === this._eventTopic()) {
        if (!text) return;
        try {
          const ev = JSON.parse(text);
          if (!isSafePlainObject(ev)) return;
          if (!this._isCompatibleEnvelope(ev)) {
            this._notifyCompatibilitySkip('mqtt_event', ev);
            return;
          }
          if (ev.type === 'delete' && typeof ev.id === 'string' && ID_RE.test(ev.id)) {
            this._applyDelete(ev.id, finiteTs(ev.ts));
          } else if (ev.type === 'clear_all') {
            this._applyClearAll(finiteTs(ev.ts));
          }
        } catch {
          // ignore malformed event JSON
        }
        return;
      }

      if (topic.startsWith(`${TOPIC_NS}/room/${this.roomCode}/profiles/`)) {
        const pid = topic.split('/').pop();
        if (!pid || !ID_RE.test(pid)) return;
        // Empty payload is only broker retained-state cleanup; live deletes require a versioned event on _eventTopic().
        if (!text) return;
        try {
          const profile = JSON.parse(text);
          if (!isSafePlainObject(profile) || profile.id !== pid) return;
          this._mergeIncomingProfile(profile, 'mqtt');
        } catch {
          // ignore malformed profile JSON
        }
      }
    });

    client.on('offline', () => {
      if (this.client === client) this._setStatus('local');
    });

    client.on('error', () => {
      // Handled by failoverTimer / offline event
    });
  }

  _publishPresence() {
    this.peers.set(this.clientId, Date.now());
    const cutoff = Date.now() - 55000;
    for (const [cid, ts] of this.peers.entries()) {
      if (cid !== this.clientId && ts < cutoff) {
        this.peers.delete(cid);
      }
    }
    if (this.client && this.client.connected) {
      try {
        this.client.publish(
          this._peerTopic(this.clientId),
          JSON.stringify({ ...this._compatEnvelope(), cid: this.clientId, ts: Date.now() }),
          { qos: 0, retain: true }
        );
      } catch {
        // ignore
      }
    }
    this._emitStatus();
  }

  _setStatus(st) {
    this.status = st;
    this._emitStatus();
  }

  _emitStatus() {
    if (this.onStatusChange) {
      this.onStatusChange({
        status: this.status,
        roomCode: this.roomCode,
        peerCount: Math.max(1, this.peers.size),
        profileCount: this.profiles.size,
        compatSkipCount: this.compatSkipCount,
      });
    }
  }

  _emitChange() {
    this._saveToLocalStorage();
    this._emitStatus();
    if (this.onChange) {
      this.onChange(this.getProfiles());
    }
  }

  _mergeIncomingProfile(incoming, source = 'remote') {
    if (!isSafePlainObject(incoming)) return false;
    if (!this._isCompatibleEnvelope(incoming)) {
      this._notifyCompatibilitySkip(source, incoming);
      return false;
    }

    const validated = validateProfileSchema(incoming, this._compatEnvelope());
    if (!validated.ok) {
      if (validated.reason === 'VERSION_MISMATCH') {
        this._notifyCompatibilitySkip(source, incoming);
      }
      return false;
    }
    const cleanIncoming = validated.profile;

    const delTs = this.tombstones.get(cleanIncoming.id) || 0;
    if (cleanIncoming.updatedAt <= delTs) return false;

    const existing = this.profiles.get(cleanIncoming.id);
    if (existing && existing.updatedAt > cleanIncoming.updatedAt) {
      return false;
    }

    const mergedCandidate = existing
      ? {
          ...existing,
          ...cleanIncoming,
          password: cleanIncoming.password ?? existing.password ?? null,
          arena: {
            ...(existing.arena || {}),
            ...(cleanIncoming.arena || {}),
          },
        }
      : cleanIncoming;

    const mergedCheck = validateProfileSchema(mergedCandidate, this._compatEnvelope());
    if (!mergedCheck.ok) return false;

    this.profiles.set(mergedCheck.profile.id, mergedCheck.profile);
    this._emitChange();
    return true;
  }

  _applyDelete(id, ts) {
    if (typeof id !== 'string' || !ID_RE.test(id)) return;
    this.tombstones.set(id, finiteTs(ts));
    if (this.profiles.delete(id)) {
      this._emitChange();
    } else {
      this._saveToLocalStorage();
    }
  }

  _applyClearAll(ts) {
    const cleanTs = finiteTs(ts);
    for (const id of this.profiles.keys()) {
      this.tombstones.set(id, cleanTs);
    }
    this.profiles.clear();
    this._emitChange();
  }

  getProfiles() {
    return Array.from(this.profiles.values()).sort((a, b) => {
      if (Boolean(a.isSample) !== Boolean(b.isSample)) return a.isSample ? 1 : -1;
      return (a.createdAt || 0) - (b.createdAt || 0);
    });
  }

  findProfileByName(name) {
    const norm = normalizeProfileName(name).toLowerCase();
    if (!norm) return null;
    for (const p of this.profiles.values()) {
      if (!p.isSample && typeof p.name === 'string' && normalizeProfileName(p.name).toLowerCase() === norm) {
        return p;
      }
    }
    return null;
  }

  upsertProfile(profile) {
    const now = Date.now();
    const candidate = {
      ...profile,
      ...this._compatEnvelope(),
      createdAt: typeof profile?.createdAt === 'number' && Number.isFinite(profile.createdAt) ? profile.createdAt : now,
      updatedAt: now,
    };
    const check = validateProfileSchema(candidate, this._compatEnvelope());
    if (!check.ok) {
      return null;
    }
    const updated = check.profile;

    this.tombstones.delete(updated.id);
    this.profiles.set(updated.id, updated);
    this._emitChange();

    if (this.bc) {
      try {
        this.bc.postMessage({ ...this._compatEnvelope(), type: 'upsert', profile: updated });
      } catch {
        // ignore
      }
    }
    if (!updated.isSample && this.client && this.client.connected) {
      try {
        this.client.publish(this._profileTopic(updated.id), JSON.stringify(updated), {
          qos: 1,
          retain: true,
        });
      } catch {
        // ignore
      }
    }
    return updated;
  }

  deleteProfile(id) {
    if (typeof id !== 'string' || !ID_RE.test(id)) return;
    const now = Date.now();
    this._applyDelete(id, now);
    const evPayload = { ...this._compatEnvelope(), type: 'delete', id, ts: now };
    if (this.bc) {
      try {
        this.bc.postMessage(evPayload);
      } catch {
        // ignore
      }
    }
    if (this.client && this.client.connected) {
      try {
        this.client.publish(this._profileTopic(id), '', { qos: 1, retain: true });
        this.client.publish(this._eventTopic(), JSON.stringify(evPayload), { qos: 1 });
      } catch {
        // ignore
      }
    }
  }

  clearAllProfiles() {
    const now = Date.now();
    const idsToClear = Array.from(this.profiles.values())
      .filter((p) => !p.isSample)
      .map((p) => p.id);
    this._applyClearAll(now);
    const evPayload = { ...this._compatEnvelope(), type: 'clear_all', ts: now };
    if (this.bc) {
      try {
        this.bc.postMessage(evPayload);
      } catch {
        // ignore
      }
    }
    if (this.client && this.client.connected) {
      try {
        for (const id of idsToClear) {
          this.client.publish(this._profileTopic(id), '', { qos: 1, retain: true });
        }
        this.client.publish(this._eventTopic(), JSON.stringify(evPayload), { qos: 1 });
      } catch {
        // ignore
      }
    }
  }

  loadSamples(sampleProfiles) {
    const now = Date.now();
    let added = false;
    for (const sp of sampleProfiles || []) {
      if (!isSafePlainObject(sp)) continue;
      const candidate = {
        ...sp,
        ...this._compatEnvelope(),
        isSample: true,
        createdAt: now,
        updatedAt: now,
      };
      const check = validateProfileSchema(candidate, {
        ...this._compatEnvelope(),
        allowUnstampedSample: true,
      });
      if (check.ok) {
        this.tombstones.delete(check.profile.id);
        this.profiles.set(check.profile.id, check.profile);
        added = true;
      }
    }
    if (added) this._emitChange();
  }

  removeSamples() {
    let changed = false;
    for (const [id, p] of this.profiles.entries()) {
      if (p.isSample) {
        this.profiles.delete(id);
        changed = true;
      }
    }
    if (changed) this._emitChange();
  }

  hasSamples() {
    for (const p of this.profiles.values()) {
      if (p.isSample) return true;
    }
    return false;
  }

  getShareUrl() {
    if (typeof window === 'undefined' || !window.location) return '';
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    return url.toString();
  }
}
