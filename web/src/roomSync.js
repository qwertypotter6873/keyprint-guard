import mqtt from 'mqtt';
import { BEHAVIOR_FEATURES } from './features.js';

const DEFAULT_BROKERS = [
  'wss://broker.emqx.io:8084/mqtt',
  'wss://broker.hivemq.com:8884/mqtt',
];
const MQTT_CONNECT_MS = 4500;
const MQTT_RETRY_MS = 4000;
const MQTT_MAX_RETRY_MS = 30000;

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
  const norm = Math.sqrt(sumSq);
  if (!Number.isFinite(norm) || Math.abs(norm - 1) > 0.01) return null;
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
    if (profile.revisionId !== undefined && (typeof profile.revisionId !== 'string' || !ID_RE.test(profile.revisionId))) {
      return { ok: false, reason: 'INVALID_REVISION_ID' };
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
        ...(profile.revisionId === undefined ? {} : { revisionId: profile.revisionId }),
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

const revisionIdOf = (profile) => profile.revisionId || 'legacy';

function compareProfileRevisions(a, b) {
  const order = a.updatedAt - b.updatedAt;
  if (order) return order;
  const aId = revisionIdOf(a);
  const bId = revisionIdOf(b);
  return aId === bId ? 0 : aId > bId ? 1 : -1;
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
    openMqtt = mqtt.connect,
  } = {}) {
    this.onChange = onChange;
    this.onStatusChange = onStatusChange;
    this.onCompatibilitySkip = onCompatibilitySkip;
    this.schemaVersion = schemaVersion;
    this.preprocessVersion = preprocessVersion;
    this.modelHash = modelHash;
    this.customBrokers = brokers;
    this.openMqtt = openMqtt;

    this.clientId = `kp_${Math.random().toString(36).slice(2, 10)}`;
    this.roomCode = 'LOCAL';
    this.shared = false;
    this.pendingUpdates = new Set();
    this.profiles = new Map(); // id -> validated profile object
    this.tombstones = new Map(); // id -> deleted timestamp
    this.peers = new Map(); // clientId -> timestamp
    this.client = null;
    this.bc = null;
    this.session = null;
    this.storageListener = null;
    this.heartbeatTimer = null;
    this.connectTimer = null;
    this.retryTimer = null;
    this.retryDelay = MQTT_RETRY_MS;
    this.mqttReady = false;
    this.inFlightProfiles = new Set();
    this.retainedRevisions = new Map();
    this.retainedDeletionTimes = new Map();
    this.status = 'local'; // 'connecting' | 'online' | 'local'
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
    const previous = this._compatEnvelope();
    this.schemaVersion = schemaVersion;
    if (preprocessVersion) this.preprocessVersion = preprocessVersion;
    if (modelHash) this.modelHash = modelHash;
    if (this.shared && this.session && !this._isCompatibleEnvelope(previous)) {
      this.connect(this.roomCode, { shared: true });
      return;
    }

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

  connect(roomCode, { shared = false } = {}) {
    this.disconnect();
    this.shared = Boolean(shared);
    this.roomCode = this.shared ? sanitizeRoomCode(roomCode) : 'LOCAL';
    this.session = {};
    this.retryDelay = MQTT_RETRY_MS;
    this.compatSkipCount = 0;

    try {
      if (this.shared && typeof localStorage !== 'undefined') {
        localStorage.setItem('keyprint_last_room', this.roomCode);
      }
    } catch {
      // Ignore storage errors
    }

    if (typeof window !== 'undefined' && window.location && window.history?.replaceState) {
      try {
        const url = new URL(window.location.href);
        if (this.shared) url.searchParams.set('room', this.roomCode);
        else url.searchParams.delete('room');
        window.history.replaceState({}, '', url.toString());
      } catch {
        // Ignore URL update errors
      }
    }

    this.profiles.clear();
    this.tombstones.clear();
    this.pendingUpdates.clear();
    this.peers.clear();
    this.peers.set(this.clientId, Date.now());

    if (this.shared) {
      this._loadSharedStorage();
      this._listenToSharedStorage();
    } else {
      this._loadFromLocalStorage();
    }
    this._setupBroadcastChannel();
    if (this.shared) this._connectMqtt(0);
    else this._setStatus('local');
    this._emitChange();
  }

  disconnect() {
    this.session = null;
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
    if (this.storageListener) {
      window.removeEventListener('storage', this.storageListener);
      this.storageListener = null;
    }
    const channel = this.bc;
    this.bc = null;
    channel?.close();
    this._closeMqtt({ announceDeparture: true });
  }

  _closeMqtt({ announceDeparture = false } = {}) {
    const client = this.client;
    this.client = null;
    this.mqttReady = false;
    clearTimeout(this.connectTimer);
    this.connectTimer = null;
    this.inFlightProfiles.clear();
    this.retainedRevisions.clear();
    this.retainedDeletionTimes.clear();
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (!client) return;
    if (announceDeparture && client.connected) {
      try {
        client.publish(this._peerTopic(this.clientId), '', { qos: 0, retain: true });
      } catch { /* The broker will also clear presence from the connection's will. */ }
    }
    try {
      client.end(true);
    } catch { /* Ignore cleanup errors from a closed transport. */ }
  }

  _storageKey() {
    return this.shared ? `keyprint_mvp_v4_room_${this.roomCode}` : 'keyprint_mvp_v4_local';
  }

  _profileTopic(id = '+') {
    return `${TOPIC_NS}/room/${this.roomCode}/profiles/${id}`;
  }

  _profileRevisionTopic(profile) {
    return profile
      ? `${this._profileTopic(profile.id)}/${profile.updatedAt}/${revisionIdOf(profile)}`
      : `${this._profileTopic()}/+/+`;
  }

  _deletionTopic(id = '+', ts = '+') {
    // Timestamp in the retained topic prevents an older offline delete from replacing a newer one.
    return `${TOPIC_NS}/room/${this.roomCode}/deletions/${id}/${ts}`;
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

    for (const [id, ts] of this.tombstones) {
      if ((this.profiles.get(id)?.updatedAt ?? Infinity) <= ts) this.profiles.delete(id);
    }
    for (const id of Array.isArray(parsed.pendingUpdates) ? parsed.pendingUpdates : []) {
      if (typeof id === 'string' && this.profiles.has(id)) this.pendingUpdates.add(id);
    }
    if (hadInvalid) {
      this._saveToLocalStorage();
    }
    return true;
  }

  _saveToLocalStorage() {
    if (this.shared || !this.preprocessVersion || !this.modelHash) return;
    try {
      if (typeof localStorage === 'undefined') return;
      const payload = {
        ...this._compatEnvelope(),
        profiles: Array.from(this.profiles.values()),
        tombstones: Object.fromEntries(this.tombstones.entries()),
        pendingUpdates: [...this.pendingUpdates],
      };
      localStorage.setItem(this._storageKey(), JSON.stringify(payload));
    } catch {
      // Ignore storage quota / sandbox errors
    }
  }

  _recordsPrefix() {
    const compatibility = JSON.stringify([this.schemaVersion, this.preprocessVersion, this.modelHash]);
    return `${this._storageKey()}_records_${encodeURIComponent(compatibility)}/`;
  }

  _recordKey(record) {
    if (record.type === 'upsert') {
      const p = record.profile;
      return `${this._recordsPrefix()}${p.id}/upsert/${p.updatedAt}/${revisionIdOf(p)}`;
    }
    return `${this._recordsPrefix()}${record.id}/delete/${record.ts}`;
  }

  _recordKeys(prefix) {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(prefix)) keys.push(key);
    }
    return keys;
  }

  _persistRecord(record) {
    if (!this.shared || !this.preprocessVersion || !this.modelHash) return false;
    try {
      const key = this._recordKey(record);
      if (localStorage.getItem(key) !== null) return true;
      localStorage.setItem(key, JSON.stringify(record));
      this._compactRecords(record.profile?.id ?? record.id);
      return true;
    } catch {
      return false;
    }
  }

  _compactRecords(id) {
    // Different revisions never overwrite each other. Remove earlier revisions once newer state is durable.
    const prefix = `${this._recordsPrefix()}${id}/`;
    const records = this._recordKeys(prefix).map((key) => {
      const [kind, timestamp, revisionId] = key.slice(prefix.length).split('/');
      return { key, kind, updatedAt: Number(timestamp), revisionId };
    }).filter(({ updatedAt }) => Number.isFinite(updatedAt));
    let latestProfile = null;
    let latestDelete = null;
    for (const record of records) {
      if (record.kind === 'upsert' && (!latestProfile || compareProfileRevisions(record, latestProfile) > 0)) latestProfile = record;
      if (record.kind === 'delete' && (!latestDelete || record.updatedAt > latestDelete.updatedAt)) latestDelete = record;
    }
    if ((latestProfile?.updatedAt ?? Infinity) <= (latestDelete?.updatedAt ?? -1)) latestProfile = null;
    for (const record of records) {
      if (record !== latestProfile && record !== latestDelete) localStorage.removeItem(record.key);
    }
  }

  _loadSharedStorage() {
    if (!this.preprocessVersion || !this.modelHash) return;
    // Upgrade existing v4 room snapshots once all their valid records have been copied.
    if (this._loadFromLocalStorage()) {
      const records = [...this.profiles.values()].map((profile) => ({
        type: 'upsert', profile, pending: this.pendingUpdates.has(profile.id),
      }));
      for (const [id, ts] of this.tombstones) records.push({ type: 'delete', id, ts });
      if (records.every((record) => this._persistRecord(record))) {
        try { localStorage.removeItem(this._storageKey()); } catch { /* Keep the legacy copy until writable. */ }
      }
    }

    try {
      for (const key of this._recordKeys(this._recordsPrefix())) {
        this._applyStoredRecord(key, localStorage.getItem(key), { emit: false });
      }
    } catch { /* The room also works without storage. */ }
  }

  _applyStoredRecord(key, raw, { emit = true } = {}) {
    let record;
    try { record = JSON.parse(raw); } catch { return; }
    if (!isSafePlainObject(record)) return;
    if (record.type === 'upsert' && isSafePlainObject(record.profile) && this._recordKey(record) === key) {
      this._mergeIncomingProfile(record.profile, 'storage', { pending: Boolean(record.pending), persist: false, emit });
      if (emit && record.pending) this._publishPendingProfile(record.profile.id);
    } else if (record.type === 'delete' && typeof record.id === 'string' && typeof record.ts === 'number' &&
        Number.isFinite(record.ts) && record.ts >= 0 && this._recordKey(record) === key) {
      const applied = this._applyDelete(record.id, record.ts, { persist: false, emit });
      if (emit && applied && record.pending) this._publishDeletion(record.id, record.ts);
    }
  }

  _listenToSharedStorage() {
    if (typeof window === 'undefined') return;
    const session = this.session;
    this.storageListener = (event) => {
      if (this.session !== session || !event.key?.startsWith(this._recordsPrefix())) return;
      if (event.newValue !== null) {
        this._applyStoredRecord(event.key, event.newValue);
        return;
      }
      let record;
      try { record = JSON.parse(event.oldValue); } catch { return; }
      const current = this.profiles.get(record?.profile?.id);
      if (current?.isSample && this._recordKey({ type: 'upsert', profile: current }) === event.key) {
        this.profiles.delete(current.id);
        this._emitChange();
      }
    };
    window.addEventListener('storage', this.storageListener);
  }

  _readPendingProfile(profile) {
    try {
      const key = this._recordKey({ type: 'upsert', profile });
      const raw = localStorage.getItem(key);
      return raw === null ? null : Boolean(JSON.parse(raw)?.pending);
    } catch {
      return null;
    }
  }

  _acknowledgeProfile(profile) {
    try {
      const key = this._recordKey({ type: 'upsert', profile });
      const record = JSON.parse(localStorage.getItem(key));
      if (record?.pending) {
        localStorage.setItem(key, JSON.stringify({ ...record, pending: false }));
        this._compactRecords(profile.id);
      }
    } catch { /* A storage failure leaves that revision safe to retry. */ }
    this._acceptProfileAcknowledgment(profile);
    this._postBroadcast({ type: 'published', id: profile.id, updatedAt: profile.updatedAt, revisionId: revisionIdOf(profile) });
  }

  _acceptProfileAcknowledgment(profile) {
    const current = this.profiles.get(profile.id);
    if (current && compareProfileRevisions(current, profile) === 0) this.pendingUpdates.delete(profile.id);
  }

  _postBroadcast(message) {
    try {
      this.bc?.postMessage({ ...this._compatEnvelope(), ...message });
    } catch { /* Storage and MQTT can still be used. */ }
  }

  _setupBroadcastChannel() {
    if (typeof BroadcastChannel === 'undefined') return;
    try {
      const channel = new BroadcastChannel(this.shared ? `keyprint_bc_v4_${this.roomCode}` : 'keyprint_bc_v4_local');
      this.bc = channel;
      channel.onmessage = (ev) => {
        if (this.bc !== channel) return;
        const msg = ev?.data;
        if (!isSafePlainObject(msg)) return;
        if (!this._isCompatibleEnvelope(msg)) {
          this._notifyCompatibilitySkip('broadcast', msg);
          return;
        }
        if (msg.type === 'upsert' && msg.profile) {
          const pending = this.shared && msg.pending ? this._readPendingProfile(msg.profile) ?? true : undefined;
          this._mergeIncomingProfile(msg.profile, 'broadcast', { pending });
          if (pending) this._publishPendingProfile(msg.profile.id);
        } else if (msg.type === 'delete' && typeof msg.id === 'string' && ID_RE.test(msg.id)) {
          const ts = finiteTs(msg.ts);
          if (this._applyDelete(msg.id, ts) && msg.pending) this._publishDeletion(msg.id, ts);
        } else if (msg.type === 'published' && Number.isFinite(msg.updatedAt)) {
          this._acceptProfileAcknowledgment(msg);
        } else if (msg.type === 'clear_all') {
          this._applyClearAll(finiteTs(msg.ts));
        }
      };
    } catch {
      this.bc = null;
    }
  }

  _connectMqtt(brokerIndex, session = this.session) {
    if (!session || this.session !== session || !this.shared) return;
    const brokers = this._brokers();
    if (brokers.length === 0) {
      this._setStatus('local');
      return;
    }
    brokerIndex %= brokers.length;
    this._setStatus('connecting');
    if (this.session !== session) return;

    const myPeerTopic = this._peerTopic(this.clientId);
    let client;
    try {
      client = this.openMqtt(brokers[brokerIndex], {
        clientId: `${this.clientId}_${Math.random().toString(36).slice(2, 6)}`,
        clean: true,
        connectTimeout: MQTT_CONNECT_MS,
        reconnectPeriod: 0, // One retry loop handles both startup and post-connect failures.
        will: {
          topic: myPeerTopic,
          payload: '',
          qos: 0,
          retain: true,
        },
      });
    } catch {
      this._scheduleRetry(session, brokerIndex, brokers.length);
      return;
    }
    this.client = client;

    const isCurrent = () => this.session === session && this.client === client;
    const failed = () => {
      if (!isCurrent()) return;
      this._closeMqtt();
      this._scheduleRetry(session, brokerIndex, brokers.length);
    };
    this.connectTimer = setTimeout(failed, MQTT_CONNECT_MS + 500);

    client.on('connect', () => {
      if (!isCurrent()) return;
      try {
        client.subscribe(
          [this._profileTopic(), this._profileRevisionTopic(), this._deletionTopic(), this._eventTopic(), this._peerTopic()],
          { qos: 1 },
          (err, grants) => {
            if (!isCurrent()) return;
            if (err || (Array.isArray(grants) && grants.some(({ qos }) => qos >= 128))) {
              failed();
              return;
            }
            clearTimeout(this.connectTimer);
            this.connectTimer = null;
            this.retryDelay = MQTT_RETRY_MS;
            this.mqttReady = true;
            this._setStatus('online');
            if (!isCurrent()) return;
            this._flushMqtt();
            for (const id of this.retainedRevisions.keys()) this._pruneRetainedRevisions(id);
            if (isCurrent()) this.heartbeatTimer = setInterval(() => this._flushMqtt(), 18000);
          }
        );
      } catch {
        failed();
      }
    });

    client.on('message', (topic, buf, packet) => {
      if (isCurrent()) this._onMqttMessage(topic, buf?.toString() || '', packet);
    });
    client.on('close', failed);
    client.on('offline', failed);
    client.on('error', failed);
  }

  _scheduleRetry(session, brokerIndex, brokerCount) {
    if (this.session !== session) return;
    const lastBroker = brokerIndex + 1 >= brokerCount;
    const delay = lastBroker ? this.retryDelay : 0;
    if (lastBroker) this.retryDelay = Math.min(this.retryDelay * 2, MQTT_MAX_RETRY_MS);
    this.peers.clear();
    this.peers.set(this.clientId, Date.now());
    this._setStatus('local');
    if (this.session !== session) return;
    this.retryTimer = setTimeout(() => {
      if (this.session !== session) return;
      this.retryTimer = null;
      this._connectMqtt(lastBroker ? 0 : brokerIndex + 1, session);
    }, delay);
  }

  _onMqttMessage(topic, text, packet) {
    const client = this.client;
    const prefix = `${TOPIC_NS}/room/${this.roomCode}/`;
    if (!topic.startsWith(prefix)) return;
    const [kind, id, timestamp, revisionId, ...extra] = topic.slice(prefix.length).split('/');
    if (extra.length > 0) return;
    if (kind === 'peers' && ID_RE.test(id || '') && timestamp === undefined) {
      if (!text) {
        if (id !== this.clientId) this.peers.delete(id);
      } else {
        try {
          const data = JSON.parse(text);
          if (isSafePlainObject(data) && this._isCompatibleEnvelope(data) && data.cid === id) {
            this.peers.set(id, finiteTs(data.ts));
          }
        } catch { /* Ignore malformed presence messages. */ }
      }
      this._emitStatus();
      return;
    }

    if (!text) return;
    try {
      const payload = JSON.parse(text);
      if (!isSafePlainObject(payload)) return;
      if (kind === 'deletions' && revisionId === undefined) {
        if (this._isCompatibleEnvelope(payload) && payload.type === 'delete' && payload.id === id &&
            String(payload.ts) === timestamp && Number.isFinite(payload.ts) && payload.ts >= 0) {
          this._applyDelete(id, payload.ts);
          if (this.client === client && packet?.retain) this._rememberRetainedDeletion(id, payload.ts);
        }
      } else if (topic === this._eventTopic()) {
        if (!this._isCompatibleEnvelope(payload)) {
          this._notifyCompatibilitySkip('mqtt_event', payload);
        } else if (payload.type === 'delete') {
          this._applyDelete(payload.id, finiteTs(payload.ts));
        } else if (payload.type === 'clear_all') {
          this._applyClearAll(finiteTs(payload.ts));
        }
      } else if (kind === 'profiles' && payload.id === id) {
        const profile = this._validateIncomingProfile(payload, 'mqtt');
        if (!profile) return;
        const versioned = revisionId !== undefined;
        if (timestamp !== undefined && (!versioned || this._profileRevisionTopic(profile) !== topic)) return;
        this._acceptProfile(profile);
        if (this.client === client && versioned) this._rememberRetainedRevision(profile, Boolean(packet?.retain));
      }
    } catch { /* Ignore malformed MQTT messages. */ }
  }

  _rememberRetainedDeletion(id, ts) {
    this.retainedDeletionTimes.set(id, Math.max(this.retainedDeletionTimes.get(id) ?? -1, ts));
    this._pruneRetainedRevisions(id);
  }

  _rememberRetainedRevision(profile, confirmed) {
    const revisions = this.retainedRevisions.get(profile.id) || new Map();
    const topic = this._profileRevisionTopic(profile);
    const revision = revisions.get(topic) || { updatedAt: profile.updatedAt, revisionId: revisionIdOf(profile), clearing: false };
    revision.confirmed ||= confirmed;
    revisions.set(topic, revision);
    this.retainedRevisions.set(profile.id, revisions);
    this._pruneRetainedRevisions(profile.id);
  }

  _flushMqtt() {
    for (const [id, ts] of this.tombstones) {
      if (ts > (this.retainedDeletionTimes.get(id) ?? -1)) this._publishDeletion(id, ts);
    }
    for (const id of this.pendingUpdates) this._publishPendingProfile(id);
    this._publishPresence();
  }

  _pruneRetainedRevisions(id) {
    const client = this.client;
    const revisions = this.retainedRevisions.get(id);
    if (!this.mqttReady || !client?.connected || !revisions) return;
    const deletedAt = this.retainedDeletionTimes.get(id) ?? -1;
    let latest = null;
    for (const revision of revisions.values()) {
      if (revision.confirmed && revision.updatedAt > deletedAt && (!latest || compareProfileRevisions(revision, latest) > 0)) {
        latest = revision;
      }
    }
    for (const [topic, revision] of revisions) {
      const obsolete = revision.updatedAt <= deletedAt || (latest && compareProfileRevisions(revision, latest) < 0);
      if (!obsolete || revision.clearing) continue;
      // An immutable revision topic cannot contain a newer profile by the time we clear it.
      revision.clearing = true;
      try {
        client.publish(topic, '', { qos: 1, retain: true }, (err) => {
          if (this.client !== client) return;
          revision.clearing = false;
          if (!err && revisions.get(topic) === revision) {
            revisions.delete(topic);
            if (revisions.size === 0) this.retainedRevisions.delete(id);
          }
        });
      } catch {
        revision.clearing = false;
      }
    }
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
        shared: this.shared,
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

  _validateIncomingProfile(incoming, source) {
    if (!isSafePlainObject(incoming)) return null;
    if (!this._isCompatibleEnvelope(incoming)) {
      this._notifyCompatibilitySkip(source, incoming);
      return null;
    }

    const validated = validateProfileSchema(incoming, this._compatEnvelope());
    if (!validated.ok) {
      if (validated.reason === 'VERSION_MISMATCH') {
        this._notifyCompatibilitySkip(source, incoming);
      }
      return null;
    }
    return validated.profile;
  }

  _mergeIncomingProfile(incoming, source = 'remote', options) {
    const profile = this._validateIncomingProfile(incoming, source);
    return profile ? this._acceptProfile(profile, options) : false;
  }

  _acceptProfile(profile, { pending, persist = true, emit = true } = {}) {
    const id = profile.id;
    if (profile.updatedAt <= (this.tombstones.get(id) ?? -1)) return false;

    const existing = this.profiles.get(id);
    if (existing) {
      const order = compareProfileRevisions(profile, existing);
      if (order < 0) return false;
      if (order === 0) {
        if (pending === true && !profile.isSample) this.pendingUpdates.add(id);
        if (pending === false) this.pendingUpdates.delete(id);
        return false;
      }
    }

    const updated = existing && !profile.revisionId
      ? {
          ...existing,
          ...profile,
          revisionId: undefined,
          password: profile.password ?? existing.password ?? null,
          arena: { ...existing.arena, ...profile.arena },
        }
      : profile;

    this.profiles.set(id, updated);
    if (pending && !updated.isSample) this.pendingUpdates.add(id);
    else this.pendingUpdates.delete(id);
    if (persist) this._persistRecord({ type: 'upsert', profile: updated, pending: Boolean(pending) });
    if (emit) this._emitChange();
    return true;
  }

  _applyDelete(id, ts, { pending = false, persist = true, emit = true } = {}) {
    if (typeof id !== 'string' || !ID_RE.test(id)) return false;
    const cleanTs = finiteTs(ts);
    if (cleanTs <= (this.tombstones.get(id) ?? -1)) return false;
    this.tombstones.set(id, cleanTs);
    if (persist) this._persistRecord({ type: 'delete', id, ts: cleanTs, pending });
    const current = this.profiles.get(id);
    if (current && current.updatedAt <= cleanTs) {
      this.profiles.delete(id);
      this.pendingUpdates.delete(id);
      if (emit) this._emitChange();
    } else if (emit) this._saveToLocalStorage();
    return true;
  }

  _applyClearAll(ts) {
    for (const id of [...this.profiles.keys()]) this._applyDelete(id, ts);
  }

  _publishDeletion(id, ts) {
    const client = this.client;
    if (!this.shared || !this.mqttReady || !client?.connected) return;
    const ev = { ...this._compatEnvelope(), type: 'delete', id, ts };
    const payload = JSON.stringify(ev);
    try {
      client.publish(this._deletionTopic(id, ts), payload, { qos: 1, retain: true }, (err) => {
        if (!err && this.client === client) this._rememberRetainedDeletion(id, ts);
      });
      client.publish(this._eventTopic(), payload, { qos: 1 });
    } catch { /* The persisted tombstone is replayed after reconnection. */ }
  }

  _publishPendingProfile(id) {
    const p = this.profiles.get(id);
    const client = this.client;
    if (!this.shared || !this.mqttReady || !client?.connected || !p || p.isSample || !this.pendingUpdates.has(id)) return;
    if (p.updatedAt <= (this.tombstones.get(id) ?? -1)) return;
    const topic = this._profileRevisionTopic(p);
    if (this.inFlightProfiles.has(topic)) return;
    if (this._readPendingProfile(p) === false) {
      this.pendingUpdates.delete(id);
      return;
    }
    this.inFlightProfiles.add(topic);
    try {
      client.publish(topic, JSON.stringify(p), { qos: 1, retain: true }, (err) => {
        if (this.client !== client) return;
        this.inFlightProfiles.delete(topic);
        if (!err) {
          this._acknowledgeProfile(p);
          this._rememberRetainedRevision(p, true);
        }
      });
    } catch {
      this.inFlightProfiles.delete(topic);
    }
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
    const existing = this.profiles.get(profile?.id);
    const now = Math.max(Date.now(), (existing?.updatedAt || 0) + 1, (this.tombstones.get(profile?.id) || 0) + 1);
    const candidate = {
      ...profile,
      ...this._compatEnvelope(),
      createdAt: typeof profile?.createdAt === 'number' && Number.isFinite(profile.createdAt) ? profile.createdAt : now,
      updatedAt: now,
      ...(this.shared ? { revisionId: this.clientId } : {}),
    };
    const check = validateProfileSchema(candidate, this._compatEnvelope());
    if (!check.ok) {
      return null;
    }
    const updated = check.profile;

    if (updated.updatedAt <= (this.tombstones.get(updated.id) || 0)) return null;
    const pending = this.shared && !updated.isSample;
    this.profiles.set(updated.id, updated);
    if (pending) this.pendingUpdates.add(updated.id);
    this._persistRecord({ type: 'upsert', profile: updated, pending });
    this._emitChange();
    this._postBroadcast({ type: 'upsert', profile: updated, pending });
    this._publishPendingProfile(updated.id);
    return updated;
  }

  deleteProfile(id) {
    if (typeof id !== 'string' || !ID_RE.test(id)) return;
    if (this.profiles.get(id)?.isSample) {
      this._removeSample(this.profiles.get(id));
      this._emitChange();
      return;
    }
    const now = Math.max(Date.now(), (this.profiles.get(id)?.updatedAt || 0) + 1, (this.tombstones.get(id) || 0) + 1);
    this._applyDelete(id, now, { pending: this.shared });
    this._postBroadcast({ type: 'delete', id, ts: now, pending: this.shared });
    this._publishDeletion(id, now);
  }

  clearAllProfiles() {
    for (const id of [...this.profiles.keys()]) this.deleteProfile(id);
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
        updatedAt: Math.max(now, (this.profiles.get(sp.id)?.updatedAt || 0) + 1, (this.tombstones.get(sp.id) || 0) + 1),
        ...(this.shared ? { revisionId: this.clientId } : {}),
      };
      const check = validateProfileSchema(candidate, {
        ...this._compatEnvelope(),
        allowUnstampedSample: true,
      });
      if (check.ok) {
        this.profiles.set(check.profile.id, check.profile);
        this._persistRecord({ type: 'upsert', profile: check.profile, pending: false });
        added = true;
      }
    }
    if (added) this._emitChange();
  }

  removeSamples() {
    let changed = false;
    for (const p of this.profiles.values()) {
      if (p.isSample) {
        this._removeSample(p);
        changed = true;
      }
    }
    if (changed) this._emitChange();
  }

  _removeSample(profile) {
    this.profiles.delete(profile.id);
    if (this.shared) {
      try {
        localStorage.removeItem(this._recordKey({ type: 'upsert', profile }));
      } catch { /* Sample state can still be updated in memory. */ }
    }
  }

  hasSamples() {
    for (const p of this.profiles.values()) {
      if (p.isSample) return true;
    }
    return false;
  }

  getShareUrl() {
    if (!this.shared || typeof window === 'undefined' || !window.location) return '';
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    return url.toString();
  }
}
