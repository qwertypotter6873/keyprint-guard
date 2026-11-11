import mqtt from 'mqtt';

const BROKERS = [
  'wss://broker.emqx.io:8084/mqtt',
  'wss://broker.hivemq.com:8884/mqtt',
];

const TOPIC_NS = 'keyprint-agy-mvp-v2';

export function sanitizeRoomCode(raw) {
  const cleaned = String(raw || '')
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9가-힣_-]/g, '')
    .slice(0, 16);
  return cleaned || 'GACHON';
}

export function getInitialRoomCode() {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get('room');
  if (fromUrl) return sanitizeRoomCode(fromUrl);
  const saved = localStorage.getItem('keyprint_last_room');
  if (saved) return sanitizeRoomCode(saved);
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

/**
 * Multi-user real-time profile store synchronized across:
 * 1) Multiple devices / laptops simultaneously via MQTT-over-WSS retained topics
 * 2) Same-browser tabs via BroadcastChannel
 * 3) Offline / page reloads via localStorage
 */
export class RoomProfileSync {
  constructor({ onChange, onStatusChange }) {
    this.onChange = onChange;
    this.onStatusChange = onStatusChange;
    this.clientId = `kp_${Math.random().toString(36).slice(2, 10)}`;
    this.roomCode = 'GACHON';
    this.profiles = new Map(); // id -> profile object
    this.tombstones = new Map(); // id -> deleted timestamp
    this.peers = new Map(); // clientId -> timestamp
    this.client = null;
    this.bc = null;
    this.heartbeatTimer = null;
    this.brokerIdx = 0;
    this.status = 'connecting'; // 'connecting' | 'online' | 'local'
  }

  connect(roomCode) {
    this.disconnect();
    this.roomCode = sanitizeRoomCode(roomCode);
    localStorage.setItem('keyprint_last_room', this.roomCode);

    // Update URL query param cleanly so copying browser URL shares the exact room
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    window.history.replaceState({}, '', url.toString());

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
    if (this.bc) {
      this.bc.close();
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
    return `keyprint_mvp_room_${this.roomCode}`;
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
    try {
      const raw = localStorage.getItem(this._storageKey());
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.profiles)) {
        for (const p of parsed.profiles) {
          if (p && p.id) this.profiles.set(p.id, p);
        }
      }
      if (parsed && parsed.tombstones) {
        for (const [k, ts] of Object.entries(parsed.tombstones)) {
          this.tombstones.set(k, Number(ts));
        }
      }
    } catch {
      // Ignore malformed storage
    }
  }

  _saveToLocalStorage() {
    try {
      const payload = {
        profiles: Array.from(this.profiles.values()),
        tombstones: Object.fromEntries(this.tombstones.entries()),
      };
      localStorage.setItem(this._storageKey(), JSON.stringify(payload));
    } catch {
      // Ignore storage quota errors
    }
  }

  _setupBroadcastChannel() {
    if (typeof BroadcastChannel === 'undefined') return;
    this.bc = new BroadcastChannel(`keyprint_bc_${this.roomCode}`);
    this.bc.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'upsert' && msg.profile) {
        this._mergeIncomingProfile(msg.profile);
      } else if (msg.type === 'delete' && msg.id) {
        this._applyDelete(msg.id, msg.ts || Date.now());
      }
    };
  }

  _connectMqtt(brokerIndex) {
    this.brokerIdx = brokerIndex % BROKERS.length;
    const brokerUrl = BROKERS[this.brokerIdx];
    this._setStatus('connecting');

    const myPeerTopic = this._peerTopic(this.clientId);
    const client = mqtt.connect(brokerUrl, {
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
    this.client = client;

    let connectedOnce = false;
    const failoverTimer = setTimeout(() => {
      if (!connectedOnce && this.client === client) {
        try {
          client.end(true);
        } catch {
          // ignore
        }
        if (brokerIndex + 1 < BROKERS.length) {
          this._connectMqtt(brokerIndex + 1);
        } else {
          this._setStatus('local');
        }
      }
    }, 4800);

    client.on('connect', () => {
      if (this.client !== client) return;
      connectedOnce = true;
      clearTimeout(failoverTimer);
      this._setStatus('online');

      client.subscribe(
        [this._profileTopic('+'), this._eventTopic(), this._peerTopic('+')],
        { qos: 1 },
        () => {
          // Re-broadcast any locally created non-sample profiles so room stays converged
          for (const p of this.profiles.values()) {
            if (!p.isSample) {
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
      const text = buf.toString();

      if (topic.startsWith(`${TOPIC_NS}/room/${this.roomCode}/peers/`)) {
        const cid = topic.split('/').pop();
        if (!text) {
          if (cid !== this.clientId) this.peers.delete(cid);
        } else {
          try {
            const data = JSON.parse(text);
            this.peers.set(cid, data.ts || Date.now());
          } catch {
            // ignore
          }
        }
        this._emitStatus();
        return;
      }

      if (topic === this._eventTopic()) {
        try {
          const ev = JSON.parse(text);
          if (ev.type === 'delete' && ev.id) {
            this._applyDelete(ev.id, ev.ts || Date.now());
          } else if (ev.type === 'clear_all') {
            this._applyClearAll(ev.ts || Date.now());
          }
        } catch {
          // ignore
        }
        return;
      }

      if (topic.startsWith(`${TOPIC_NS}/room/${this.roomCode}/profiles/`)) {
        const pid = topic.split('/').pop();
        if (!text) {
          this._applyDelete(pid, Date.now());
          return;
        }
        try {
          const profile = JSON.parse(text);
          if (profile && profile.id) {
            this._mergeIncomingProfile(profile);
          }
        } catch {
          // ignore
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
    // Prune stale peers (> 55s old)
    const cutoff = Date.now() - 55000;
    for (const [cid, ts] of this.peers.entries()) {
      if (cid !== this.clientId && ts < cutoff) {
        this.peers.delete(cid);
      }
    }
    if (this.client && this.client.connected) {
      this.client.publish(
        this._peerTopic(this.clientId),
        JSON.stringify({ cid: this.clientId, ts: Date.now() }),
        { qos: 0, retain: true }
      );
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
      });
    }
  }

  _emitChange() {
    this._saveToLocalStorage();
    this._emitStatus();
    if (this.onChange) {
      const list = Array.from(this.profiles.values()).sort((a, b) => {
        if (Boolean(a.isSample) !== Boolean(b.isSample)) return a.isSample ? 1 : -1;
        return (a.createdAt || 0) - (b.createdAt || 0);
      });
      this.onChange(list);
    }
  }

  _mergeIncomingProfile(incoming) {
    const delTs = this.tombstones.get(incoming.id) || 0;
    if ((incoming.updatedAt || 0) <= delTs) return;

    const existing = this.profiles.get(incoming.id);
    if (existing && (existing.updatedAt || 0) > (incoming.updatedAt || 0)) {
      return;
    }

    // Merge password & arena sub-profiles cleanly if both exist
    const merged = existing
      ? {
          ...existing,
          ...incoming,
          password: incoming.password ?? existing.password ?? null,
          arena: {
            ...(existing.arena || {}),
            ...(incoming.arena || {}),
          },
        }
      : incoming;

    this.profiles.set(merged.id, merged);
    this._emitChange();
  }

  _applyDelete(id, ts) {
    this.tombstones.set(id, ts);
    if (this.profiles.delete(id)) {
      this._emitChange();
    } else {
      this._saveToLocalStorage();
    }
  }

  _applyClearAll(ts) {
    for (const id of this.profiles.keys()) {
      this.tombstones.set(id, ts);
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
    const norm = String(name || '').trim().toLowerCase();
    if (!norm) return null;
    for (const p of this.profiles.values()) {
      if (!p.isSample && p.name.trim().toLowerCase() === norm) {
        return p;
      }
    }
    return null;
  }

  upsertProfile(profile) {
    const now = Date.now();
    const updated = {
      ...profile,
      createdAt: profile.createdAt || now,
      updatedAt: now,
    };
    this.tombstones.delete(updated.id);
    this.profiles.set(updated.id, updated);
    this._emitChange();

    if (this.bc) {
      this.bc.postMessage({ type: 'upsert', profile: updated });
    }
    if (!updated.isSample && this.client && this.client.connected) {
      this.client.publish(this._profileTopic(updated.id), JSON.stringify(updated), {
        qos: 1,
        retain: true,
      });
    }
    return updated;
  }

  deleteProfile(id) {
    const now = Date.now();
    this._applyDelete(id, now);
    if (this.bc) {
      this.bc.postMessage({ type: 'delete', id, ts: now });
    }
    if (this.client && this.client.connected) {
      // Empty retained message clears broker state per MQTT spec
      this.client.publish(this._profileTopic(id), '', { qos: 1, retain: true });
      this.client.publish(this._eventTopic(), JSON.stringify({ type: 'delete', id, ts: now }), {
        qos: 1,
      });
    }
  }

  loadSamples(sampleProfiles) {
    for (const sp of sampleProfiles || []) {
      const now = Date.now();
      const item = { ...sp, createdAt: now, updatedAt: now };
      this.tombstones.delete(item.id);
      this.profiles.set(item.id, item);
    }
    this._emitChange();
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
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    return url.toString();
  }
}
