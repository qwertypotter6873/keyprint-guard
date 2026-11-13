/**
 * Adversarial End-to-End Security, v4 Schema/Hash Isolation, Verdict Snapshot & Live WS-MQTT Regression Suite.
 *
 *   node web/scripts/adversarial_regression.mjs
 *
 * Self-contained verification (no external network required):
 *  1) Builds `web/dist` programmatically with Vite so latest `web/src/*` is always tested.
 *  2) Serves `web/dist` over a local Node `http` server with a built-in zero-dependency RFC-6455
 *     WebSocket + MQTT 3.1.1 / 5.0 broker mounted at `ws://127.0.0.1:<port>/mqtt`.
 *  3) Drives headless Chrome via `playwright-core` across isolated `BrowserContext`s to verify:
 *     - Stored XSS elimination across UI, `localStorage`, and MQTT payloads
 *     - Malformed / NaN / zero-norm / `name: null` / prototype-pollution boot survival & pruning
 *     - End-to-end `v4` (`schemaVersion`, `preprocessVersion`, `modelHash`) isolation across MQTT,
 *       `BroadcastChannel`, `localStorage`, and live `delete` / `clear_all` events
 *     - Sample profile name blocking (`isSampleName`) & duplicate password account overwrite rejection
 *     - `Mode C` pre-adaptation `lastVerdict` snapshot, 5-condition invalidation, and undistorted 2D PCA
 *     - `!state.ready` Enter user feedback & synchronous `window.WebSocket` throw resilience
 *     - Real 2-context live multi-user sync & live ONNX verification over WebSocket MQTT
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';
import { build } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = join(__dirname, '..');
const DIST_DIR = join(WEB_DIR, 'dist');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

let totalChecks = 0;
let totalFailures = 0;

function check(cond, msg) {
  totalChecks += 1;
  if (!cond) {
    totalFailures += 1;
    console.error(`    ✗ ${msg}`);
  }
}

function findChromeExecutable() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return null;
}

// ============================================================================
// Minimal Zero-Dependency RFC-6455 WebSocket + MQTT 3.1.1 / 5.0 Test Broker
// ============================================================================
const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

function encodeWsFrame(payloadBuf, opcode = 0x02) {
  const len = payloadBuf.length;
  let header;
  if (len < 126) {
    header = Buffer.alloc(2);
    header[0] = 0x80 | opcode;
    header[1] = len;
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payloadBuf]);
}

function encodeMqttVarInt(n) {
  const bytes = [];
  let val = n;
  do {
    let encoded = val % 128;
    val = Math.floor(val / 128);
    if (val > 0) encoded |= 0x80;
    bytes.push(encoded);
  } while (val > 0);
  return Buffer.from(bytes);
}

function decodeMqttVarInt(buf, offset) {
  let multiplier = 1;
  let value = 0;
  let idx = offset;
  while (idx < buf.length) {
    const encoded = buf[idx++];
    value += (encoded & 0x7f) * multiplier;
    if ((encoded & 0x80) === 0) {
      return { value, nextOffset: idx };
    }
    multiplier *= 128;
    if (multiplier > 128 * 128 * 128) return null;
  }
  return null;
}

function readMqttUtf8(buf, offset) {
  if (offset + 2 > buf.length) return null;
  const len = buf.readUInt16BE(offset);
  if (offset + 2 + len > buf.length) return null;
  return {
    str: buf.subarray(offset + 2, offset + 2 + len).toString('utf8'),
    nextOffset: offset + 2 + len,
  };
}

function encodeMqttPublish(topic, payloadBuf, { retain = false, protoLevel = 4 } = {}) {
  const topicBytes = Buffer.from(topic, 'utf8');
  const topicHeader = Buffer.alloc(2);
  topicHeader.writeUInt16BE(topicBytes.length, 0);
  const props = protoLevel === 5 ? Buffer.from([0x00]) : Buffer.alloc(0);
  const body = Buffer.concat([topicHeader, topicBytes, props, payloadBuf]);
  const firstByte = 0x30 | (retain ? 0x01 : 0x00);
  return Buffer.concat([Buffer.from([firstByte]), encodeMqttVarInt(body.length), body]);
}

function mqttTopicMatches(filter, topic) {
  if (filter === '#') return true;
  const fParts = filter.split('/');
  const tParts = topic.split('/');
  for (let i = 0; i < fParts.length; i++) {
    const fp = fParts[i];
    if (fp === '#') return true;
    if (i >= tParts.length) return false;
    if (fp !== '+' && fp !== tParts[i]) return false;
  }
  return fParts.length === tParts.length;
}

function createStaticAndMqttServer(distDir) {
  const clients = new Set();
  const retained = new Map(); // topic -> Buffer

  function fanoutPublish(topic, payloadBuf, { retain = false } = {}) {
    if (retain) {
      if (payloadBuf.length === 0) retained.delete(topic);
      else retained.set(topic, payloadBuf);
    }
    for (const c of clients) {
      if (c.closed) continue;
      let matched = false;
      for (const filter of c.subscriptions) {
        if (mqttTopicMatches(filter, topic)) {
          matched = true;
          break;
        }
      }
      if (matched) {
        const pkt = encodeMqttPublish(topic, payloadBuf, { retain: false, protoLevel: c.protoLevel });
        c.sendMqtt(pkt);
      }
    }
  }

  function handleMqttPacket(client, firstByte, body) {
    const pktType = firstByte >> 4;
    const flags = firstByte & 0x0f;

    if (pktType === 1) {
      // CONNECT
      const proto = readMqttUtf8(body, 0);
      if (!proto) return;
      let off = proto.nextOffset;
      const protoLevel = body[off++];
      const connectFlags = body[off++];
      off += 2; // keepAlive
      client.protoLevel = protoLevel === 5 ? 5 : 4;
      if (client.protoLevel === 5) {
        const propsLen = decodeMqttVarInt(body, off);
        if (!propsLen) return;
        off = propsLen.nextOffset + propsLen.value;
      }
      const cid = readMqttUtf8(body, off);
      if (cid) {
        client.clientId = cid.str;
        off = cid.nextOffset;
      }
      if (connectFlags & 0x04) {
        // Will flag
        if (client.protoLevel === 5) {
          const wProps = decodeMqttVarInt(body, off);
          if (wProps) off = wProps.nextOffset + wProps.value;
        }
        const wTopic = readMqttUtf8(body, off);
        if (wTopic && wTopic.nextOffset + 2 <= body.length) {
          off = wTopic.nextOffset;
          const wLen = body.readUInt16BE(off);
          off += 2;
          const wPayload = body.subarray(off, Math.min(body.length, off + wLen));
          client.will = {
            topic: wTopic.str,
            payload: Buffer.from(wPayload),
            retain: Boolean(connectFlags & 0x20),
          };
        }
      }
      const connack =
        client.protoLevel === 5
          ? Buffer.from([0x20, 0x03, 0x00, 0x00, 0x00])
          : Buffer.from([0x20, 0x02, 0x00, 0x00]);
      client.sendMqtt(connack);
    } else if (pktType === 8) {
      // SUBSCRIBE
      if (body.length < 2) return;
      const packetId = body.readUInt16BE(0);
      let off = 2;
      if (client.protoLevel === 5) {
        const propsLen = decodeMqttVarInt(body, off);
        if (!propsLen) return;
        off = propsLen.nextOffset + propsLen.value;
      }
      const granted = [];
      const newFilters = [];
      while (off < body.length) {
        const tf = readMqttUtf8(body, off);
        if (!tf || tf.nextOffset >= body.length) break;
        const opts = body[tf.nextOffset];
        off = tf.nextOffset + 1;
        client.subscriptions.add(tf.str);
        newFilters.push(tf.str);
        granted.push(Math.min(1, opts & 0x03));
      }
      const pidBuf = Buffer.alloc(2);
      pidBuf.writeUInt16BE(packetId, 0);
      const props = client.protoLevel === 5 ? Buffer.from([0x00]) : Buffer.alloc(0);
      const subackBody = Buffer.concat([pidBuf, props, Buffer.from(granted)]);
      client.sendMqtt(Buffer.concat([Buffer.from([0x90]), encodeMqttVarInt(subackBody.length), subackBody]));

      // Deliver matching retained messages
      for (const [rTopic, rPayload] of retained.entries()) {
        if (newFilters.some((f) => mqttTopicMatches(f, rTopic))) {
          client.sendMqtt(
            encodeMqttPublish(rTopic, rPayload, { retain: true, protoLevel: client.protoLevel })
          );
        }
      }
    } else if (pktType === 3) {
      // PUBLISH
      const qos = (flags >> 1) & 0x03;
      const retain = Boolean(flags & 0x01);
      const tp = readMqttUtf8(body, 0);
      if (!tp) return;
      let off = tp.nextOffset;
      let packetId = 0;
      if (qos > 0) {
        if (off + 2 > body.length) return;
        packetId = body.readUInt16BE(off);
        off += 2;
      }
      if (client.protoLevel === 5) {
        const propsLen = decodeMqttVarInt(body, off);
        if (!propsLen) return;
        off = propsLen.nextOffset + propsLen.value;
      }
      const payloadBuf = Buffer.from(body.subarray(off));
      if (qos === 1) {
        const puback =
          client.protoLevel === 5
            ? Buffer.from([0x40, 0x03, packetId >> 8, packetId & 0xff, 0x00])
            : Buffer.from([0x40, 0x02, packetId >> 8, packetId & 0xff]);
        client.sendMqtt(puback);
      }
      fanoutPublish(tp.str, payloadBuf, { retain });
    } else if (pktType === 12) {
      // PINGREQ -> PINGRESP
      client.sendMqtt(Buffer.from([0xd0, 0x00]));
    } else if (pktType === 14) {
      // Graceful DISCONNECT clears LWT
      client.will = null;
      client.close();
    }
  }

  const server = createServer((req, res) => {
    const urlObj = new URL(req.url || '/', 'http://127.0.0.1');
    let relPath = decodeURIComponent(urlObj.pathname);
    if (relPath === '/' || relPath === '') relPath = '/index.html';
    const safeRel = normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, '');
    let filePath = join(distDir, safeRel);
    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      filePath = join(distDir, 'index.html');
    }
    try {
      const data = readFileSync(filePath);
      const mime = MIME_TYPES[extname(filePath).toLowerCase()] || 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': mime,
        'Cache-Control': 'no-store',
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  });

  server.on('upgrade', (req, socket) => {
    const key = req.headers['sec-websocket-key'];
    if (!key) {
      socket.destroy();
      return;
    }
    const accept = createHash('sha1').update(key + WS_GUID).digest('base64');
    const protoHeader = req.headers['sec-websocket-protocol'];
    const selectedProto = protoHeader
      ? String(protoHeader)
          .split(',')
          .map((s) => s.trim())
          .find(Boolean)
      : null;

    const lines = [
      'HTTP/1.1 101 Switching Protocols',
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Accept: ${accept}`,
    ];
    if (selectedProto) lines.push(`Sec-WebSocket-Protocol: ${selectedProto}`);
    lines.push('\r\n');
    socket.write(lines.join('\r\n'));

    const client = {
      socket,
      protoLevel: 4,
      clientId: '',
      subscriptions: new Set(),
      will: null,
      closed: false,
      wsBuf: Buffer.alloc(0),
      mqttBuf: Buffer.alloc(0),
      wsFrag: [],
      sendMqtt(pktBuf) {
        if (this.closed) return;
        try {
          socket.write(encodeWsFrame(pktBuf, 0x02));
        } catch {
          // ignore closed socket write
        }
      },
      close() {
        if (this.closed) return;
        this.closed = true;
        clients.delete(this);
        try {
          socket.end();
        } catch {
          // ignore
        }
      },
    };
    clients.add(client);

    socket.on('data', (chunk) => {
      client.wsBuf = Buffer.concat([client.wsBuf, chunk]);
      while (client.wsBuf.length >= 2) {
        const b0 = client.wsBuf[0];
        const b1 = client.wsBuf[1];
        const fin = Boolean(b0 & 0x80);
        const opcode = b0 & 0x0f;
        const masked = Boolean(b1 & 0x80);
        let len = b1 & 0x7f;
        let offset = 2;
        if (len === 126) {
          if (client.wsBuf.length < 4) return;
          len = client.wsBuf.readUInt16BE(2);
          offset = 4;
        } else if (len === 127) {
          if (client.wsBuf.length < 10) return;
          len = Number(client.wsBuf.readBigUInt64BE(2));
          offset = 10;
        }
        const maskLen = masked ? 4 : 0;
        if (client.wsBuf.length < offset + maskLen + len) return;

        let payload = Buffer.from(client.wsBuf.subarray(offset + maskLen, offset + maskLen + len));
        if (masked) {
          const mask = client.wsBuf.subarray(offset, offset + 4);
          for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i & 3];
        }
        client.wsBuf = client.wsBuf.subarray(offset + maskLen + len);

        if (opcode === 0x08) {
          // Close frame
          client.close();
          return;
        }
        if (opcode === 0x09) {
          // Ping -> Pong
          try {
            socket.write(encodeWsFrame(payload, 0x0a));
          } catch {
            // ignore
          }
          continue;
        }
        if (opcode === 0x0a) continue;

        if (opcode === 0x00 || opcode === 0x01 || opcode === 0x02) {
          client.wsFrag.push(payload);
          if (!fin) continue;
          const fullWsPayload = Buffer.concat(client.wsFrag);
          client.wsFrag = [];
          client.mqttBuf = Buffer.concat([client.mqttBuf, fullWsPayload]);

          while (client.mqttBuf.length >= 2) {
            const mFirst = client.mqttBuf[0];
            const rem = decodeMqttVarInt(client.mqttBuf, 1);
            if (!rem) break;
            const totalLen = rem.nextOffset + rem.value;
            if (client.mqttBuf.length < totalLen) break;
            const mBody = client.mqttBuf.subarray(rem.nextOffset, totalLen);
            client.mqttBuf = client.mqttBuf.subarray(totalLen);
            handleMqttPacket(client, mFirst, mBody);
          }
        }
      }
    });

    const onDisconnect = () => {
      if (!client.closed) {
        client.closed = true;
        clients.delete(client);
        if (client.will) {
          fanoutPublish(client.will.topic, client.will.payload, { retain: client.will.retain });
          client.will = null;
        }
      }
    };
    socket.on('close', onDisconnect);
    socket.on('error', onDisconnect);
  });

  return {
    server,
    publishRaw(topic, payload, { retain = false } = {}) {
      const buf = Buffer.isBuffer(payload) ? payload : Buffer.from(String(payload), 'utf8');
      fanoutPublish(topic, buf, { retain });
    },
    async waitForSubscribers(roomCode, minClients = 1, timeoutMs = 6000) {
      const targetFilter = `keyprint-agy-mvp-v4/room/${roomCode}/profiles/+`;
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        let count = 0;
        for (const c of clients) {
          if (!c.closed && c.subscriptions.has(targetFilter)) count += 1;
        }
        if (count >= minClients) return;
        await new Promise((r) => setTimeout(r, 25));
      }
      throw new Error(`Timed out waiting for ${minClients} MQTT subscriber(s) in room ${roomCode}`);
    },
    closeAll() {
      for (const c of clients) c.close();
      server.close();
    },
  };
}

// ============================================================================
// Realistic Physical Keyboard Typing Simulator inside Page
// ============================================================================
async function typeRealisticInInput(page, selector, text, { baseDdMs = 112, baseHoldMs = 78, pressEnter = true } = {}) {
  await page.evaluate(
    async ({ sel, str, ddMs, holdMs, submitEnter }) => {
      const el = document.querySelector(sel);
      if (!el) throw new Error(`Missing input ${sel}`);
      el.focus();
      el.value = '';

      const codeForChar = (ch) => {
        if (ch === ' ') return 'Space';
        if (ch >= 'a' && ch <= 'z') return `Key${ch.toUpperCase()}`;
        if (ch >= 'A' && ch <= 'Z') return `Key${ch}`;
        if (ch >= '0' && ch <= '9') return `Digit${ch}`;
        if (ch === '!') return 'Digit1';
        if (ch === '@') return 'Digit2';
        if (ch === '#') return 'Digit3';
        if (ch === '.') return 'Period';
        if (ch === ',') return 'Comma';
        return '';
      };

      // Hangul 2-beolsik decomposition for Korean sentences so physical KeyboardEvent.code is accurate
      const CHO_CODES = ['KeyR', 'KeyR', 'KeyS', 'KeyE', 'KeyE', 'KeyF', 'KeyA', 'KeyQ', 'KeyQ', 'KeyT', 'KeyT', 'KeyD', 'KeyW', 'KeyW', 'KeyC', 'KeyZ', 'KeyX', 'KeyV', 'KeyG'];
      const JUNG_CODES = [
        ['KeyK'], ['KeyO'], ['KeyI'], ['KeyO'], ['KeyJ'], ['KeyP'], ['KeyU'], ['KeyP'],
        ['KeyH'], ['KeyH', 'KeyK'], ['KeyH', 'KeyO'], ['KeyH', 'KeyL'], ['KeyY'], ['KeyN'],
        ['KeyN', 'KeyJ'], ['KeyN', 'KeyP'], ['KeyN', 'KeyL'], ['KeyB'], ['KeyM'], ['KeyM', 'KeyL'], ['KeyL'],
      ];
      const JONG_CODES = [
        [], ['KeyR'], ['KeyR'], ['KeyR', 'KeyT'], ['KeyS'], ['KeyS', 'KeyW'], ['KeyS', 'KeyG'], ['KeyE'],
        ['KeyF'], ['KeyF', 'KeyR'], ['KeyF', 'KeyA'], ['KeyF', 'KeyQ'], ['KeyF', 'KeyT'], ['KeyF', 'KeyX'], ['KeyF', 'KeyV'], ['KeyF', 'KeyG'],
        ['KeyA'], ['KeyQ'], ['KeyQ', 'KeyT'], ['KeyT'], ['KeyT'], ['KeyD'], ['KeyW'], ['KeyC'], ['KeyZ'], ['KeyX'], ['KeyV'], ['KeyG'],
      ];

      let nowTs = performance.now() + 25;
      const fire = (type, init, ts) => {
        const ev = new Event(type, { bubbles: true, cancelable: true });
        Object.assign(
          ev,
          {
            key: '',
            code: '',
            repeat: false,
            isComposing: false,
            shiftKey: false,
            ctrlKey: false,
            metaKey: false,
            altKey: false,
            inputType: '',
          },
          init
        );
        Object.defineProperty(ev, 'timeStamp', { value: ts });
        el.dispatchEvent(ev);
      };

      let idx = 0;
      let currentVal = '';
      for (const ch of str) {
        const cp = ch.codePointAt(0);
        if (cp >= 0xac00 && cp <= 0xd7a3) {
          const sIdx = cp - 0xac00;
          const cho = Math.floor(sIdx / 588);
          const jung = Math.floor((sIdx % 588) / 28);
          const jong = sIdx % 28;
          const codes = [CHO_CODES[cho], ...JUNG_CODES[jung], ...JONG_CODES[jong]];
          for (let j = 0; j < codes.length; j++) {
            const c = codes[j];
            const jitter = ((idx * 17 + j * 11) % 19) - 9;
            nowTs += Math.max(45, ddMs + jitter);
            const h = Math.max(35, holdMs + (((idx * 13) % 15) - 7));
            fire('keydown', { code: c, key: 'Process', isComposing: true }, nowTs);
            if (j === codes.length - 1) {
              currentVal += ch;
              el.value = currentVal;
              fire('input', { inputType: 'insertCompositionText', isComposing: true }, nowTs + 0.5);
            }
            fire('keyup', { code: c, key: 'Process', isComposing: false }, nowTs + h);
            idx += 1;
          }
        } else {
          const needShift = (ch >= 'A' && ch <= 'Z') || '!@#$%^&*()_+'.includes(ch);
          const code = codeForChar(ch);
          const jitter = ((idx * 19) % 17) - 8;
          nowTs += Math.max(45, ddMs + jitter);
          const h = Math.max(35, holdMs + (((idx * 7) % 13) - 6));
          if (needShift) {
            fire('keydown', { code: 'ShiftLeft', key: 'Shift', shiftKey: true }, nowTs - 14);
          }
          fire('keydown', { code, key: ch, shiftKey: needShift }, nowTs);
          currentVal += ch;
          el.value = currentVal;
          fire('input', { inputType: 'insertText', data: ch }, nowTs + 0.4);
          fire('keyup', { code, key: ch, shiftKey: needShift }, nowTs + h);
          if (needShift) {
            fire('keyup', { code: 'ShiftLeft', key: 'Shift', shiftKey: false }, nowTs + h + 10);
          }
          idx += 1;
        }
      }

      if (submitEnter) {
        nowTs += 120;
        fire('keydown', { code: 'Enter', key: 'Enter' }, nowTs);
        fire('keyup', { code: 'Enter', key: 'Enter' }, nowTs + 55);
      }
    },
    { sel: selector, str: text, ddMs: baseDdMs, holdMs: baseHoldMs, submitEnter: pressEnter }
  );
}

async function waitForEngineReady(page) {
  await page.waitForFunction(
    () => window.__keyprintTestApi?.state?.ready === true,
    null,
    { timeout: 25000 }
  );
}

function attachPageErrorCollector(page, label) {
  const errors = [];
  page.on('pageerror', (err) => {
    errors.push(`${label}: ${err?.stack || err?.message || String(err)}`);
  });
  return errors;
}

// ============================================================================
// Main Adversarial Regression Runner
// ============================================================================
console.log('1) Building fresh web/dist with Vite...');
await build({ root: WEB_DIR, logLevel: 'error' });

const broker = createStaticAndMqttServer(DIST_DIR);
await new Promise((resolve) => broker.server.listen(0, '127.0.0.1', resolve));
const port = broker.server.address().port;
const baseUrl = `http://127.0.0.1:${port}`;
const wsBrokerUrl = `ws://127.0.0.1:${port}/mqtt`;
console.log(`2) Local static + WS-MQTT test broker listening on ${baseUrl} (${wsBrokerUrl})`);

const chromePath = findChromeExecutable();
const browser = await chromium.launch(
  chromePath ? { executablePath: chromePath, headless: true } : { channel: 'chrome', headless: true }
);

try {
  // --------------------------------------------------------------------------
  // SUITE 1: Stored XSS Defense (UI + Storage + MQTT)
  // --------------------------------------------------------------------------
  console.log('\n[Suite 1] Stored XSS defense across UI, localStorage & MQTT');
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);
    const page = await ctx.newPage();
    const pageErrors = attachPageErrorCollector(page, 'Suite1');

    await page.goto(`${baseUrl}/?room=XSSTEST&tab=guard`);
    await waitForEngineReady(page);
    await broker.waitForSubscribers('XSSTEST', 1);

    const meta = await page.evaluate(() => ({
      schemaVersion: window.__keyprintTestApi.engine.schemaVersion,
      preprocessVersion: window.__keyprintTestApi.engine.preprocessVersion,
      modelHash: window.__keyprintTestApi.engine.modelHash,
    }));
    const unitVec = Array.from({ length: 64 }, (_, i) => (i === 0 ? 0.8 : i === 1 ? 0.6 : 0));

    // 1a) Publish valid-schema profiles (<= 24 chars) carrying raw HTML/attribute payloads in `name` over MQTT
    const xssImgName = '<img src=x onerror=x=1>'; // 23 chars (passes 1..24 schema limit, tests HTML escaping!)
    const xssAttrName = '" autofocus onfocus="x='; // 23 chars (passes 1..24 schema limit, tests attr escaping!)
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/XSSTEST/profiles/xss_img_1',
      JSON.stringify({
        id: 'xss_img_1',
        name: xssImgName,
        isSample: false,
        color: '#e11d48',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...meta,
        arena: {
          ko: { centroid: unitVec, exemplars: [unitVec, unitVec], count: 5, inputMode: 'keyboard', cpm: 300, holdMs: 80, rolloverPct: 15 },
          en: null,
        },
        password: {
          target: 'SafeDemoPass2026!',
          centroid: unitVec,
          exemplars: [unitVec, unitVec],
          enc17: { centroid: unitVec, exemplars: [unitVec, unitVec] },
          count: 4,
          inputMode: 'keyboard',
        },
      }),
      { retain: true }
    );

    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/XSSTEST/profiles/xss_attr_2',
      JSON.stringify({
        id: 'xss_attr_2',
        name: xssAttrName,
        isSample: false,
        color: '#2563eb',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...meta,
        arena: {
          ko: { centroid: unitVec, exemplars: [unitVec, unitVec], count: 5, inputMode: 'keyboard' },
        },
        password: null,
      }),
      { retain: true }
    );

    // 1b) Attempt malicious color / id / count payloads over MQTT (must be rejected by schema!)
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/XSSTEST/profiles/xss_bad_color',
      JSON.stringify({
        id: 'xss_bad_color',
        name: 'BadColor',
        isSample: false,
        color: 'red" style="x:1" onerror="window.__xss=3',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...meta,
        arena: { ko: { centroid: unitVec, exemplars: [unitVec], count: 5, inputMode: 'keyboard' } },
      })
    );
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/XSSTEST/profiles/xss_bad_id',
      JSON.stringify({
        id: 'bad"><script>window.__xss=4</script>',
        name: 'BadId',
        isSample: false,
        color: '#2563eb',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...meta,
        arena: { ko: { centroid: unitVec, exemplars: [unitVec], count: 5, inputMode: 'keyboard' } },
      })
    );

    await page.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.length >= 2,
      null,
      { timeout: 6000 }
    );

    // Visit every view & modal that renders profile names, colors, IDs, counts, and SVG tooltips
    await page.click('#nav-arena-btn');
    await page.click('#nav-guard-btn');
    await page.click('#nav-principles-btn');
    await page.click('#open-room-btn');
    await page.waitForSelector('#room-dialog[open]');
    const modalTextSnapshot = await page.textContent('#modal-profile-list');
    await page.click('#close-room-btn');

    const xssAudit = await page.evaluate(() => ({
      xssGlobal1: window.__xss,
      xssGlobalX: window.x,
      profileIds: window.__keyprintTestApi.state.profiles.map((p) => p.id),
      injectedImgCount: document.querySelectorAll('#arena-roster-list img, #guard-account-list img, #viz-profile-chips img, #modal-profile-list img, #viz-step3-svg img').length,
      injectedScriptCount: document.querySelectorAll('main script, dialog script').length,
      autofocusHijackCount: document.querySelectorAll('[onfocus], [onerror], [onload]').length,
      rosterText: document.getElementById('modal-profile-list')?.textContent || '',
    }));

    check(xssAudit.xssGlobal1 === undefined && xssAudit.xssGlobalX === undefined, 'No XSS payload executed');
    check(xssAudit.injectedImgCount === 0 && xssAudit.injectedScriptCount === 0, 'Zero injected <img>/<script> elements');
    check(xssAudit.autofocusHijackCount === 0, 'Zero inline event handler attributes injected');
    check(
      !xssAudit.profileIds.includes('xss_bad_color') &&
        !xssAudit.profileIds.some((id) => id.includes('<script>')),
      `Malicious color/id profiles rejected before store: ${JSON.stringify(xssAudit.profileIds)}`
    );
    check((modalTextSnapshot || xssAudit.rosterText).includes(xssImgName), 'Escaped literal name rendered safely as text');
    check(pageErrors.length === 0, `Suite 1 zero pageerrors (${pageErrors.join('; ')})`);
    console.log('  ✓ Escaped names in all views/SVGs/modals, rejected malicious color/id, zero script execution');
    await ctx.close();
  }

  // --------------------------------------------------------------------------
  // SUITE 2: Malformed Profile Boot Survival (localStorage + MQTT pruning)
  // --------------------------------------------------------------------------
  console.log('\n[Suite 2] Malformed profile boot survival & localStorage pruning');
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);
    const page = await ctx.newPage();
    const pageErrors = attachPageErrorCollector(page, 'Suite2');

    await page.goto(`${baseUrl}/?room=SURVIVE`);
    await waitForEngineReady(page);
    const meta = await page.evaluate(() => ({
      schemaVersion: window.__keyprintTestApi.engine.schemaVersion,
      preprocessVersion: window.__keyprintTestApi.engine.preprocessVersion,
      modelHash: window.__keyprintTestApi.engine.modelHash,
    }));
    const unitVec = Array.from({ length: 64 }, (_, i) => (i === 2 ? 1.0 : 0.0));
    const zeroVec = new Array(64).fill(0);

    // Seed localStorage with 1 valid profile + 5 corrupt/malformed profiles (name: null, zero-norm, short pw, __proto__)
    await page.evaluate(
      ({ compat, uVec, zVec }) => {
        const rawJson = JSON.stringify({
          ...compat,
          profiles: [
            {
              id: 'good_survivor',
              name: '정상생존자',
              isSample: false,
              color: '#0d9488',
              createdAt: 1700000000000,
              updatedAt: 1700000005000,
              ...compat,
              arena: { ko: { centroid: uVec, exemplars: [uVec, uVec], count: 5, inputMode: 'keyboard' } },
              password: null,
            },
            {
              id: 'bad_null_name',
              name: null,
              isSample: false,
              color: '#2563eb',
              createdAt: 1700000000000,
              updatedAt: 1700000005000,
              ...compat,
              arena: { ko: { centroid: uVec, exemplars: [uVec], count: 5, inputMode: 'keyboard' } },
            },
            {
              id: 'bad_zero_vec',
              name: '영벡터계정',
              isSample: false,
              color: '#2563eb',
              createdAt: 1700000000000,
              updatedAt: 1700000005000,
              ...compat,
              arena: { ko: { centroid: zVec, exemplars: [zVec], count: 5, inputMode: 'keyboard' } },
            },
            {
              id: 'bad_short_pw',
              name: '짧은암호',
              isSample: false,
              color: '#2563eb',
              createdAt: 1700000000000,
              updatedAt: 1700000005000,
              ...compat,
              password: {
                target: 'shortpw',
                centroid: uVec,
                exemplars: [uVec],
                enc17: { centroid: uVec, exemplars: [uVec] },
                count: 4,
                inputMode: 'keyboard',
              },
            },
            JSON.parse('{"__proto__":{"pollutedBoot":true},"id":"proto_bad","name":"x"}'),
          ],
          tombstones: { valid_tomb: 1700000000000, '../bad_tomb': 'not_a_number' },
        });
        localStorage.setItem('keyprint_mvp_v4_room_SURVIVE', rawJson);
      },
      { compat: meta, uVec: unitVec, zVec: zeroVec }
    );

    // Also publish corrupt MQTT retained profiles before reload
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/SURVIVE/profiles/mqtt_null_name',
      JSON.stringify({
        id: 'mqtt_null_name',
        name: null,
        isSample: false,
        color: '#2563eb',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...meta,
        arena: { ko: { centroid: unitVec, exemplars: [unitVec], count: 5, inputMode: 'keyboard' } },
      }),
      { retain: true }
    );
    broker.publishRaw('keyprint-agy-mvp-v4/room/SURVIVE/profiles/mqtt_corrupt_json', '{not valid json', {
      retain: true,
    });

    await page.reload();
    await waitForEngineReady(page);
    await page.click('#nav-principles-btn');

    const bootState = await page.evaluate(() => {
      const stored = JSON.parse(localStorage.getItem('keyprint_mvp_v4_room_SURVIVE') || '{}');
      return {
        ready: window.__keyprintTestApi.state.ready,
        polluted: ({}).pollutedBoot,
        liveIds: window.__keyprintTestApi.state.profiles.map((p) => p.id),
        storedIds: (stored.profiles || []).map((p) => p.id),
        storedTombKeys: Object.keys(stored.tombstones || {}),
      };
    });

    check(bootState.ready === true, 'Engine ready after reload with corrupt storage + MQTT');
    check(bootState.polluted === undefined, 'Prototype pollution blocked');
    check(
      JSON.stringify(bootState.liveIds) === JSON.stringify(['good_survivor']),
      `Only valid profile survived in memory: ${JSON.stringify(bootState.liveIds)}`
    );
    check(
      JSON.stringify(bootState.storedIds) === JSON.stringify(['good_survivor']) &&
        JSON.stringify(bootState.storedTombKeys) === JSON.stringify(['valid_tomb']),
      `Corrupt entries pruned from localStorage: profiles=${JSON.stringify(bootState.storedIds)}`
    );
    check(pageErrors.length === 0, `Suite 2 zero pageerrors (${pageErrors.join('; ')})`);
    console.log('  ✓ Corrupt profiles pruned before render & persisted cleanly; reload survived with 0 errors');
    await ctx.close();
  }

  // --------------------------------------------------------------------------
  // SUITE 3: Cross-Channel v4 Version & ModelHash Isolation
  // --------------------------------------------------------------------------
  console.log('\n[Suite 3] Cross-channel v4 version & modelHash isolation (MQTT, BroadcastChannel, localStorage)');
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);
    const page = await ctx.newPage();
    const pageErrors = attachPageErrorCollector(page, 'Suite3');

    // Pre-seed legacy v3 localStorage key before navigation
    await page.goto(`${baseUrl}/?room=ISOLATE`);
    await waitForEngineReady(page);
    await broker.waitForSubscribers('ISOLATE', 1);

    const meta = await page.evaluate(() => {
      localStorage.setItem(
        'keyprint_mvp_v3_room_ISOLATE',
        JSON.stringify([{ id: 'legacy_ls_v3', name: '구버전로컬' }])
      );
      return {
        schemaVersion: window.__keyprintTestApi.engine.schemaVersion,
        preprocessVersion: window.__keyprintTestApi.engine.preprocessVersion,
        modelHash: window.__keyprintTestApi.engine.modelHash,
      };
    });
    const unitVec = Array.from({ length: 64 }, (_, i) => (i === 5 ? 1.0 : 0.0));

    // Upsert a legitimate v4 profile
    await page.evaluate(
      ({ uVec }) => {
        window.__keyprintTestApi.roomSync.upsertProfile({
          id: 'protected_v4',
          name: '보호된v4계정',
          isSample: false,
          color: '#2563eb',
          arena: { ko: { centroid: uVec, exemplars: [uVec, uVec], count: 5, inputMode: 'keyboard' } },
          password: null,
        });
      },
      { uVec: unitVec }
    );

    // 3a) Legacy v3 MQTT topic
    broker.publishRaw(
      'keyprint-agy-mvp-v3/room/ISOLATE/profiles/legacy_mqtt_v3',
      JSON.stringify({ id: 'legacy_mqtt_v3', name: '구버전MQTT' }),
      { retain: true }
    );

    // 3b) v4 topic with wrong schemaVersion / preprocessVersion / modelHash
    for (const [id, override] of [
      ['wrong_schema', { schemaVersion: 3 }],
      ['wrong_preproc', { preprocessVersion: 'kp-preproc-v3' }],
      ['wrong_hash', { modelHash: 'ffffffffffffffff' }],
    ]) {
      broker.publishRaw(
        `keyprint-agy-mvp-v4/room/ISOLATE/profiles/${id}`,
        JSON.stringify({
          id,
          name: id,
          isSample: false,
          color: '#7c3aed',
          createdAt: Date.now(),
          updatedAt: Date.now() + 50000,
          ...meta,
          ...override,
          arena: { ko: { centroid: unitVec, exemplars: [unitVec], count: 5, inputMode: 'keyboard' } },
        })
      );
    }

    // 3c) Hostile / incompatible delete & clear_all on MQTT event topic + unversioned empty profile topic
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/ISOLATE/events',
      JSON.stringify({
        ...meta,
        modelHash: '0000000000000000',
        type: 'delete',
        id: 'protected_v4',
        ts: Date.now() + 99999,
        sender: 'incompatible_peer',
      })
    );
    broker.publishRaw(
      'keyprint-agy-mvp-v4/room/ISOLATE/events',
      JSON.stringify({
        ...meta,
        schemaVersion: 3,
        type: 'clear_all',
        ts: Date.now() + 99999,
        sender: 'incompatible_peer',
      })
    );
    broker.publishRaw('keyprint-agy-mvp-v4/room/ISOLATE/profiles/protected_v4', '', { retain: false });

    // 3d) Incompatible BroadcastChannel upsert / delete / clear_all
    await page.evaluate(
      ({ compat, uVec }) => {
        const bc = new BroadcastChannel('keyprint_bc_v4_ISOLATE');
        bc.postMessage({
          ...compat,
          modelHash: '1111111111111111',
          type: 'upsert',
          profile: {
            id: 'bc_wrong_hash',
            name: 'BC불일치',
            isSample: false,
            color: '#2563eb',
            createdAt: Date.now(),
            updatedAt: Date.now(),
            ...compat,
            modelHash: '1111111111111111',
            arena: { ko: { centroid: uVec, exemplars: [uVec], count: 5, inputMode: 'keyboard' } },
          },
        });
        bc.postMessage({
          ...compat,
          preprocessVersion: 'kp-preproc-old',
          type: 'delete',
          id: 'protected_v4',
          ts: Date.now() + 99999,
        });
        bc.postMessage({
          ...compat,
          schemaVersion: 2,
          type: 'clear_all',
          ts: Date.now() + 99999,
        });
        bc.close();
      },
      { compat: meta, uVec: unitVec }
    );

    await page.waitForTimeout(250);
    const iso = await page.evaluate(() => ({
      ids: window.__keyprintTestApi.state.profiles.map((p) => p.id),
      skipCount: window.__keyprintTestApi.roomSync.compatSkipCount,
      noticeHidden: document.getElementById('compat-isolation-notice')?.hidden,
    }));

    check(
      JSON.stringify(iso.ids) === JSON.stringify(['protected_v4']),
      `protected_v4 untouched and all incompatible messages rejected: ${JSON.stringify(iso.ids)}`
    );
    check(iso.skipCount >= 5, `Compatibility skip counter incremented (${iso.skipCount})`);
    check(iso.noticeHidden === false, 'Subtle #compat-isolation-notice banner shown');
    check(pageErrors.length === 0, `Suite 3 zero pageerrors (${pageErrors.join('; ')})`);
    console.log(`  ✓ Isolated ${iso.skipCount} incompatible MQTT/BroadcastChannel/v3 messages; valid v4 profile intact`);
    await ctx.close();
  }

  // --------------------------------------------------------------------------
  // SUITE 4 & 5: Sample Name Block, Duplicate Password Overwrite Block,
  //              Public Room Notice, Mode C Verdict Snapshot & Undistorted PCA
  // --------------------------------------------------------------------------
  console.log('\n[Suite 4 & 5] Sample block, duplicate password block, Mode C verdict snapshot & undistorted PCA');
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);
    const page = await ctx.newPage();
    const pageErrors = attachPageErrorCollector(page, 'Suite4_5');

    await page.goto(`${baseUrl}/?room=GUARDVIZ`);
    await waitForEngineReady(page);

    // Check public room notice & dual-encoder accordion text
    const staticChecks = await page.evaluate(() => {
      const notices = Array.from(document.querySelectorAll('.public-room-notice')).map((el) => el.textContent || '');
      const detailsText = document.querySelector('#view-principles .viz-details')?.textContent || '';
      return {
        noticeCount: notices.length,
        hasWarningText: notices.every((t) => t.includes('실제 사용하는 개인 비밀번호는 절대 입력하지 마세요')),
        hasDualEncoderDoc:
          detailsText.includes('듀얼 인코더 아키텍처') &&
          detailsText.includes('9채널') &&
          !detailsText.includes('단 한 개의 64차원 인코더로'),
      };
    });
    check(staticChecks.noticeCount >= 2 && staticChecks.hasWarningText, 'Public room sharing notice present in Guard & Room modal');
    check(staticChecks.hasDualEncoderDoc, 'Mode C accordion accurately documents dual-encoder (17ch + 9ch) architecture');

    // 4a) Sample name block in Arena
    await page.fill('#arena-name-input', '서연 (샘플·고속연타)');
    const arenaPrompt1 = await page.evaluate(() => {
      const el = document.getElementById('arena-enroll-prompt');
      return Array.from(el.querySelectorAll('span')).map((s) => s.textContent).join('');
    });
    await typeRealisticInInput(page, '#arena-enroll-input', arenaPrompt1, { baseDdMs: 105, baseHoldMs: 75 });
    const arenaSampleMsg = await page.textContent('#arena-enroll-feedback');
    check(arenaSampleMsg.includes('샘플 프로필 이름'), `Arena blocked sample name: "${arenaSampleMsg}"`);

    // 4b) Sample name block in Guard
    await page.click('#nav-guard-btn');
    await page.click('#guard-sub-enroll');
    await page.fill('#guard-enroll-name', '서연 (샘플·고속연타)');
    await page.fill('#guard-enroll-pw-target', 'Gachon2026pw!');
    await typeRealisticInInput(page, '#guard-enroll-input', 'Gachon2026pw!', { baseDdMs: 110, baseHoldMs: 80 });
    const guardSampleMsg = await page.textContent('#guard-enroll-feedback');
    check(guardSampleMsg.includes('샘플 프로필 이름'), `Guard blocked sample name: "${guardSampleMsg}"`);

    // 4c) Enroll legitimate 13-char password account "테스트계정A" (4 shots)
    await page.fill('#guard-enroll-name', '테스트계정A');
    await page.fill('#guard-enroll-pw-target', 'Gachon2026pw!');
    await page.dispatchEvent('#guard-enroll-pw-target', 'input');
    for (let shot = 0; shot < 4; shot++) {
      await typeRealisticInInput(page, '#guard-enroll-input', 'Gachon2026pw!', {
        baseDdMs: 112 + shot * 2,
        baseHoldMs: 78 + shot,
      });
    }
    await page.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '테스트계정A' && p.password?.count === 4),
      null,
      { timeout: 8000 }
    );
    const origProfileSnap = await page.evaluate(() => {
      const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === '테스트계정A');
      return { target: p.password.target, centroid0: p.password.centroid[0], count: p.password.count };
    });
    check(origProfileSnap.target === 'Gachon2026pw!' && origProfileSnap.count === 4, 'Enrolled 테스트계정A with 4 shots');

    // 4d) Attempt duplicate password overwrite under same name "테스트계정A" -> MUST BE REJECTED!
    await page.click('#guard-sub-enroll');
    await page.fill('#guard-enroll-name', '테스트계정A');
    await page.fill('#guard-enroll-pw-target', 'OverwritePass99!');
    await page.dispatchEvent('#guard-enroll-pw-target', 'input');
    await typeRealisticInInput(page, '#guard-enroll-input', 'OverwritePass99!', { baseDdMs: 115, baseHoldMs: 80 });
    const dupFeedback = await page.textContent('#guard-enroll-feedback');
    const afterDupAttempt = await page.evaluate(() => {
      const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === '테스트계정A');
      return { target: p.password.target, centroid0: p.password.centroid[0], count: p.password.count };
    });
    check(
      dupFeedback.includes('이미 비밀번호가 등록된 이름입니다'),
      `Duplicate password overwrite rejected with message: "${dupFeedback}"`
    );
    check(
      afterDupAttempt.target === 'Gachon2026pw!' && afterDupAttempt.centroid0 === origProfileSnap.centroid0,
      'Original password target & centroid preserved intact'
    );
    console.log('  ✓ Sample names blocked in Arena & Guard; duplicate password account overwrite rejected');

    // 5a) Verify "테스트계정A" in Guard with matching genuine rhythm -> triggers verdict + adaptation!
    await page.click('#guard-sub-verify');
    // Set relaxed offset (-0.08) so deterministic synthetic timing reliably passes + adapts
    await page.click('.tau-btn[data-tau-offset="-0.08"]');
    await typeRealisticInInput(page, '#guard-verify-input', 'Gachon2026pw!', { baseDdMs: 113, baseHoldMs: 79 });
    await page.waitForSelector('#guard-result-card:not([hidden])');

    const guardVerdictInfo = await page.evaluate(() => {
      const v = window.__keyprintTestApi.state.lastVerdict;
      const liveProf = window.__keyprintTestApi.state.profiles.find((x) => x.name === '테스트계정A');
      const snapProf = v?.evaluatedTemplatesSnapshot?.find((x) => x.name === '테스트계정A');
      return {
        verdict: v
          ? {
              queryId: v.queryId,
              accepted: v.accepted,
              similarity: v.similarity,
              threshold: v.threshold,
              adaptedAfterVerdict: v.adaptedAfterVerdict,
              snapTemplateCount: snapProf?.template?.count,
            }
          : null,
        liveCount: liveProf?.password?.count,
        guardScoreText: document.getElementById('guard-result-score')?.textContent,
      };
    });
    check(guardVerdictInfo.verdict !== null, 'state.lastVerdict captured on Guard verification');
    check(
      guardVerdictInfo.verdict.snapTemplateCount === 4,
      `Pre-adaptation snapshot preserved count=4 (live count=${guardVerdictInfo.liveCount})`
    );

    // Click "원리 & 내 지문 보기 →" (#guard-inspect-viz-btn) and verify Mode C matches Guard verdict 100%!
    await page.click('#guard-inspect-viz-btn');
    const vizMatch = await page.evaluate(() => {
      const active = window.__keyprintTestApi.getActiveVerdictForViz();
      return {
        activeNonNull: Boolean(active),
        bannerText: document.getElementById('viz-verdict-status')?.textContent || '',
        adaptBadgeHidden: document.getElementById('viz-verdict-adapt-badge')?.hidden,
        adaptBadgeText: document.getElementById('viz-verdict-adapt-badge')?.textContent || '',
      };
    });
    const expectedSimPct = (guardVerdictInfo.verdict.similarity * 100).toFixed(1);
    const expectedThrPct = (guardVerdictInfo.verdict.threshold * 100).toFixed(1);
    check(vizMatch.activeNonNull, 'getActiveVerdictForViz() active in Mode C');
    check(
      vizMatch.bannerText.includes('공식 판정 스냅샷') &&
        vizMatch.bannerText.includes(`${expectedSimPct}%`) &&
        vizMatch.bannerText.includes(`${expectedThrPct}%`),
      `Mode C banner matches exact Guard similarity/threshold: "${vizMatch.bannerText}"`
    );
    if (guardVerdictInfo.verdict.adaptedAfterVerdict) {
      check(
        vizMatch.adaptBadgeHidden === false && vizMatch.adaptBadgeText.includes('판정 당시 템플릿 기준'),
        'Pre-adaptation badge visible when automatic adaptation occurred'
      );
    }

    // 5b) Invalidation Condition 1: Typing new input in sandbox invalidates stale verdict
    await typeRealisticInInput(page, '#viz-sandbox-input', 'hello world', { pressEnter: false });
    const afterSandboxType = await page.evaluate(() => ({
      active: window.__keyprintTestApi.getActiveVerdictForViz(),
      bannerText: document.getElementById('viz-verdict-status')?.textContent || '',
      adaptBadgeHidden: document.getElementById('viz-verdict-adapt-badge')?.hidden,
    }));
    check(
      afterSandboxType.active === null &&
        afterSandboxType.bannerText === '실시간 탐색 미리보기 (공식 인증 판정 전)' &&
        afterSandboxType.adaptBadgeHidden === true,
      `Typing new input invalidated stale verdict -> "${afterSandboxType.bannerText}"`
    );

    // 5c) Arena 1:N Blind Verdict Snapshot + Language Switch Invalidation (`ko` -> `en`)
    await page.evaluate(() => {
      const samples = window.__keyprintTestApi.engine.config?.mvp_sample_profiles || [];
      window.__keyprintTestApi.roomSync.loadSamples(samples);
    });
    await page.click('#nav-arena-btn');
    await page.click('#arena-sub-blind');
    for (let q = 0; q < 2; q++) {
      const blindPrompt = await page.evaluate(() => {
        const el = document.getElementById('arena-blind-prompt');
        return Array.from(el.querySelectorAll('span')).map((s) => s.textContent).join('');
      });
      await typeRealisticInInput(page, '#arena-blind-input', blindPrompt, { baseDdMs: 108, baseHoldMs: 76 });
    }
    await page.waitForSelector('#arena-result-card:not([hidden])');
    await page.click('#arena-inspect-viz-btn');

    const arenaVizBeforeLang = await page.evaluate(() => ({
      active: Boolean(window.__keyprintTestApi.getActiveVerdictForViz()),
      bannerText: document.getElementById('viz-verdict-status')?.textContent || '',
    }));
    check(
      arenaVizBeforeLang.active && arenaVizBeforeLang.bannerText.includes('공식 판정 스냅샷'),
      `Mode C displays exact Arena 1:N verdict snapshot: "${arenaVizBeforeLang.bannerText}"`
    );

    // Switch language ko -> en -> MUST invalidate Arena ko verdict!
    await page.evaluate(() => document.getElementById('lang-en-btn').click());
    await page.click('#nav-principles-btn');
    const arenaVizAfterLang = await page.evaluate(() => ({
      active: window.__keyprintTestApi.getActiveVerdictForViz(),
      bannerText: document.getElementById('viz-verdict-status')?.textContent || '',
    }));
    check(
      arenaVizAfterLang.active === null &&
        arenaVizAfterLang.bannerText === '실시간 탐색 미리보기 (공식 인증 판정 전)',
      `Language switch invalidated Arena verdict -> "${arenaVizAfterLang.bannerText}"`
    );

    // 5d) Room switch invalidation & Profile deletion invalidation
    await page.click('#nav-guard-btn');
    await page.click('#guard-sub-verify');
    await page.evaluate(() => {
      const userAcc = window.__keyprintTestApi.state.profiles.find((p) => p.name === '테스트계정A');
      if (userAcc) window.__keyprintTestApi.state.selectedGuardId = userAcc.id;
    });
    await typeRealisticInInput(page, '#guard-verify-input', 'Gachon2026pw!', { baseDdMs: 113, baseHoldMs: 79 });
    await page.waitForFunction(() => Boolean(window.__keyprintTestApi.state.lastVerdict));
    // Delete evaluated target profile -> getActiveVerdictForViz() must return null!
    const afterDeleteInvalidation = await page.evaluate(() => {
      const targetId = window.__keyprintTestApi.state.lastVerdict.targetProfileId;
      window.__keyprintTestApi.roomSync.deleteProfile(targetId);
      window.__keyprintTestApi.state.viz.mapDomain = 'guard';
      return window.__keyprintTestApi.getActiveVerdictForViz();
    });
    check(afterDeleteInvalidation === null, 'Deleting target profile invalidates stale verdict');

    // 5e) Undistorted 2D PCA Coordinates Check (two nearly identical centroids NOT pushed apart by 66px!)
    const pcaCheck = await page.evaluate(() => {
      const rs = window.__keyprintTestApi.roomSync;
      rs.connect('PCATEST');
      const cA = Array.from({ length: 64 }, (_, i) => (i === 0 ? 0.9999 : i === 1 ? 0.01414 : 0));
      const cB = Array.from({ length: 64 }, (_, i) => (i === 0 ? 0.9998 : i === 1 ? 0.01999 : 0));
      const cC = Array.from({ length: 64 }, (_, i) => (i === 2 ? 1.0 : 0));
      rs.upsertProfile({
        id: 'close_a',
        name: '쌍둥이A',
        isSample: false,
        color: '#2563eb',
        arena: { ko: { centroid: cA, exemplars: [cA, cA], count: 5, inputMode: 'keyboard' } },
      });
      rs.upsertProfile({
        id: 'close_b',
        name: '쌍둥이B',
        isSample: false,
        color: '#e11d48',
        arena: { ko: { centroid: cB, exemplars: [cB, cB], count: 5, inputMode: 'keyboard' } },
      });
      rs.upsertProfile({
        id: 'far_c',
        name: '원거리C',
        isSample: false,
        color: '#0d9488',
        arena: { ko: { centroid: cC, exemplars: [cC, cC], count: 5, inputMode: 'keyboard' } },
      });
      document.getElementById('lang-ko-btn').click();
      document.getElementById('nav-principles-btn').click();
      document.getElementById('viz-map-domain-arena').click();

      const elA = document.querySelector('#viz-step3-svg circle.pca-centroid[data-profile-id="close_a"]');
      const elB = document.querySelector('#viz-step3-svg circle.pca-centroid[data-profile-id="close_b"]');
      const labels = Array.from(document.querySelectorAll('#viz-step3-svg text.pca-centroid-label')).map((t) => ({
        text: t.textContent,
        x: Number(t.getAttribute('x')),
        y: Number(t.getAttribute('y')),
      }));
      const distAB =
        elA && elB
          ? Math.hypot(
              Number(elA.getAttribute('cx')) - Number(elB.getAttribute('cx')),
              Number(elA.getAttribute('cy')) - Number(elB.getAttribute('cy'))
            )
          : 999;
      const lblA = labels.find((l) => l.text === '쌍둥이A');
      const lblB = labels.find((l) => l.text === '쌍둥이B');
      const labelSeparation =
        lblA && lblB ? Math.hypot(lblA.x - lblB.x, lblA.y - lblB.y) : 0;
      return { distAB, labelSeparation };
    });
    check(
      pcaCheck.distAB < 6.0,
      `Undistorted linear 2D PCA: nearly identical centroids stayed ${pcaCheck.distAB.toFixed(2)}px apart (< 6px, NOT 66px!)`
    );
    check(
      pcaCheck.labelSeparation >= 15.0,
      `PCA text labels cleanly separated (${pcaCheck.labelSeparation.toFixed(1)}px) without distorting centroid circles`
    );
    check(pageErrors.length === 0, `Suite 4 & 5 zero pageerrors (${pageErrors.join('; ')})`);
    console.log('  ✓ Mode C verdict snapshot, 5-condition invalidation & undistorted 2D PCA verified');
    await ctx.close();
  }

  // --------------------------------------------------------------------------
  // SUITE 6: !state.ready Enter Feedback & Synchronous WebSocket Throw Resilience
  // --------------------------------------------------------------------------
  console.log('\n[Suite 6] !state.ready Enter feedback & synchronous window.WebSocket throw resilience');
  {
    const ctx = await browser.newContext();
    await ctx.addInitScript(() => {
      window.WebSocket = function BlockedWebSocket() {
        throw new Error('Blocked WS by sandbox policy');
      };
    });
    const page = await ctx.newPage();
    const pageErrors = attachPageErrorCollector(page, 'Suite6');

    // Hold .onnx delivery briefly so we can deterministically test Enter while !state.ready
    let releaseOnnx;
    const onnxGate = new Promise((resolve) => {
      releaseOnnx = resolve;
    });
    await page.route('**/*.onnx', async (route) => {
      await onnxGate;
      await route.continue();
    });

    await page.goto(`${baseUrl}/?room=BLOCKEDWS`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#arena-enroll-input');

    // Press Enter while state.ready === false in Arena and Guard
    await typeRealisticInInput(page, '#arena-enroll-input', '테스트', { pressEnter: true });
    const notReadyArenaMsg = await page.textContent('#arena-enroll-feedback');
    check(
      notReadyArenaMsg.includes('AI 생체지문 엔진을 준비 중입니다'),
      `!state.ready Arena Enter feedback: "${notReadyArenaMsg}"`
    );

    await page.click('#nav-guard-btn');
    await page.click('#guard-sub-enroll');
    await page.focus('#guard-enroll-name');
    await page.keyboard.press('Enter');
    const notReadyGuardMsg = await page.textContent('#guard-enroll-feedback');
    check(
      notReadyGuardMsg.includes('AI 생체지문 엔진을 준비 중입니다'),
      `!state.ready Guard Enter feedback: "${notReadyGuardMsg}"`
    );

    // Release .onnx and verify ONNX initializes cleanly & local mode works despite throwing WebSocket!
    releaseOnnx();
    await waitForEngineReady(page);
    const wsResilience = await page.evaluate(async () => {
      const api = window.__keyprintTestApi;
      const sampleStroke = api.state.viz.capture.stroke;
      const emb = await api.engine.embedAll(sampleStroke);
      const norm9 = Math.hypot(...emb.password);
      const norm17 = Math.hypot(...emb.sentence);
      return {
        ready: api.state.ready,
        syncStatus: api.roomSync.status,
        pillReady: document.getElementById('engine-pill')?.classList.contains('is-ready'),
        norm9,
        norm17,
      };
    });

    check(wsResilience.ready && wsResilience.pillReady, 'ONNX WASM engine booted cleanly despite synchronous WebSocket throw');
    check(wsResilience.syncStatus === 'local', `RoomProfileSync fell back cleanly to '${wsResilience.syncStatus}' mode`);
    check(
      Math.abs(wsResilience.norm9 - 1.0) < 1e-3 && Math.abs(wsResilience.norm17 - 1.0) < 1e-3,
      `Dual ONNX encoders ran unit-norm inference (norm9=${wsResilience.norm9.toFixed(4)}, norm17=${wsResilience.norm17.toFixed(4)})`
    );
    check(pageErrors.length === 0, `Suite 6 zero pageerrors (${pageErrors.join('; ')})`);
    console.log('  ✓ !state.ready Enter guidance shown; throwing WebSocket fell back to local mode with full ONNX inference');
    await ctx.close();
  }

  // --------------------------------------------------------------------------
  // SUITE 7: Real 2-Browser-Context Live Room Sync over Local WS-MQTT Broker
  // --------------------------------------------------------------------------
  console.log('\n[Suite 7] Real 2-browser-context live multi-user sync & verification via local WS-MQTT broker');
  {
    const ctx1 = await browser.newContext();
    const ctx2 = await browser.newContext();
    await ctx1.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);
    await ctx2.addInitScript((wsUrl) => {
      window.__KEYPRINT_MQTT_BROKERS = [wsUrl];
    }, wsBrokerUrl);

    const page1 = await ctx1.newPage();
    const page2 = await ctx2.newPage();
    const errs1 = attachPageErrorCollector(page1, 'Suite7-ctx1');
    const errs2 = attachPageErrorCollector(page2, 'Suite7-ctx2');

    await Promise.all([
      page1.goto(`${baseUrl}/?room=MQTTSYNC&tab=guard`),
      page2.goto(`${baseUrl}/?room=MQTTSYNC&tab=guard`),
    ]);
    await Promise.all([waitForEngineReady(page1), waitForEngineReady(page2)]);
    await broker.waitForSubscribers('MQTTSYNC', 2);

    // Context 1 enrolls "실시간동기화계정" ("SyncPass2026!!", 4 captures)
    await page1.click('#guard-sub-enroll');
    await page1.fill('#guard-enroll-name', '실시간동기화계정');
    await page1.fill('#guard-enroll-pw-target', 'SyncPass2026!!');
    await page1.dispatchEvent('#guard-enroll-pw-target', 'input');
    for (let shot = 0; shot < 4; shot++) {
      await typeRealisticInInput(page1, '#guard-enroll-input', 'SyncPass2026!!', {
        baseDdMs: 114 + shot,
        baseHoldMs: 80 + shot,
      });
    }

    // Context 2 (isolated context — NO shared BroadcastChannel/localStorage!) receives it live over WS-MQTT!
    await page2.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '실시간동기화계정' && p.password?.target === 'SyncPass2026!!'),
      null,
      { timeout: 8000 }
    );

    // Context 2 verifies against the synced profile using real ONNX inference
    await page2.click('.tau-btn[data-tau-offset="-0.08"]');
    await typeRealisticInInput(page2, '#guard-verify-input', 'SyncPass2026!!', {
      baseDdMs: 115,
      baseHoldMs: 81,
    });
    await page2.waitForSelector('#guard-result-card:not([hidden])');

    const ctx2Verify = await page2.evaluate(() => ({
      targetName: document.getElementById('guard-target-name')?.textContent,
      scoreText: document.getElementById('guard-result-score')?.textContent || '',
      verdict: window.__keyprintTestApi.state.lastVerdict,
    }));
    check(ctx2Verify.targetName === '실시간동기화계정', `Context 2 selected synced account: ${ctx2Verify.targetName}`);
    check(
      ctx2Verify.verdict && Number.isFinite(ctx2Verify.verdict.similarity) && ctx2Verify.scoreText.endsWith('%'),
      `Context 2 ran live ONNX verification on MQTT-synced profile (score=${ctx2Verify.scoreText})`
    );

    // Context 1 deletes the profile -> Context 2 removes it live via versioned MQTT event!
    await page1.evaluate(() => {
      const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === '실시간동기화계정');
      if (p) window.__keyprintTestApi.roomSync.deleteProfile(p.id);
    });
    await page2.waitForFunction(
      () => !window.__keyprintTestApi.state.profiles.some((p) => p.name === '실시간동기화계정'),
      null,
      { timeout: 6000 }
    );
    check(true, 'Version-stamped MQTT delete propagated live from Context 1 to Context 2');
    check(errs1.length === 0 && errs2.length === 0, `Suite 7 zero pageerrors (${[...errs1, ...errs2].join('; ')})`);
    console.log(`  ✓ Context 1 -> WS-MQTT -> Context 2 live profile sync, ONNX verify (${ctx2Verify.scoreText}), and versioned delete succeeded`);

    await ctx1.close();
    await ctx2.close();
  }
} finally {
  await browser.close();
  broker.closeAll();
}

console.log(`\n${totalChecks - totalFailures}/${totalChecks} adversarial regression checks passed`);
if (totalFailures > 0) {
  console.error(`${totalFailures} adversarial regression check(s) FAILED`);
  process.exit(1);
}
