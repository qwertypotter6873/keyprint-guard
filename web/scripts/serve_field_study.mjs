/**
 * Zero-dependency Local Multi-Device Field-Study Hub for KeyPrint Guard.
 *
 * Solves cross-device (Laptop Mechanical / External Pantograph / Laptop Pantograph / Smartphone Touch)
 * live evaluation on local Wi-Fi without external cloud servers:
 *   1. Serves `web/dist` on `0.0.0.0` over BOTH:
 *        - HTTPS (`:8443` default, `mkcert` or `openssl` SAN cert) -> `[SecureContext]` required by
 *          mobile browsers for `DeviceMotionEvent` accelerometer/gyroscope capture + touch typing.
 *        - HTTP  (`:8780` default) -> zero-TLS-warning instant smartphone touch typing.
 *   2. Embeds a private in-memory WebSocket MQTT 3.1.1/5.0 broker (`/mqtt` shared across HTTP `ws://`
 *      and HTTPS `wss://`) so profiles enrolled on the laptop immediately appear on the smartphone
 *      (and vice versa) in room `#LOCALSTUDY` without sending data to any external public broker.
 *   3. Provides `/api/field-study/{info,records,ca.pem}` so every enrollment and verification record
 *      from every connected device is automatically deduplicated and appended directly to
 *      `logs/field_study/field_study_merged.jsonl` on the laptop, ready for `src/analyze_field_study.py`.
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { createServer as createHttpsServer } from 'node:https';
import os from 'node:os';
import { dirname, extname, join, normalize, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(WEB_ROOT, '..');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
};

// ============================================================================
// 1. Minimal ISO/IEC 18004 QR Code Generator (Byte Mode, ECC-L, Versions 1..6)
// ============================================================================
const GF_EXP = new Uint8Array(512);
const GF_LOG = new Uint8Array(256);
{
  let v = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = v;
    GF_LOG[v] = i;
    v <<= 1;
    if (v & 0x100) v ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];
}

function gfMul(a, b) {
  return a === 0 || b === 0 ? 0 : GF_EXP[GF_LOG[a] + GF_LOG[b]];
}

function rsGeneratorPoly(degree) {
  let poly = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(poly.length + 1).fill(0);
    const root = GF_EXP[i];
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= poly[j];
      next[j + 1] ^= gfMul(poly[j], root);
    }
    poly = next;
  }
  return poly;
}

function rsRemainder(dataBytes, ecCount) {
  const gen = rsGeneratorPoly(ecCount);
  const rem = new Uint8Array(ecCount);
  for (const byte of dataBytes) {
    const factor = byte ^ rem[0];
    rem.copyWithin(0, 1);
    rem[ecCount - 1] = 0;
    if (factor !== 0) {
      for (let j = 0; j < ecCount; j++) {
        rem[j] ^= gfMul(gen[j + 1], factor);
      }
    }
  }
  return rem;
}

// ISO/IEC 18004 ECC-L single-block parameters for Versions 1..5 (up to 106 UTF-8 bytes)
// plus Version 6 (2 interleaved blocks, up to 134 UTF-8 bytes).
const QR_ECC_L_VERSIONS = [
  null,
  { version: 1, size: 21, dataCodewords: 19, ecPerBlock: 7, blocks: 1, align: [] },
  { version: 2, size: 25, dataCodewords: 34, ecPerBlock: 10, blocks: 1, align: [6, 18] },
  { version: 3, size: 29, dataCodewords: 55, ecPerBlock: 15, blocks: 1, align: [6, 22] },
  { version: 4, size: 33, dataCodewords: 80, ecPerBlock: 20, blocks: 1, align: [6, 26] },
  { version: 5, size: 37, dataCodewords: 108, ecPerBlock: 26, blocks: 1, align: [6, 30] },
  { version: 6, size: 41, dataCodewords: 136, ecPerBlock: 18, blocks: 2, align: [6, 34] },
];

function computeFormatBitsEccL(mask) {
  const data = (0b01 << 3) | (mask & 0x07);
  let rem = data << 10;
  for (let bit = 14; bit >= 10; bit--) {
    if ((rem >>> bit) & 1) rem ^= 0x537 << (bit - 10);
  }
  return (((data << 10) | rem) ^ 0x5412) & 0x7fff;
}

function maskCondition(mask, r, c) {
  switch (mask) {
    case 0:
      return (r + c) % 2 === 0;
    case 1:
      return r % 2 === 0;
    case 2:
      return c % 3 === 0;
    case 3:
      return (r + c) % 3 === 0;
    case 4:
      return (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0;
    case 5:
      return ((r * c) % 2) + ((r * c) % 3) === 0;
    case 6:
      return (((r * c) % 2) + ((r * c) % 3)) % 2 === 0;
    default:
      return (((r + c) % 2) + ((r * c) % 3)) % 2 === 0;
  }
}

function writeFormatBits(modules, isFunction, size, mask) {
  const bits = computeFormatBitsEccL(mask);
  const getBit = (idx) => (bits >>> idx) & 1;

  // Copy 1 around top-left finder (MSB bit 14 -> LSB bit 0)
  const tlCoords = [
    [8, 0],
    [8, 1],
    [8, 2],
    [8, 3],
    [8, 4],
    [8, 5],
    [8, 7],
    [8, 8],
    [7, 8],
    [5, 8],
    [4, 8],
    [3, 8],
    [2, 8],
    [1, 8],
    [0, 8],
  ];
  for (let i = 0; i < 15; i++) {
    const [r, c] = tlCoords[i];
    modules[r][c] = getBit(14 - i);
    isFunction[r][c] = true;
  }

  // Copy 2 split between bottom-left (bits 14..8) and top-right (bits 7..0)
  for (let i = 0; i < 7; i++) {
    const r = size - 1 - i;
    modules[r][8] = getBit(14 - i);
    isFunction[r][8] = true;
  }
  for (let i = 0; i < 8; i++) {
    const c = size - 8 + i;
    modules[8][c] = getBit(7 - i);
    isFunction[8][c] = true;
  }
}

function qrPenaltyScore(modules, size) {
  let score = 0;
  // Rule 1: runs of >= 5 identical modules in rows/cols
  for (let r = 0; r < size; r++) {
    let runLen = 1;
    for (let c = 1; c < size; c++) {
      if (modules[r][c] === modules[r][c - 1]) {
        runLen += 1;
        if (runLen === 5) score += 3;
        else if (runLen > 5) score += 1;
      } else {
        runLen = 1;
      }
    }
  }
  for (let c = 0; c < size; c++) {
    let runLen = 1;
    for (let r = 1; r < size; r++) {
      if (modules[r][c] === modules[r - 1][c]) {
        runLen += 1;
        if (runLen === 5) score += 3;
        else if (runLen > 5) score += 1;
      } else {
        runLen = 1;
      }
    }
  }
  // Rule 2: 2x2 solid blocks
  for (let r = 0; r < size - 1; r++) {
    for (let c = 0; c < size - 1; c++) {
      const m = modules[r][c];
      if (m === modules[r][c + 1] && m === modules[r + 1][c] && m === modules[r + 1][c + 1]) {
        score += 3;
      }
    }
  }
  return score;
}

export function encodeQrMatrix(text) {
  const payload = Buffer.from(String(text), 'utf8');
  const spec = QR_ECC_L_VERSIONS.find((s) => s && payload.length + 2 <= s.dataCodewords);
  if (!spec) {
    throw new Error(`QR text too long (${payload.length} bytes; max 134 bytes supported)`);
  }

  // Encode bit stream: 4-bit Byte mode (0100) + 8-bit length + payload + terminator + pad bytes
  const bits = [];
  const pushBits = (val, len) => {
    for (let i = len - 1; i >= 0; i--) bits.push((val >>> i) & 1);
  };
  pushBits(0b0100, 4);
  pushBits(payload.length, 8);
  for (const b of payload) pushBits(b, 8);

  const capacityBits = spec.dataCodewords * 8;
  const termLen = Math.min(4, capacityBits - bits.length);
  for (let i = 0; i < termLen; i++) bits.push(0);
  while (bits.length % 8 !== 0) bits.push(0);

  const dataBytes = new Uint8Array(spec.dataCodewords);
  for (let i = 0; i < bits.length / 8; i++) {
    let b = 0;
    for (let k = 0; k < 8; k++) b = (b << 1) | bits[i * 8 + k];
    dataBytes[i] = b;
  }
  const padBytes = [0xec, 0x11];
  for (let i = bits.length / 8, p = 0; i < spec.dataCodewords; i++, p ^= 1) {
    dataBytes[i] = padBytes[p];
  }

  // Reed-Solomon blocks & interleaving
  const codewords = [];
  if (spec.blocks === 1) {
    const ec = rsRemainder(dataBytes, spec.ecPerBlock);
    codewords.push(...dataBytes, ...ec);
  } else {
    const blockLen = spec.dataCodewords / spec.blocks;
    const dBlocks = [];
    const eBlocks = [];
    for (let b = 0; b < spec.blocks; b++) {
      const slice = dataBytes.subarray(b * blockLen, (b + 1) * blockLen);
      dBlocks.push(slice);
      eBlocks.push(rsRemainder(slice, spec.ecPerBlock));
    }
    for (let i = 0; i < blockLen; i++) {
      for (let b = 0; b < spec.blocks; b++) codewords.push(dBlocks[b][i]);
    }
    for (let i = 0; i < spec.ecPerBlock; i++) {
      for (let b = 0; b < spec.blocks; b++) codewords.push(eBlocks[b][i]);
    }
  }

  const codewordBits = [];
  for (const cw of codewords) {
    for (let i = 7; i >= 0; i--) codewordBits.push((cw >>> i) & 1);
  }

  const { size, align, version } = spec;
  const baseModules = Array.from({ length: size }, () => new Uint8Array(size));
  const isFunction = Array.from({ length: size }, () => new Array(size).fill(false));

  const setFunc = (r, c, dark) => {
    if (r < 0 || r >= size || c < 0 || c >= size) return;
    baseModules[r][c] = dark ? 1 : 0;
    isFunction[r][c] = true;
  };

  // Finder patterns + separators
  const drawFinder = (topR, leftC) => {
    for (let dr = -1; dr <= 7; dr++) {
      for (let dc = -1; dc <= 7; dc++) {
        const inOuter = dr >= 0 && dr <= 6 && dc >= 0 && dc <= 6;
        const onRing = dr === 0 || dr === 6 || dc === 0 || dc === 6;
        const inCore = dr >= 2 && dr <= 4 && dc >= 2 && dc <= 4;
        setFunc(topR + dr, leftC + dc, inOuter && (onRing || inCore));
      }
    }
  };
  drawFinder(0, 0);
  drawFinder(0, size - 7);
  drawFinder(size - 7, 0);

  // Timing patterns
  for (let i = 8; i < size - 8; i++) {
    setFunc(6, i, i % 2 === 0);
    setFunc(i, 6, i % 2 === 0);
  }

  // Alignment patterns
  for (let ai = 0; ai < align.length; ai++) {
    for (let aj = 0; aj < align.length; aj++) {
      const isFinderCorner =
        (ai === 0 && aj === 0) ||
        (ai === 0 && aj === align.length - 1) ||
        (ai === align.length - 1 && aj === 0);
      if (isFinderCorner) continue;
      const cr = align[ai];
      const cc = align[aj];
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const dist = Math.max(Math.abs(dr), Math.abs(dc));
          setFunc(cr + dr, cc + dc, dist !== 1);
        }
      }
    }
  }

  // Dark module + format reservation
  setFunc(4 * version + 9, 8, true);
  writeFormatBits(baseModules, isFunction, size, 0);

  // Place data bits in upward/downward 2-column zigzag
  let bitIdx = 0;
  let upward = true;
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5; // skip vertical timing line
    for (let step = 0; step < size; step++) {
      const r = upward ? size - 1 - step : step;
      for (const c of [right, right - 1]) {
        if (!isFunction[r][c]) {
          baseModules[r][c] = bitIdx < codewordBits.length ? codewordBits[bitIdx++] : 0;
        }
      }
    }
    upward = !upward;
  }

  // Pick lowest-penalty mask among 0..7
  let bestMask = 0;
  let bestPenalty = Infinity;
  let bestModules = null;
  for (let m = 0; m < 8; m++) {
    const candidate = baseModules.map((row) => Uint8Array.from(row));
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (!isFunction[r][c] && maskCondition(m, r, c)) {
          candidate[r][c] ^= 1;
        }
      }
    }
    writeFormatBits(candidate, isFunction, size, m);
    const pen = qrPenaltyScore(candidate, size);
    if (pen < bestPenalty) {
      bestPenalty = pen;
      bestMask = m;
      bestModules = candidate;
    }
  }

  return { size, version, mask: bestMask, modules: bestModules };
}

export function renderQrSvg(text, { margin = 3 } = {}) {
  const { size, modules } = encodeQrMatrix(text);
  const dim = size + margin * 2;
  const parts = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (modules[r][c]) {
        parts.push(`M${c + margin},${r + margin}h1v1h-1z`);
      }
    }
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dim} ${dim}" ` +
    `shape-rendering="crispEdges" role="img" aria-label="QR Code">` +
    `<rect width="${dim}" height="${dim}" fill="#ffffff"/>` +
    `<path d="${parts.join('')}" fill="#111827"/>` +
    `</svg>`
  );
}

export function renderQrTerminal(text, { margin = 2 } = {}) {
  const { size, modules } = encodeQrMatrix(text);
  const dim = size + margin * 2;
  const isDark = (r, c) => {
    const mr = r - margin;
    const mc = c - margin;
    if (mr < 0 || mr >= size || mc < 0 || mc >= size) return false;
    return Boolean(modules[mr][mc]);
  };
  const lines = [];
  for (let r = 0; r < dim; r += 2) {
    let line = '  ';
    for (let c = 0; c < dim; c++) {
      const top = isDark(r, c);
      const bot = r + 1 < dim ? isDark(r + 1, c) : false;
      // Quiet zone should stay bright/contrasted regardless of dark/light terminal theme:
      // invert so white background is solid block and dark modules are spaces, or vice versa.
      // Standard light plate (`█` = white, ` ` = black):
      if (!top && !bot) line += '█';
      else if (!top && bot) line += '▀';
      else if (top && !bot) line += '▄';
      else line += ' ';
    }
    lines.push(line);
  }
  return lines.join('\n');
}

// ============================================================================
// 2. LAN IPv4 Detection & Local TLS Certificate Provisioning (mkcert / openssl)
// ============================================================================
function isPrivateLanIpv4(ip) {
  if (ip.startsWith('192.168.') || ip.startsWith('10.')) return true;
  const m = ip.match(/^172\.(\d+)\./);
  if (m) {
    const second = Number(m[1]);
    return second >= 16 && second <= 31;
  }
  return false;
}

export function detectLanAddresses() {
  const nets = os.networkInterfaces();
  const candidates = [];
  for (const [ifName, list] of Object.entries(nets)) {
    if (!Array.isArray(list)) continue;
    // Deprioritize VPN/virtual tunnels
    const isVirtual = /^(utun|awdl|llw|bridge|lo|docker|veth|tun|tap)/i.test(ifName);
    const isPrimaryWifi = /^(en0|en1|wlan0|wlp|eth0)/i.test(ifName);
    for (const info of list) {
      if (!info || info.internal) continue;
      const fam = info.family === 4 || info.family === 'IPv4';
      if (!fam || !info.address) continue;
      let rank = 10;
      if (isPrimaryWifi && isPrivateLanIpv4(info.address)) rank = 1;
      else if (!isVirtual && isPrivateLanIpv4(info.address)) rank = 2;
      else if (!isVirtual) rank = 5;
      else rank = 20;
      candidates.push({ ifName, address: info.address, rank });
    }
  }
  candidates.sort((a, b) => a.rank - b.rank);
  const allIps = [...new Set(candidates.map((c) => c.address))];
  return {
    primaryIp: allIps[0] || '127.0.0.1',
    allIps: allIps.length ? allIps : ['127.0.0.1'],
  };
}

function findExecutable(names) {
  for (const candidate of names) {
    if (candidate.startsWith('/')) {
      if (existsSync(candidate)) return candidate;
      continue;
    }
    try {
      const out = execFileSync('which', [candidate], { stdio: ['ignore', 'pipe', 'ignore'] })
        .toString('utf8')
        .trim();
      if (out && existsSync(out)) return out;
    } catch {
      // continue
    }
  }
  return null;
}

export function ensureLocalTlsCertificate({ certDir, hosts }) {
  mkdirSync(certDir, { recursive: true });
  const certPath = join(certDir, 'field-study-cert.pem');
  const keyPath = join(certDir, 'field-study-key.pem');
  const metaPath = join(certDir, 'field-study-hosts.json');
  const uniqueHosts = [...new Set(['localhost', '127.0.0.1', ...hosts.filter(Boolean)])];

  const mkcertBin = findExecutable(['/opt/homebrew/bin/mkcert', '/usr/local/bin/mkcert', 'mkcert']);
  let caPemPath = null;
  if (mkcertBin) {
    try {
      const caRoot = execFileSync(mkcertBin, ['-CAROOT'], { stdio: ['ignore', 'pipe', 'ignore'] })
        .toString('utf8')
        .trim();
      const candidateCa = join(caRoot, 'rootCA.pem');
      if (existsSync(candidateCa)) caPemPath = candidateCa;
    } catch {
      // ignore
    }
  }

  if (existsSync(certPath) && existsSync(keyPath) && existsSync(metaPath)) {
    try {
      const prev = JSON.parse(readFileSync(metaPath, 'utf8'));
      const covered = Array.isArray(prev.hosts) && uniqueHosts.every((h) => prev.hosts.includes(h));
      if (covered) {
        return {
          cert: readFileSync(certPath),
          key: readFileSync(keyPath),
          provider: prev.provider || (mkcertBin ? 'mkcert' : 'openssl'),
          caPemPath,
        };
      }
    } catch {
      // regenerate below
    }
  }

  let provider = 'openssl';
  let generated = false;
  if (mkcertBin) {
    try {
      execFileSync(mkcertBin, ['-cert-file', certPath, '-key-file', keyPath, ...uniqueHosts], {
        stdio: 'ignore',
      });
      provider = 'mkcert';
      generated = existsSync(certPath) && existsSync(keyPath);
    } catch {
      generated = false;
    }
  }

  if (!generated) {
    const opensslBin = findExecutable(['/opt/homebrew/bin/openssl', '/usr/bin/openssl', 'openssl']);
    if (!opensslBin) {
      throw new Error('Neither mkcert nor openssl is available to generate local HTTPS certificate.');
    }
    const sanEntries = uniqueHosts.map((h) => (/^\d+\.\d+\.\d+\.\d+$/.test(h) ? `IP:${h}` : `DNS:${h}`));
    execFileSync(
      opensslBin,
      [
        'req',
        '-x509',
        '-newkey',
        'rsa:2048',
        '-nodes',
        '-sha256',
        '-days',
        '365',
        '-keyout',
        keyPath,
        '-out',
        certPath,
        '-subj',
        `/CN=${uniqueHosts[uniqueHosts.length - 1] || 'localhost'}`,
        '-addext',
        `subjectAltName=${sanEntries.join(',')}`,
      ],
      { stdio: 'ignore' }
    );
    provider = 'openssl';
  }

  writeFileSync(metaPath, JSON.stringify({ hosts: uniqueHosts, provider, updatedAt: new Date().toISOString() }, null, 2));
  return {
    cert: readFileSync(certPath),
    key: readFileSync(keyPath),
    provider,
    caPemPath,
  };
}

// ============================================================================
// 3. Shared In-Memory RFC-6455 WebSocket + MQTT 3.1.1 / 5.0 Broker
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
    if ((encoded & 0x80) === 0) return { value, nextOffset: idx };
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

function createSharedMqttBroker() {
  const clients = new Set();
  const retained = new Map();

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
        c.sendMqtt(encodeMqttPublish(topic, payloadBuf, { retain: false, protoLevel: c.protoLevel }));
      }
    }
  }

  function handleMqttPacket(client, firstByte, body) {
    const pktType = firstByte >> 4;
    const flags = firstByte & 0x0f;

    if (pktType === 1) {
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

      for (const [rTopic, rPayload] of retained.entries()) {
        if (newFilters.some((f) => mqttTopicMatches(f, rTopic))) {
          client.sendMqtt(encodeMqttPublish(rTopic, rPayload, { retain: true, protoLevel: client.protoLevel }));
        }
      }
    } else if (pktType === 3) {
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
      client.sendMqtt(Buffer.from([0xd0, 0x00]));
    } else if (pktType === 14) {
      client.will = null;
      client.close();
    }
  }

  function attach(server) {
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
            client.close();
            return;
          }
          if (opcode === 0x09) {
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
  }

  function closeAllClients() {
    for (const c of clients) {
      c.closed = true;
      try {
        c.socket.destroy();
      } catch {
        // ignore
      }
    }
    clients.clear();
  }

  return { attach, closeAllClients, getConnectedCount: () => clients.size };
}

// ============================================================================
// 4. Dual HTTP + HTTPS Field-Study Hub Server
// ============================================================================
function sendJson(res, statusCode, payload) {
  const body = Buffer.from(JSON.stringify(payload), 'utf8');
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Content-Length': body.length,
  });
  res.end(body);
}

function readRequestBody(req, maxBytes = 10 * 1024 * 1024) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = [];
    let total = 0;
    req.on('data', (chunk) => {
      total += chunk.length;
      if (total > maxBytes) {
        rejectBody(new Error('Request body exceeded 10MB limit'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolveBody(Buffer.concat(chunks).toString('utf8')));
    req.on('error', rejectBody);
  });
}

export async function startFieldStudyServer({
  distDir = resolve(WEB_ROOT, 'dist'),
  logFile = resolve(
    REPO_ROOT,
    process.env.FIELD_STUDY_LOG_PATH || 'logs/field_study/field_study_merged.jsonl'
  ),
  certDir = resolve(REPO_ROOT, '.local_certs'),
  httpPort = Number(process.env.PORT_HTTP ?? 8780),
  httpsPort = Number(process.env.PORT_HTTPS ?? 8443),
  host = process.env.HOST || '0.0.0.0',
  roomCode = process.env.FIELD_STUDY_ROOM || 'LOCALSTUDY',
  quiet = false,
} = {}) {
  mkdirSync(dirname(logFile), { recursive: true });

  // Hydrate existing JSONL records & enrollment environment snapshots
  const recordsMap = new Map(); // recordId -> record
  const enrollEnvsMap = new Map(); // key -> env
  const ingestRecordMeta = (rec) => {
    if (rec?.eventType === 'enroll' && rec.profileId && rec.env) {
      const domKey = rec.domain === 'password' ? 'password' : `arena_${rec.lang || 'en'}`;
      const key = `${rec.roomCode || 'LOCAL'}::${rec.profileId}::${domKey}`;
      enrollEnvsMap.set(key, rec.env);
    }
  };

  if (existsSync(logFile)) {
    const raw = readFileSync(logFile, 'utf8');
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const obj = JSON.parse(trimmed);
        if (obj?.recordId && obj?.eventType && !recordsMap.has(obj.recordId)) {
          recordsMap.set(obj.recordId, obj);
          ingestRecordMeta(obj);
        }
      } catch {
        // skip malformed line
      }
    }
  }

  const { primaryIp: lanIp, allIps: allLanIps } = detectLanAddresses();
  const tls = ensureLocalTlsCertificate({ certDir, hosts: allLanIps });
  const broker = createSharedMqttBroker();

  let actualHttpPort = httpPort;
  let actualHttpsPort = httpsPort;

  const logFileRel = relative(REPO_ROOT, logFile) || logFile;
  const hubBannerScript = `<script>window.__KEYPRINT_FIELD_HUB={active:true,defaultRoom:${JSON.stringify(
    roomCode
  )}};</script>`;

  const handleHttpRequest = async (req, res) => {
    const urlObj = new URL(req.url || '/', 'http://127.0.0.1');
    const pathname = decodeURIComponent(urlObj.pathname);

    if (pathname.endsWith('/api/field-study/info')) {
      const httpsUrl = `https://${lanIp}:${actualHttpsPort}/?eval=1`;
      const httpUrl = `http://${lanIp}:${actualHttpPort}/?eval=1`;
      const laptopUrl = `http://localhost:${actualHttpPort}/?eval=1`;
      return sendJson(res, 200, {
        ok: true,
        hubVersion: '1.0',
        lanIp,
        allLanIps,
        httpPort: actualHttpPort,
        httpsPort: actualHttpsPort,
        httpsUrl,
        httpUrl,
        laptopUrl,
        roomCode,
        savedRecordCount: recordsMap.size,
        connectedPeers: broker.getConnectedCount(),
        logFileRel,
        tlsProvider: tls.provider,
        hasCaCert: Boolean(tls.caPemPath),
        caCertUrl: './api/field-study/ca.pem',
        qrSvgHttps: renderQrSvg(httpsUrl),
        qrSvgHttp: renderQrSvg(httpUrl),
      });
    }

    if (pathname.endsWith('/api/field-study/records')) {
      if (req.method === 'GET') {
        const allRecords = Array.from(recordsMap.values());
        const jsonl =
          allRecords.map((r) => JSON.stringify(r)).join('\n') + (allRecords.length ? '\n' : '');
        const enrollEnvs = Array.from(enrollEnvsMap.entries()).map(([key, env]) => ({ key, env }));
        return sendJson(res, 200, {
          ok: true,
          total: allRecords.length,
          jsonl,
          enrollEnvs,
        });
      }

      if (req.method === 'POST') {
        try {
          const rawBody = await readRequestBody(req);
          const parsed = JSON.parse(rawBody || '{}');
          const incoming = Array.isArray(parsed.records)
            ? parsed.records
            : parsed.record
              ? [parsed.record]
              : [];
          const incomingEnvs = Array.isArray(parsed.enrollEnvs) ? parsed.enrollEnvs : [];

          for (const item of incomingEnvs) {
            if (item?.key && item?.env && typeof item.env === 'object') {
              enrollEnvsMap.set(String(item.key), item.env);
            }
          }

          let added = 0;
          let skipped = 0;
          const linesToAppend = [];
          for (const rec of incoming) {
            if (!rec || typeof rec !== 'object' || !rec.recordId || !rec.eventType) {
              skipped += 1;
              continue;
            }
            const id = String(rec.recordId);
            if (recordsMap.has(id)) {
              skipped += 1;
              continue;
            }
            recordsMap.set(id, rec);
            ingestRecordMeta(rec);
            linesToAppend.push(JSON.stringify(rec));
            added += 1;
          }

          if (linesToAppend.length > 0) {
            appendFileSync(logFile, linesToAppend.join('\n') + '\n', 'utf8');
            if (!quiet) {
              const latest = incoming[incoming.length - 1];
              const envObj = latest?.env || latest?.queryEnv || {};
              console.log(
                `[field-hub] +${added}건 저장 (누적 ${recordsMap.size}건) · ${latest?.eventType}/${latest?.domain} · ${
                  envObj.participantId || '?'
                } · ${envObj.keyboardType || '?'} (${envObj.os || '?'})`
              );
            }
          }

          return sendJson(res, 200, {
            ok: true,
            added,
            skipped,
            total: recordsMap.size,
            logFileRel,
          });
        } catch (err) {
          return sendJson(res, 400, { ok: false, error: err?.message || 'Invalid JSON payload' });
        }
      }

      res.writeHead(405);
      res.end('Method Not Allowed');
      return;
    }

    if (pathname.endsWith('/api/field-study/ca.pem')) {
      if (!tls.caPemPath || !existsSync(tls.caPemPath)) {
        res.writeHead(404);
        res.end('mkcert rootCA.pem not available');
        return;
      }
      const pem = readFileSync(tls.caPemPath);
      res.writeHead(200, {
        'Content-Type': 'application/x-x509-ca-cert',
        'Content-Disposition': 'attachment; filename="keyprint-mkcert-rootCA.pem"',
        'Cache-Control': 'no-store',
        'Content-Length': pem.length,
      });
      res.end(pem);
      return;
    }

    // Static asset serving from `distDir`
    let relPath = pathname === '/' || pathname === '' ? '/index.html' : pathname;
    const safeRel = normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, '');
    let filePath = join(distDir, safeRel);
    if (!filePath.startsWith(distDir) || !existsSync(filePath) || statSync(filePath).isDirectory()) {
      if (extname(safeRel) && extname(safeRel) !== '.html') {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      filePath = join(distDir, 'index.html');
    }

    try {
      const ext = extname(filePath).toLowerCase();
      const mime = MIME_TYPES[ext] || 'application/octet-stream';
      if (ext === '.html') {
        const htmlRaw = readFileSync(filePath, 'utf8');
        const injected = htmlRaw.includes('</head>')
          ? htmlRaw.replace('</head>', `${hubBannerScript}</head>`)
          : `${hubBannerScript}${htmlRaw}`;
        const buf = Buffer.from(injected, 'utf8');
        res.writeHead(200, {
          'Content-Type': mime,
          'Cache-Control': 'no-store',
          'Content-Length': buf.length,
        });
        res.end(buf);
        return;
      }
      const data = readFileSync(filePath);
      res.writeHead(200, {
        'Content-Type': mime,
        'Cache-Control': 'no-store',
        'Content-Length': data.length,
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  };

  const httpServer = createHttpServer((req, res) => {
    handleHttpRequest(req, res).catch((err) => {
      if (!res.headersSent) sendJson(res, 500, { ok: false, error: err?.message || 'Server error' });
    });
  });

  const httpsServer = createHttpsServer({ key: tls.key, cert: tls.cert }, (req, res) => {
    handleHttpRequest(req, res).catch((err) => {
      if (!res.headersSent) sendJson(res, 500, { ok: false, error: err?.message || 'Server error' });
    });
  });

  broker.attach(httpServer);
  broker.attach(httpsServer);

  await new Promise((resolveListen, rejectListen) => {
    httpServer.once('error', rejectListen);
    httpServer.listen(httpPort, host, () => {
      httpServer.removeListener('error', rejectListen);
      resolveListen();
    });
  });
  actualHttpPort = httpServer.address().port;

  await new Promise((resolveListen, rejectListen) => {
    httpsServer.once('error', rejectListen);
    httpsServer.listen(httpsPort, host, () => {
      httpsServer.removeListener('error', rejectListen);
      resolveListen();
    });
  });
  actualHttpsPort = httpsServer.address().port;

  const httpsUrl = `https://${lanIp}:${actualHttpsPort}/?eval=1`;
  const httpUrl = `http://${lanIp}:${actualHttpPort}/?eval=1`;
  const laptopUrl = `http://localhost:${actualHttpPort}/?eval=1`;

  if (!quiet) {
    console.log('\n========================================================================');
    console.log('  KeyPrint Guard · 로컬 다기기(노트북 + 외장키보드 + 스마트폰) 실측 허브');
    console.log('========================================================================');
    console.log(`  [1] 노트북 브라우저 접속 : ${laptopUrl}`);
    console.log(`  [2] 스마트폰 권장 (HTTPS): ${httpsUrl}  (터치 + 모션센서 지원)`);
    console.log(`  [3] 스마트폰 즉시 (HTTP) : ${httpUrl}  (경고창 없는 즉시 터치)`);
    console.log(`  [4] 로컬 Wi-Fi 동기화 방 : #${roomCode} (동일 공유기 내 자동 연결 · 외부 서버 미사용)`);
    console.log(`  [5] 통합 로그 자동 저장  : ${logFileRel} (현재 ${recordsMap.size}건)`);
    console.log('------------------------------------------------------------------------');
    console.log('  스마트폰 카메라로 아래 QR 코드(HTTPS)를 스캔하거나 화면의 QR을 스캔하세요:');
    console.log('  (최초 1회 브라우저에서 [고급 -> 안전하지 않음으로 이동 / 웹사이트 방문] 선택)\n');
    console.log(renderQrTerminal(httpsUrl));
    console.log('\n  실측 완료 후 통합 분석 실행:');
    console.log(`    .venv/bin/python src/analyze_field_study.py ${logFileRel}`);
    console.log('========================================================================\n');
  }

  return {
    httpServer,
    httpsServer,
    httpPort: actualHttpPort,
    httpsPort: actualHttpsPort,
    lanIp,
    allLanIps,
    laptopUrl,
    httpUrl,
    httpsUrl,
    roomCode,
    logFile,
    logFileRel,
    getRecordCount: () => recordsMap.size,
    async close() {
      broker.closeAllClients();
      await Promise.all([
        new Promise((r) => httpServer.close(() => r())),
        new Promise((r) => httpsServer.close(() => r())),
      ]);
    },
  };
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (isMain) {
  const distIndex = resolve(WEB_ROOT, 'dist', 'index.html');
  if (!existsSync(distIndex)) {
    console.log('[field-hub] web/dist 빌드가 없어 자동 빌드를 먼저 수행합니다...');
    execFileSync('npm', ['run', 'build'], { cwd: WEB_ROOT, stdio: 'inherit' });
  }
  const hub = await startFieldStudyServer();
  const shutdown = async () => {
    console.log('\n[field-hub] 서버를 종료합니다.');
    await hub.close();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}
