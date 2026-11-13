/**
 * Offline browser smoke test of the production build (web/dist).
 *
 *   npm run build && node scripts/offline_smoke.mjs
 *
 * Every non-localhost request is aborted and every WebSocket (MQTT room sync) is mocked
 * without a server, so no profile ever leaves the machine. Uses generic typist names only.
 *
 * Desktop (physical keyboard events):  enroll 2 typists x 5 sentences (with typos + Backspace),
 *   blind 1:N identification from 2 sentences, password enroll x4 (both encoders), genuine accept with
 *   template adaptation / impostor reject without adaptation, samples.
 * Mobile emulation (soft keyboard: keydown key='Unidentified' code='', text via `input` events,
 *   Korean IME composition states): the path that used to fail with
 *   "추출된 연속 리듬 구간이 부족합니다" must now enroll 5 sentences, identify, enroll a password
 *   and accept the genuine thumb typist; the password verdict is labelled "모바일 참고용".
 *
 * Impostor checks with a leaked password are split in two:
 *   - functional (must pass): a desktop impostor whose tempo AND rhythm shape differ.
 *   - known limitation (reported, not fatal): other people who differ only in rhythm shape — two thumb
 *     typists on a phone and one same-tempo keyboard typist. On a user-chosen password the deployed
 *     score (AS-Norm against the CMU-password cohort) is inflated, so these can be accepted. Real-data
 *     numbers: README section 5.6 / PR #1. They print "⚠" and are counted separately.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { textToKeystrokes } from '../src/features.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.resolve(__dirname, '..', 'dist');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SHOT_DIR = process.env.SMOKE_SHOT_DIR || null;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function serve(root) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const rel = decodeURIComponent((req.url || '/').split('?')[0]).replace(/^\/+/, '') || 'index.html';
      const full = path.join(root, rel);
      if (!full.startsWith(root) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(full).toLowerCase()] || 'application/octet-stream' });
      fs.createReadStream(full).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` }));
  });
}

// ---------------------------------------------------------------------------------------
// Event plans (absolute ms offsets) executed inside the page with real timers
// ---------------------------------------------------------------------------------------
let seed = 7;
const rand = () => {
  seed = (seed * 16807) % 2147483647;
  return seed / 2147483647;
};

/**
 * Real typists differ per key pair (fast on some digraphs, slow on others), not only in overall
 * tempo. A deterministic per-typist factor in [0.65, 1.35] gives each cadence its own shape.
 * (A flat rhythm that differs only in tempo is a known weak spot — see README limitations.)
 */
function factor(seedStr, key) {
  let x = 2166136261;
  for (const c of `${seedStr}|${key}`) x = Math.imul(x ^ c.codePointAt(0), 16777619) >>> 0;
  return 0.65 + 0.7 * ((x % 10007) / 10007);
}

const CODE_FOR = { ' ': 'Space', '.': 'Period', ',': 'Comma', "'": 'Quote', '-': 'Minus' };
const codeForChar = (ch) =>
  CODE_FOR[ch] || (/^[a-z]$/i.test(ch) ? `Key${ch.toUpperCase()}` : /^[0-9]$/.test(ch) ? `Digit${ch}` : 'KeyA');

/** Physical keyboard plan: keydown -> input -> keyup with a cadence (optionally a typo + Backspace repair). */
function keyboardPlan(text, cadence, { typo = false } = {}) {
  const plan = [];
  let t = 20;
  const chars = [...text];
  const typoAt = typo ? Math.floor(chars.length * 0.4) : -1;
  const press = (ch) => {
    const code = codeForChar(ch);
    const hold = cadence.holdMs * factor(cadence.seed, `h${ch}`) * (1 + cadence.jitter * (rand() - 0.5));
    plan.push({ t, type: 'keydown', key: ch, code, shift: /^[A-Z]$/.test(ch), caps: Boolean(cadence.capsLock) && /^[A-Z]$/.test(ch) });
    plan.push({ t, type: 'input', op: 'insert', data: ch, inputType: 'insertText' });
    plan.push({ t: t + hold, type: 'keyup', key: ch, code });
    t += cadence.ddMs * factor(cadence.seed, `d${ch}`) * (1 + cadence.jitter * (rand() - 0.5)) + (ch === ' ' ? cadence.spaceExtraMs : 0);
  };
  chars.forEach((ch, i) => {
    if (i === typoAt) {
      press('x');
      t += cadence.detectMs;
      plan.push({ t, type: 'keydown', key: 'Backspace', code: 'Backspace' });
      plan.push({ t, type: 'input', op: 'delete', inputType: 'deleteContentBackward' });
      plan.push({ t: t + cadence.bsHoldMs, type: 'keyup', key: 'Backspace', code: 'Backspace' });
      t += cadence.resumeMs;
    }
    press(ch);
  });
  plan.push({ t: t + 120, type: 'keydown', key: 'Enter', code: 'Enter' });
  return plan.sort((a, b) => a.t - b.t);
}

const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ';
const JONG = ['', ...'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'];
const V_FIRST = { ㅘ: 'ㅗ', ㅙ: 'ㅗ', ㅚ: 'ㅗ', ㅝ: 'ㅜ', ㅞ: 'ㅜ', ㅟ: 'ㅜ', ㅢ: 'ㅡ' };
const J_FIRST = { ㄳ: 'ㄱ', ㄵ: 'ㄴ', ㄶ: 'ㄴ', ㄺ: 'ㄹ', ㄻ: 'ㄹ', ㄼ: 'ㄹ', ㄽ: 'ㄹ', ㄾ: 'ㄹ', ㄿ: 'ㄹ', ㅀ: 'ㄹ', ㅄ: 'ㅂ' };
const syl = (c, v, j = 0) => String.fromCodePoint(0xac00 + c * 588 + v * 28 + j);

/** IME intermediate states a 2-beolsik soft keyboard shows while composing one character. */
function compositionStates(ch) {
  const cp = ch.codePointAt(0);
  if (cp < 0xac00 || cp > 0xd7a3) return [ch];
  const idx = cp - 0xac00;
  const c = Math.floor(idx / 588);
  const v = Math.floor((idx % 588) / 28);
  const j = idx % 28;
  const states = [CHO[c]];
  const vowel = JUNG[v];
  if (V_FIRST[vowel]) states.push(syl(c, JUNG.indexOf(V_FIRST[vowel])));
  states.push(syl(c, v));
  if (j) {
    const fin = JONG[j];
    if (J_FIRST[fin]) states.push(syl(c, v, JONG.indexOf(J_FIRST[fin])));
    states.push(syl(c, v, j));
  }
  return states;
}

/** Soft keyboard plan: every tap = keydown(Unidentified, code '') + value change; keyup arrives at once. */
function touchPlan(text, cadence, { typo = false } = {}) {
  const plan = [];
  let t = 20;
  let committed = '';
  const chars = [...text];
  const typoAt = typo ? Math.floor(chars.length * 0.45) : -1;
  const tap = (value, inputType) => {
    plan.push({ t, type: 'keydown', key: 'Unidentified', code: '' });
    plan.push({ t, type: 'input', value, inputType });
    plan.push({ t: t + 1, type: 'keyup', key: 'Unidentified', code: '' });
  };
  chars.forEach((ch, i) => {
    if (i === typoAt) {
      tap(committed + 'ㅋ', 'insertCompositionText');
      t += cadence.detectMs;
      tap(committed, 'deleteContentBackward');
      t += cadence.resumeMs;
    }
    const states = compositionStates(ch);
    for (const st of states) {
      tap(committed + st, /^[ㄱ-ㆎ가-힣]$/.test(st) ? 'insertCompositionText' : 'insertText');
      t += cadence.ddMs * factor(cadence.seed, `d${st}`) * (1 + cadence.jitter * (rand() - 0.5)) + (ch === ' ' ? cadence.spaceExtraMs : 0);
    }
    committed += ch;
  });
  plan.push({ t: t + 150, type: 'keydown', key: 'Enter', code: 'Enter' });
  return plan;
}

/**
 * Field values a real 2-beolsik IME shows after each jamo key, including the final consonant moving
 * to the next syllable (하 + ㅅ → 핫, + ㅔ → 하세) and compound vowels / final clusters.
 */
const IME_CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const IME_JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ';
const IME_JONG = ['', ...'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'];
const IME_VOWEL_PAIR = { ㅗㅏ: 'ㅘ', ㅗㅐ: 'ㅙ', ㅗㅣ: 'ㅚ', ㅜㅓ: 'ㅝ', ㅜㅔ: 'ㅞ', ㅜㅣ: 'ㅟ', ㅡㅣ: 'ㅢ' };
const IME_CLUSTER = { ㄱㅅ: 'ㄳ', ㄴㅈ: 'ㄵ', ㄴㅎ: 'ㄶ', ㄹㄱ: 'ㄺ', ㄹㅁ: 'ㄻ', ㄹㅂ: 'ㄼ', ㄹㅅ: 'ㄽ', ㄹㅌ: 'ㄾ', ㄹㅍ: 'ㄿ', ㄹㅎ: 'ㅀ', ㅂㅅ: 'ㅄ' };
const IME_SPLIT = Object.fromEntries(Object.entries(IME_CLUSTER).map(([k, v]) => [v, [...k]]));
function imeValues(text) {
  const out = [];
  let done = '';
  let cur = null; // { cho, jung, jong } jamo strings
  const render = (c) => {
    if (!c) return '';
    if (c.cho && c.jung) return String.fromCodePoint(0xac00 + IME_CHO.indexOf(c.cho) * 588 + IME_JUNG.indexOf(c.jung) * 28 + IME_JONG.indexOf(c.jong || ''));
    return c.cho || c.jung;
  };
  for (const k of textToKeystrokes(text)) {
    const isVowel = IME_JUNG.includes(k);
    const isCons = IME_CHO.includes(k) || IME_JONG.includes(k);
    if (!isVowel && !isCons) { done += render(cur) + k; cur = null; out.push(done); continue; }
    if (isCons) {
      if (cur?.cho && cur.jung && !cur.jong && IME_JONG.includes(k)) cur.jong = k;
      else if (cur?.jong && IME_CLUSTER[cur.jong + k]) cur.jong = IME_CLUSTER[cur.jong + k];
      else { done += render(cur); cur = { cho: k, jung: '', jong: '' }; }
    } else if (cur?.cho && !cur.jung) cur.jung = k;
    else if (cur?.jung && !cur.jong && IME_VOWEL_PAIR[cur.jung + k]) cur.jung = IME_VOWEL_PAIR[cur.jung + k];
    else if (cur?.jong) {
      const [keep, move] = IME_SPLIT[cur.jong] || ['', cur.jong];
      done += render({ ...cur, jong: keep });
      cur = { cho: move, jung: k, jong: '' };
    } else { done += render(cur); cur = { cho: '', jung: k, jong: '' }; }
    out.push(done + render(cur));
  }
  return out;
}

/** iOS / Samsung style soft keyboard: a changed syllable arrives as delete + insert input events. */
function iosPlan(text, cadence) {
  const plan = [];
  let t = 20;
  let prev = '';
  for (const v of imeValues(text)) {
    plan.push({ t, type: 'keydown', key: 'Unidentified', code: '' });
    if (v.startsWith(prev) && v.length > prev.length) {
      plan.push({ t, type: 'input', value: v, inputType: 'insertText' });
    } else {
      plan.push({ t, type: 'input', value: prev.slice(0, -1), inputType: 'deleteContentBackward' });
      plan.push({ t: t + 0.6, type: 'input', value: v, inputType: 'insertText' });
    }
    plan.push({ t: t + 1, type: 'keyup', key: 'Unidentified', code: '' });
    const ch = [...v].at(-1) || '';
    t += cadence.ddMs * factor(cadence.seed, `d${ch}`) * (1 + cadence.jitter * (rand() - 0.5)) + (ch === ' ' ? cadence.spaceExtraMs : 0);
    prev = v;
  }
  plan.push({ t: t + 150, type: 'keydown', key: 'Enter', code: 'Enter' });
  return plan;
}

async function runPlan(page, selector, plan) {
  await page.evaluate(
    async ({ selector, plan }) => {
      const el = document.querySelector(selector);
      el.focus();
      const t0 = performance.now();
      for (const ev of plan) {
        const wait = t0 + ev.t - performance.now();
        if (wait > 1) await new Promise((r) => setTimeout(r, wait));
        if (ev.type === 'keydown' || ev.type === 'keyup') {
          el.dispatchEvent(new KeyboardEvent(ev.type, {
            key: ev.key, code: ev.code, bubbles: true, cancelable: true,
            shiftKey: Boolean(ev.shift && !ev.caps), modifierCapsLock: Boolean(ev.caps),
          }));
        } else {
          if (ev.value !== undefined) el.value = ev.value;
          else if (ev.op === 'delete') el.value = [...el.value].slice(0, -1).join('');
          else el.value += ev.data;
          el.dispatchEvent(new InputEvent('input', { inputType: ev.inputType, data: ev.data ?? null, bubbles: true }));
        }
      }
    },
    { selector, plan },
  );
  await page.waitForTimeout(200);
}

const FAST = { seed: 'typist-a', holdMs: 112, ddMs: 86, jitter: 0.2, spaceExtraMs: 25, detectMs: 260, resumeMs: 210, bsHoldMs: 70 };
const SLOW = { seed: 'typist-b', holdMs: 64, ddMs: 175, jitter: 0.2, spaceExtraMs: 140, detectMs: 520, resumeMs: 480, bsHoldMs: 110 };
const THUMB = { seed: 'thumb', ddMs: 175, jitter: 0.3, spaceExtraMs: 90, detectMs: 380, resumeMs: 300 };
// Other people on a phone who know the leaked password: one slower with a different rhythm shape,
// one at the SAME tempo as THUMB with only the per-key shape different (the harder case).
const THUMB_SLOW = { seed: 'thumb-other-1', ddMs: 255, jitter: 0.3, spaceExtraMs: 90, detectMs: 380, resumeMs: 300 };
const THUMB_SAME_TEMPO = { seed: 'thumb-other-2', ddMs: 175, jitter: 0.3, spaceExtraMs: 90, detectMs: 380, resumeMs: 300 };
// Keyboard typist with Typist A's tempo and hold length but a different per-key rhythm shape
const FAST_OTHER_SHAPE = { ...FAST, seed: 'typist-c' };

// ---------------------------------------------------------------------------------------
const results = [];
function expect(cond, msg) {
  results.push({ ok: Boolean(cond), msg });
  console.log(`${cond ? '  ✓' : '  ✗'} ${msg}`);
}
const known = [];
function expectKnownLimitation(cond, msg) {
  known.push({ ok: Boolean(cond), msg });
  console.log(`${cond ? '  ✓' : '  ⚠'} ${msg}${cond ? '' : '  [known limitation, not fatal]'}`);
}

async function newOfflineContext(browser, options) {
  const ctx = await browser.newContext(options);
  const blocked = { http: [], ws: [] };
  await ctx.route('**/*', (route) => {
    const u = new URL(route.request().url());
    if (u.hostname === '127.0.0.1' || u.protocol === 'data:' || u.protocol === 'blob:') return route.continue();
    blocked.http.push(u.host);
    return route.abort();
  });
  // Mocked WebSocket: no server is ever contacted (MQTT CONNECT is swallowed)
  await ctx.routeWebSocket(/.*/, (ws) => {
    blocked.ws.push(new URL(ws.url()).host);
  });
  return { ctx, blocked };
}

function watchErrors(page, errors) {
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
}

async function openApp(ctx, url) {
  const page = await ctx.newPage();
  const errors = [];
  watchErrors(page, errors);
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('engine-status-text')?.textContent?.includes('ONNX 준비 완료'), {
    timeout: 20000,
  });
  return { page, errors };
}

const api = (page, fn, arg) => page.evaluate(fn, arg);

async function enrollArena(page, name, lang, planFor) {
  await page.fill('#arena-name-input', name);
  const feedbacks = [];
  const n = await api(page, () => window.__keyprintTestApi.engine.arenaProtocol().enroll);
  for (let s = 0; s < n; s++) {
    const sentence = await api(page, () => window.__keyprintTestApi.state.arenaEnrollSentence);
    await runPlan(page, '#arena-enroll-input', planFor(sentence, s));
    await page.waitForTimeout(250);
    feedbacks.push(await page.textContent('#arena-enroll-feedback'));
  }
  const profile = await api(
    page,
    ({ name, lang }) => {
      const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === name);
      const t = p?.arena?.[lang];
      return t ? { count: t.count, inputMode: t.inputMode, hasBehavior: Boolean(t.behavior), dims: t.centroid?.length } : null;
    },
    { name, lang },
  );
  return { feedbacks, profile, expected: n };
}

async function blindIdentify(page, planFor) {
  await page.click('#arena-sub-blind');
  await page.evaluate(() => { document.getElementById('arena-result-card').hidden = true; });
  const n = await api(page, () => window.__keyprintTestApi.engine.arenaProtocol().query);
  const interim = [];
  for (let i = 0; i < n; i++) {
    const sentence = await api(page, () => window.__keyprintTestApi.state.arenaBlindSentence);
    await runPlan(page, '#arena-blind-input', planFor(sentence));
    await page.waitForTimeout(200);
    if (i < n - 1) {
      interim.push({
        feedback: (await page.textContent('#arena-blind-feedback'))?.trim(),
        cardHidden: await page.evaluate(() => document.getElementById('arena-result-card').hidden),
      });
    }
  }
  await page.waitForSelector('#arena-result-card:not([hidden])', { timeout: 5000 });
  await page.waitForTimeout(150);
  return {
    title: (await page.textContent('#arena-result-title'))?.trim(),
    meta: (await page.textContent('#arena-result-meta'))?.trim(),
    feedback: (await page.textContent('#arena-blind-feedback'))?.trim(),
    interim,
    sentences: n,
  };
}

async function pwTemplate(page, name) {
  return api(page, (name) => {
    const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === name)?.password;
    return p ? { count: p.count, ex9: p.exemplars?.length, ex17: p.enc17?.exemplars?.length, mode: p.inputMode, capsUse: p.capsUse || null, len: p.target?.length } : null;
  }, name);
}

const PW = 'Gachon2026guard'; // 15 characters, one capital typed with Shift

async function enrollPassword(page, name, pw, planFor, reps = 4) {
  await page.click('#nav-guard-btn');
  await page.click('#guard-sub-enroll');
  await page.fill('#guard-enroll-name', name);
  await page.fill('#guard-enroll-pw-target', pw);
  const feedbacks = [];
  for (let r = 0; r < reps; r++) {
    await runPlan(page, '#guard-enroll-input', planFor(pw, r));
    await page.waitForTimeout(200);
    feedbacks.push((await page.textContent('#guard-enroll-feedback'))?.trim());
  }
  return feedbacks;
}

async function verifyPassword(page, pw, plan) {
  await page.evaluate(() => { document.getElementById('guard-result-card').hidden = true; });
  await runPlan(page, '#guard-verify-input', plan);
  await page.waitForSelector('#guard-result-card:not([hidden])', { timeout: 5000 });
  await page.waitForTimeout(150);
  return {
    pass: await page.evaluate(() => document.getElementById('guard-result-card').classList.contains('is-pass')),
    title: (await page.textContent('#guard-result-title'))?.trim(),
    score: (await page.textContent('#guard-result-score'))?.trim(),
    meta: (await page.textContent('#guard-result-meta'))?.trim(),
    kicker: (await page.textContent('#guard-result-kicker'))?.trim(),
    reference: await page.evaluate(() => document.getElementById('guard-result-card').dataset.reference === 'true'),
  };
}

async function run() {
  if (!fs.existsSync(path.join(DIST_DIR, 'index.html'))) throw new Error('web/dist missing — run `npm run build` first');
  const { server, url } = await serve(DIST_DIR);
  const appUrl = `${url}/?room=OFFLINE${Math.floor(1000 + rand() * 8999)}`;
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  const allBlocked = { http: new Set(), ws: new Set() };
  const allErrors = [];

  try {
    // ------------------------------------------------------------- Desktop keyboard
    console.log('[desktop] physical keyboard');
    {
      const { ctx, blocked } = await newOfflineContext(browser, { viewport: { width: 1280, height: 880 } });
      const { page, errors } = await openApp(ctx, appUrl);
      const cfg = await api(page, () => ({
        inCh: window.__keyprintTestApi.engine.config.in_channels,
        version: window.__keyprintTestApi.engine.config.model_version,
        samples: window.__keyprintTestApi.engine.config.mvp_sample_profiles.map((p) => p.name),
      }));
      expect(cfg.inCh === 17, `model config: ${cfg.inCh} channels, ${cfg.version}`);
      expect(cfg.samples.length === 4, `sample profiles in config: ${cfg.samples.join(', ')}`);

      await page.click('#lang-en-btn');
      const a = await enrollArena(page, 'Typist A', 'en', (s) => keyboardPlan(s, FAST, { typo: true }));
      expect(a.expected === 5 && a.profile && a.profile.count === 5 && a.profile.inputMode === 'keyboard' && a.profile.hasBehavior, `Typist A enrolled (${a.expected} sentences): ${JSON.stringify(a.profile)}`);
      const b = await enrollArena(page, 'Typist B', 'en', (s, i) => keyboardPlan(s, SLOW, { typo: i === 0 }));
      expect(b.profile && b.profile.count === 5, `Typist B enrolled: ${JSON.stringify(b.profile)}`);

      const id = await blindIdentify(page, (s) => keyboardPlan(s, FAST, { typo: true }));
      expect(id.sentences === 2 && id.interim[0]?.cardHidden && /1\/2/.test(id.interim[0]?.feedback || ''), `identification waits for the 2nd sentence: "${id.interim[0]?.feedback}"`);
      expect(id.title?.includes('Typist A'), `blind 1:N (keyboard, 2 sentences) → "${id.title}" | ${id.meta}`);
      if (SHOT_DIR) await page.screenshot({ path: path.join(SHOT_DIR, 'smoke_desktop_identify.png') });

      const shortFb = await enrollPassword(page, 'Typist A', 'gachon2026', (pw) => keyboardPlan(pw, FAST), 1);
      expect(/12자/.test(shortFb[0] || ''), `10-character password refused at enrolment: "${shortFb[0]}"`);
      const pwFb = await enrollPassword(page, 'Typist A', PW, (pw, r) => keyboardPlan(pw, FAST, { typo: r === 1 }));
      expect(!pwFb.some((f) => /부족|불일치|일치하지|12자/.test(f || '')), `password enroll x4 (keyboard, ${PW.length} chars): ${pwFb.at(-1) || 'ok'}`);
      const t0 = await pwTemplate(page, 'Typist A');
      expect(t0 && t0.ex9 === 4 && t0.ex17 === 4 && t0.capsUse?.[1] === 4, `password template: both encoders + capital habit (Shift x4): ${JSON.stringify(t0)}`);
      const genuine = await verifyPassword(page, PW, keyboardPlan(PW, FAST));
      expect(genuine.pass && !genuine.reference, `genuine password login → ${genuine.title} ${genuine.score} | ${genuine.meta}`);
      const t1 = await pwTemplate(page, 'Typist A');
      expect(t1.count === t0.count + 1 && /등록 리듬 갱신/.test(genuine.meta), `confident genuine login adapts the template: ${t0.count} -> ${t1.count}`);
      const impostor = await verifyPassword(page, PW, keyboardPlan(PW, SLOW));
      expect(!impostor.pass, `impostor with leaked password → ${impostor.title} ${impostor.score}`);
      const t2 = await pwTemplate(page, 'Typist A');
      expect(t2.count === t1.count, `rejected attempt does not touch the template (count ${t2.count})`);
      // Same impostor rhythm, but capitals typed with CapsLock while the owner always uses Shift
      const kbCaps = await verifyPassword(page, PW, keyboardPlan(PW, { ...FAST_OTHER_SHAPE, capsLock: true }));
      const t3 = await pwTemplate(page, 'Typist A');
      expect(!kbCaps.pass && t3.count === t2.count, `same-tempo impostor typing the capital with CapsLock (owner uses Shift) rejected → ${kbCaps.title} ${kbCaps.score}`);
      const kbShape = await verifyPassword(page, PW, keyboardPlan(PW, FAST_OTHER_SHAPE));
      expectKnownLimitation(!kbShape.pass, `keyboard impostor (same tempo, other rhythm) rejected → ${kbShape.title} ${kbShape.score}`);

      // Samples load + identify with the rebuilt sample gallery present
      await page.click('#open-room-btn');
      const hasSamples = await api(page, () => window.__keyprintTestApi.roomSync.hasSamples());
      if (!hasSamples) await page.click('#toggle-samples-btn');
      await page.click('#close-room-btn');
      const sampleState = await api(page, () =>
        window.__keyprintTestApi.state.profiles.filter((p) => p.isSample).map((p) => ({ name: p.name, ko: Boolean(p.arena?.ko), en: Boolean(p.arena?.en), pw: Boolean(p.password) })),
      );
      expect(sampleState.length === 4 && sampleState.every((s) => s.en && s.pw), `samples loaded: ${JSON.stringify(sampleState)}`);
      await page.click('#nav-arena-btn').catch(() => {});
      const id2 = await blindIdentify(page, (s) => keyboardPlan(s, FAST));
      expect(id2.title?.includes('Typist A'), `blind 1:N with samples present → "${id2.title}"`);

      // Principles tab still renders with a live sandbox capture
      await page.click('#nav-principles-btn');
      await page.waitForSelector('#view-principles:not([hidden])');
      await runPlan(page, '#viz-sandbox-input', keyboardPlan('keyprint live rhythm', FAST).filter((e) => e.key !== 'Enter'));
      const badge = await page.textContent('#viz-signal-badge');
      expect(badge && !badge.includes('예시 파형'), `principles sandbox live badge: ${badge?.trim()}`);
      // Password map lives in the 9ch encoder space: the query must be re-plotted from its password embedding
      await page.click('#viz-map-domain-guard');
      await page.waitForTimeout(200);
      const guardMap = await api(page, () => ({
        svgLen: document.getElementById('viz-step3-svg').innerHTML.length,
        hasPwEmb: Array.isArray(window.__keyprintTestApi.state.viz.lastQuery?.embeddings?.password),
      }));
      expect(guardMap.svgLen > 200 && guardMap.hasPwEmb, `principles password map renders with password-space query: ${JSON.stringify(guardMap)}`);
      await page.click('#viz-map-domain-arena');

      blocked.http.forEach((h) => allBlocked.http.add(h));
      blocked.ws.forEach((h) => allBlocked.ws.add(h));
      allErrors.push(...errors.map((e) => `[desktop] ${e}`));
      await ctx.close();
    }

    // ------------------------------------------------------------- Mobile soft keyboard
    console.log('[mobile] soft keyboard emulation (iPhone viewport, touch, no key codes)');
    {
      const { ctx, blocked } = await newOfflineContext(browser, {
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
        userAgent:
          'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
      });
      const { page, errors } = await openApp(ctx, appUrl);
      const m = await enrollArena(page, '모바일 사용자', 'ko', (s, i) => touchPlan(s, THUMB, { typo: i === 1 }));
      expect(!m.feedbacks.some((f) => (f || '').includes('연속 리듬 구간이 부족')), `no INSUFFICIENT_CLEAN_KEYS: ${JSON.stringify(m.feedbacks)}`);
      expect(m.profile && m.profile.count === 5 && m.profile.inputMode === 'touch', `Korean touch profile enrolled: ${JSON.stringify(m.profile)}`);
      const mid = await blindIdentify(page, (s) => touchPlan(s, THUMB));
      expect(Boolean(mid.title) && !/부족/.test(mid.feedback || ''), `blind 1:N (touch) → "${mid.title}" | ${mid.meta}`);
      // iOS / Samsung keyboards replace the composing syllable with delete + insert events and move the
      // final consonant to the next syllable: none of that may count as a Backspace correction
      const ios = await blindIdentify(page, (s) => iosPlan(s, THUMB));
      expect(/무수정/.test(ios.meta || '') && !/백스페이스/.test(ios.meta || '') && ios.title?.includes('모바일 사용자'),
        `iOS-style Hangul composition: no false Backspace → "${ios.title}" | ${ios.meta}`);
      if (SHOT_DIR) await page.screenshot({ path: path.join(SHOT_DIR, 'smoke_mobile_identify.png') });

      const pwFb = await enrollPassword(page, '모바일 사용자', PW, (pw) => touchPlan(pw, THUMB));
      expect(!pwFb.some((f) => /부족|불일치|일치하지/.test(f || '')), `password enroll x4 (touch): ${pwFb.at(-1) || 'ok'}`);
      const mv = await verifyPassword(page, PW, touchPlan(PW, THUMB));
      expect(mv.pass, `genuine touch password login → ${mv.title} ${mv.score} | ${mv.meta}`);
      expect(mv.reference && /모바일 참고용/.test(mv.kicker || ''), `touch password verdict labelled reference-only: "${mv.kicker}"`);
      const mi1 = await verifyPassword(page, PW, touchPlan(PW, THUMB_SLOW));
      expectKnownLimitation(!mi1.pass, `touch impostor (other rhythm, slower) rejected → ${mi1.title} ${mi1.score}`);
      const mi2 = await verifyPassword(page, PW, touchPlan(PW, THUMB_SAME_TEMPO));
      expectKnownLimitation(!mi2.pass, `touch impostor (same tempo, other rhythm) rejected → ${mi2.title} ${mi2.score}`);

      blocked.http.forEach((h) => allBlocked.http.add(h));
      blocked.ws.forEach((h) => allBlocked.ws.add(h));
      allErrors.push(...errors.map((e) => `[mobile] ${e}`));
      await ctx.close();
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`\nblocked external HTTP hosts: ${[...allBlocked.http].join(', ') || '(none)'}`);
  console.log(`mocked WebSocket hosts (never contacted): ${[...allBlocked.ws].join(', ') || '(none)'}`);
  // Aborted external fonts/sockets are expected offline; anything else is a real error
  const real = allErrors.filter((e) => !/net::ERR_FAILED|ERR_INTERNET_DISCONNECTED|WebSocket|mqtt|fonts\.g/i.test(e));
  expect(real.length === 0, `console/page errors: ${real.length ? real.join(' | ') : 'none'}`);
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} smoke checks passed`);
  const knownFailed = known.filter((r) => !r.ok);
  if (known.length) {
    console.log(
      `known-limitation impostor checks: ${known.length - knownFailed.length}/${known.length} rejected` +
        (knownFailed.length ? ` — ${knownFailed.length} accepted (see README section 5.6)` : ''),
    );
  }
  if (failed.length) process.exit(1);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
