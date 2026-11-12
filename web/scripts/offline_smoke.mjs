/**
 * Offline browser smoke test of the production build (web/dist).
 *
 *   npm run build && node scripts/offline_smoke.mjs
 *
 * Every non-localhost request is aborted and every WebSocket (MQTT room sync) is mocked
 * without a server, so no profile ever leaves the machine. Uses generic typist names only.
 *
 * Desktop (physical keyboard events):  enroll 2 typists x 3 sentences (with typos + Backspace),
 *   blind 1:N identification, password enroll x4, genuine accept / impostor reject, samples.
 * Mobile emulation (soft keyboard: keydown key='Unidentified' code='', text via `input` events,
 *   Korean IME composition states): the path that used to fail with
 *   "추출된 연속 리듬 구간이 부족합니다" must now enroll 3 sentences, identify and enroll a password.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

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
    plan.push({ t, type: 'keydown', key: ch, code });
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
          el.dispatchEvent(new KeyboardEvent(ev.type, { key: ev.key, code: ev.code, bubbles: true, cancelable: true }));
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

// ---------------------------------------------------------------------------------------
const results = [];
function expect(cond, msg) {
  results.push({ ok: Boolean(cond), msg });
  console.log(`${cond ? '  ✓' : '  ✗'} ${msg}`);
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
  for (let s = 0; s < 3; s++) {
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
  return { feedbacks, profile };
}

async function blindIdentify(page, planFor) {
  await page.click('#arena-sub-blind');
  const sentence = await api(page, () => window.__keyprintTestApi.state.arenaBlindSentence);
  await runPlan(page, '#arena-blind-input', planFor(sentence));
  await page.waitForSelector('#arena-result-card:not([hidden])', { timeout: 5000 });
  await page.waitForTimeout(150);
  return {
    title: (await page.textContent('#arena-result-title'))?.trim(),
    meta: (await page.textContent('#arena-result-meta'))?.trim(),
    feedback: (await page.textContent('#arena-blind-feedback'))?.trim(),
  };
}

async function enrollPassword(page, name, pw, planFor) {
  await page.click('#nav-guard-btn');
  await page.click('#guard-sub-enroll');
  await page.fill('#guard-enroll-name', name);
  await page.fill('#guard-enroll-pw-target', pw);
  const feedbacks = [];
  for (let r = 0; r < 4; r++) {
    await runPlan(page, '#guard-enroll-input', planFor(pw, r));
    await page.waitForTimeout(200);
    feedbacks.push((await page.textContent('#guard-enroll-feedback'))?.trim());
  }
  return feedbacks;
}

async function verifyPassword(page, pw, plan) {
  await runPlan(page, '#guard-verify-input', plan);
  await page.waitForSelector('#guard-result-card:not([hidden])', { timeout: 5000 });
  await page.waitForTimeout(150);
  return {
    pass: await page.evaluate(() => document.getElementById('guard-result-card').classList.contains('is-pass')),
    title: (await page.textContent('#guard-result-title'))?.trim(),
    score: (await page.textContent('#guard-result-score'))?.trim(),
    meta: (await page.textContent('#guard-result-meta'))?.trim(),
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
      expect(a.profile && a.profile.count === 3 && a.profile.inputMode === 'keyboard' && a.profile.hasBehavior, `Typist A enrolled: ${JSON.stringify(a.profile)}`);
      const b = await enrollArena(page, 'Typist B', 'en', (s, i) => keyboardPlan(s, SLOW, { typo: i === 0 }));
      expect(b.profile && b.profile.count === 3, `Typist B enrolled: ${JSON.stringify(b.profile)}`);

      const id = await blindIdentify(page, (s) => keyboardPlan(s, FAST, { typo: true }));
      expect(id.title?.includes('Typist A'), `blind 1:N (keyboard) → "${id.title}" | ${id.meta}`);
      if (SHOT_DIR) await page.screenshot({ path: path.join(SHOT_DIR, 'smoke_desktop_identify.png') });

      const pwFb = await enrollPassword(page, 'Typist A', 'gachon2026', (pw, r) => keyboardPlan(pw, FAST, { typo: r === 1 }));
      expect(!pwFb.some((f) => /부족|불일치|일치하지/.test(f || '')), `password enroll x4 (keyboard): ${pwFb.at(-1) || 'ok'}`);
      const genuine = await verifyPassword(page, 'gachon2026', keyboardPlan('gachon2026', FAST));
      expect(genuine.pass, `genuine password login → ${genuine.title} ${genuine.score} | ${genuine.meta}`);
      const impostor = await verifyPassword(page, 'gachon2026', keyboardPlan('gachon2026', SLOW));
      expect(!impostor.pass, `impostor with leaked password → ${impostor.title} ${impostor.score}`);

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
      expect(m.profile && m.profile.count === 3 && m.profile.inputMode === 'touch', `Korean touch profile enrolled: ${JSON.stringify(m.profile)}`);
      const mid = await blindIdentify(page, (s) => touchPlan(s, THUMB));
      expect(Boolean(mid.title) && !/부족/.test(mid.feedback || ''), `blind 1:N (touch) → "${mid.title}" | ${mid.meta}`);
      if (SHOT_DIR) await page.screenshot({ path: path.join(SHOT_DIR, 'smoke_mobile_identify.png') });

      const pwFb = await enrollPassword(page, '모바일 사용자', 'gachon2026', (pw) => touchPlan(pw, THUMB));
      expect(!pwFb.some((f) => /부족|불일치|일치하지/.test(f || '')), `password enroll x4 (touch): ${pwFb.at(-1) || 'ok'}`);
      const mv = await verifyPassword(page, 'gachon2026', touchPlan('gachon2026', THUMB));
      expect(Boolean(mv.title), `touch password verify → ${mv.title} ${mv.score} | ${mv.meta}`);

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
  if (failed.length) process.exit(1);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
