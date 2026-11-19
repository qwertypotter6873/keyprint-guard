/** Normal workspace and sample flows, with all external networking disabled. */
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { findChromeExecutable } from './browser_helpers.mjs';
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(process.env.KEYPRINT_WEB_DIST || path.join(web, 'dist'));
const prefix = '/grading/keyprint-guard/';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json' };
const server = http.createServer((req, res) => {
  const { pathname } = new URL(req.url, 'http://localhost');
  if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
  const relative = decodeURIComponent(pathname.slice(prefix.length));
  if (relative === 'recorder-test') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<input id="test"><script type="module">import { SmartKeystrokeRecorder } from "./recorder.js"; window.recorder = new SmartKeystrokeRecorder(document.getElementById("test"));</script>');
    return;
  }
  if (relative === 'recorder.js' || relative === 'features.js') {
    res.setHeader('Content-Type', mime['.js']);
    fs.createReadStream(path.join(web, 'src', relative)).pipe(res);
    return;
  }
  const file = path.resolve(root, relative || 'index.html');
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
  fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ executablePath: findChromeExecutable(chromium), headless: true });

const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ';
const JONG = ['', ...'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'];
const V_FIRST = { ㅘ: 'ㅗ', ㅙ: 'ㅗ', ㅚ: 'ㅗ', ㅝ: 'ㅜ', ㅞ: 'ㅜ', ㅟ: 'ㅜ', ㅢ: 'ㅡ' };
const J_FIRST = { ㄳ: 'ㄱ', ㄵ: 'ㄴ', ㄶ: 'ㄴ', ㄺ: 'ㄹ', ㄻ: 'ㄹ', ㄼ: 'ㄹ', ㄽ: 'ㄹ', ㄾ: 'ㄹ', ㄿ: 'ㄹ', ㅀ: 'ㄹ', ㅄ: 'ㅂ' };
const syl = (c, v, j = 0) => String.fromCodePoint(0xac00 + c * 588 + v * 28 + j);

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

async function typeSoftSentence(tab, selector, text) {
  let committed = '';
  for (const ch of text) {
    const states = compositionStates(ch);
    for (const st of states) {
      const value = committed + st;
      const inputType = /^[ㄱ-ㆎ가-힣]$/.test(st) ? 'insertCompositionText' : 'insertText';
      await tab.evaluate(({ sel, val, type }) => {
        const el = document.querySelector(sel);
        el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Unidentified', code: '', bubbles: true }));
        el.value = val;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: type, data: val.slice(-1) }));
        el.dispatchEvent(new KeyboardEvent('keyup', { key: 'Unidentified', code: '', bubbles: true }));
      }, { sel: selector, val: value, type: inputType });
      await tab.waitForTimeout(20);
    }
    committed += ch;
  }
}

async function typeWithHold(tab, text) {
  for (const key of text) {
    await tab.keyboard.down(key);
    await tab.waitForTimeout(20);
    await tab.keyboard.up(key);
    await tab.waitForTimeout(20);
  }
}

async function assertArenaLayout(page, description) {
  const layout = await page.evaluate(() => {
    const viewport = document.documentElement;
    const controls = document.querySelectorAll(`
      .mode-tab, .arena-head button, .experience-copy, .experience-guide a,
      #arena-sample-start-btn, .arena-roster .roster-label, #arena-manage-link,
      #arena-roster-list, .arena-roster .user-chip, .arena-roster-note,
      #arena-blind-panel .section-caption, #arena-result-title, #arena-result-decision,
      #arena-result-card .result-score-box, #arena-result-criteria
    `);
    const overflow = [...controls].filter(element => {
      if (element.getClientRects().length === 0) return false;
      const { left, right } = element.getBoundingClientRect();
      return left < 0 || right > viewport.clientWidth + 1 || element.scrollWidth > element.clientWidth + 1;
    }).map(element => element.id || element.className);

    const bounds = selector => document.querySelector(selector).getBoundingClientRect();
    const adjacent = [
      ['.arena-head .sub-tabs', '.arena-head .lang-switch'],
      ['.experience-copy', '#arena-sample-start-btn'],
      ['.experience-guide', '.arena-roster'],
      ['#arena-roster-title', '#arena-manage-link'],
      ['#arena-roster-list', '#arena-manage-link'],
      ['#arena-roster-list', '#arena-roster-note'],
      ['.arena-result-subject', '#arena-result-card .result-score-box'],
    ];
    const overlaps = adjacent.filter(([first, second]) => {
      const a = bounds(first);
      const b = bounds(second);
      return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
    });
    return {
      overflow,
      overlaps,
      pageOverflow: viewport.scrollWidth - viewport.clientWidth,
      guideSpacing: bounds('.arena-roster').top - bounds('.experience-guide').bottom,
    };
  });
  assert.deepEqual(layout.overflow, [], description);
  assert.deepEqual(layout.overlaps, [], description);
  assert.ok(layout.pageOverflow <= 1, description);
  assert.ok(layout.guideSpacing >= 12, description);
}

async function assertArenaResult(page, candidateCount, inputMode = 'keyboard') {
  const shots = await page.evaluate(() => window.__keyprintTestApi.engine.arenaProtocol().query);
  const prompt = page.locator('#arena-blind-prompt');
  const input = page.locator('#arena-blind-input');
  const card = page.locator('#arena-result-card');
  for (let i = 0; i < shots; i++) {
    const sentence = await prompt.textContent();
    await input.focus();
    if (inputMode === 'touch') {
      await typeSoftSentence(page, '#arena-blind-input', sentence);
    } else {
      await typeWithHold(page, sentence);
    }
    await input.press('Enter');
    await page.waitForFunction(previous => {
      return !document.getElementById('arena-blind-input').readOnly
        && document.getElementById('arena-blind-prompt').textContent !== previous;
    }, sentence);
    assert.equal(await card.isVisible(), i === shots - 1, 'Show a result once all query sentences are submitted');
  }

  const verdict = await page.evaluate(() => window.__keyprintTestApi.state.lastVerdict);
  const { candidates, threshold, lead, margin, accepted } = verdict;
  assert.equal(verdict.inputMode, inputMode);
  assert.equal(threshold, inputMode === 'touch' ? 0.78 : 0.92);
  assert.equal(margin, 0.12);
  assert.equal(candidates.length, candidateCount);
  const winner = candidates[0];
  const singleCandidate = candidateCount === 1;
  const status = singleCandidate
    ? accepted ? '리듬 일치' : '리듬 비교 보류'
    : accepted ? '판정 수락' : '판정 보류';
  assert.equal(await page.textContent('#arena-result-kicker'), singleCandidate ? '1:1 비교 대상' : '예상 작성자 1위');
  assert.equal(await page.textContent('#arena-result-title'), winner.name);
  assert.equal(await page.textContent('#arena-result-status'), status);
  const reason = await page.textContent('#arena-result-reason');
  if (accepted) assert.match(reason, /기준을 충족/);
  else if (verdict.rejectedReason === 'margin_hold') assert.match(reason, /1·2위 점수 차가 기준보다 작/);
  else assert.match(reason, /리듬 점수가 기준보다 낮/);
  assert.equal(await page.textContent('#arena-result-card .score-caption'), '리듬 점수 (0~100)');
  assert.equal(await page.textContent('#arena-result-score'), `${(winner.similarity * 100).toFixed(1)}점`);
  const minimum = `최소 리듬 점수 ${(threshold * 100).toFixed(1)}점`;
  const comparison = `1·2위 점수 차 ${(lead * 100).toFixed(2)}점 (최소 ${(margin * 100).toFixed(2)}점)`;
  assert.equal(await page.textContent('#arena-result-criteria'), singleCandidate ? minimum : `${comparison} · ${minimum}`);
  assert.equal(await page.locator('#arena-ranking-bars').isVisible(), !singleCandidate);
  if (!singleCandidate) assert.equal(await page.locator('#arena-ranking-bars .rank-name').first().textContent(), `1. ${winner.name}`);
  await assertArenaLayout(page, `Comparison result for ${candidateCount} participants`);
  return verdict;
}

try {
  const ctx = await browser.newContext();
  const host = `http://127.0.0.1:${server.address().port}`;
  const origin = host + prefix;
  const outside = [];
  await ctx.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.origin === host && url.pathname.startsWith(prefix)) return route.continue();
    outside.push(url.href);
    return route.abort();
  });
  await ctx.routeWebSocket('**/*', route => route.close());
  await ctx.addInitScript(() => { window.__connectionAttempts = 0; window.WebSocket = function () { window.__connectionAttempts++; throw new Error('Mocked offline transport'); }; });
  const page = await ctx.newPage();
  const errors = [];
  const responses = new Map();
  page.on('response', (response) => responses.set(new URL(response.url()).pathname, [response.status(), response.headers()['content-type']]));
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${origin}?room=INVITED`);
  await page.waitForFunction(() => window.__keyprintTestApi?.state.ready);
  assert.deepEqual(responses.get(`${prefix}ort/ort-wasm-simd-threaded.mjs`), [200, 'text/javascript']);
  assert.deepEqual(responses.get(`${prefix}ort/ort-wasm-simd-threaded.wasm`), [200, 'application/wasm']);
  assert.deepEqual(responses.get(`${prefix}models/model_config.json`), [200, 'application/json']);
  assert.deepEqual(responses.get(`${prefix}models/keyprint_net.onnx`), [200, 'application/octet-stream']);
  assert.deepEqual(responses.get(`${prefix}models/keyprint_net_password.onnx`), [200, 'application/octet-stream']);
  const dimensions = await page.evaluate(async () => {
    const { engine, state } = window.__keyprintTestApi;
    const result = await engine.embedAll(state.viz.capture.stroke);
    return [result.sentence, result.password, result.passwordAux].map((vector) => vector.length);
  });
  assert.deepEqual(dimensions, [64, 64, 64]);
  assert.deepEqual(await page.evaluate(() => ({ shared: window.__keyprintTestApi.roomSync.shared, calls: window.__connectionAttempts, lang: window.__keyprintTestApi.state.lang })), { shared: false, calls: 0, lang: 'en' });
  await page.locator('#change-room-form button[type="submit"]').click();
  assert.equal(await page.evaluate(() => window.__connectionAttempts), 0, 'An unchecked invitation does not connect');
  await page.click('#sample-start-btn');
  assert.equal(await page.locator('#room-dialog[open]').count(), 0);
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.state.arenaSub), 'blind');
  assert.equal(await page.locator('#arena-roster-list [data-sample="1"]').count(), 4);
  await page.click('#lang-ko-btn');
  assert.equal(await page.locator('#arena-roster-list [data-sample="1"]').count(), 4, 'Korean samples are available in Korean mode');
  assert.equal(await page.textContent('#arena-sample-start-btn'), '한국어 샘플로 체험');
  assert.match(await page.textContent('#arena-language-notice'), /공개 스마트폰 타건.*PC 한글 입력은 체험용/);

  const studyPage = await ctx.newPage();
  await studyPage.setViewportSize({ width: 320, height: 780 });
  const studyUrl = await page.locator('#arena-korean-study-link').getAttribute('href');
  const studyResponse = await studyPage.goto(new URL(studyUrl, origin).href);
  assert.equal(studyResponse.status(), 200);
  assert.equal(await studyPage.locator('h1').innerText(), '한국어 타건 평가');
  assert.equal(await studyPage.locator('#aihub tbody tr').count(), 5);
  assert.equal(await studyPage.locator('#fact tbody tr').count(), 5);
  assert.equal(await studyPage.locator('#acceptance tbody tr').count(), 3);
  const chart = studyPage.locator('#acceptance img');
  await chart.scrollIntoViewIfNeeded();
  await chart.evaluate(image => image.decode());
  assert.match(await studyPage.textContent('#decision'), /예상 작성자 1위.*판정 수락.*판정 보류/s);
  assert.equal(await studyPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await studyPage.getByRole('link', { name: '한국어 타자연습' }).click();
  await studyPage.waitForFunction(() => window.__keyprintTestApi?.state.ready);
  assert.equal(await studyPage.locator('#lang-ko-btn').getAttribute('aria-pressed'), 'true');
  await studyPage.close();

  // Use one local Korean-only profile alongside the four bilingual (KO + EN) samples from the demo.
  await page.evaluate(() => {
    const centroid = [1, ...Array(63).fill(0)];
    window.__keyprintTestApi.roomSync.upsertProfile({
      id: 'responsive_korean_fixture', name: '한글'.repeat(12), isSample: false, color: '#e11d48',
      arena: { ko: { centroid, exemplars: [centroid], count: 5, inputMode: 'keyboard' } },
    });
  });
  const viewport = page.viewportSize();
  for (const width of [320, 375, 768, 1280]) {
    await page.setViewportSize({ ...viewport, width });
    for (const lang of ['ko', 'en']) {
      await page.click(`#lang-${lang}-btn`);
      const korean = lang === 'ko';
      assert.equal(await page.getAttribute(`#lang-${lang}-btn`, 'aria-pressed'), 'true');
      assert.equal(await page.locator('#arena-korean-study-link').isVisible(), korean);
      assert.equal(await page.locator('#arena-roster-list .user-chip').count(), korean ? 5 : 4);
      if (korean) {
        assert.equal(await page.locator('#arena-roster-note').isVisible(), false);
      } else {
        assert.match(await page.textContent('#arena-roster-note'), /1명.*제외/);
      }
      await assertArenaLayout(page, `${lang} comparison at ${width}px`);
      await page.click('#arena-sub-enroll');
      await assertArenaLayout(page, `${lang} enrollment at ${width}px`);
      await page.click('#arena-sub-blind');
    }
    await page.click('#nav-principles-btn');
    const details = page.locator('#view-principles .viz-details').first();
    if (!(await details.evaluate(element => element.open))) await details.locator('summary').click();
    const encoders = details.locator('.encoder-details .detail-col');
    assert.equal(await encoders.count(), 2);
    assert.match(await encoders.nth(0).textContent(), /17채널 KeyPrintNet/);
    assert.match(await encoders.nth(1).textContent(), /9채널 타이밍 인코더.*듀얼 점수 보정/s);
    const [sentence, password] = await encoders.evaluateAll(blocks => blocks.map(block => {
      const { top, bottom, left, right } = block.getBoundingClientRect();
      return { top, bottom, left, right, overflow: block.scrollWidth - block.clientWidth };
    }));
    assert.ok(sentence.overflow <= 1 && password.overflow <= 1, `Encoder text fits at ${width}px`);
    assert.ok(width <= 680 ? sentence.bottom < password.top : sentence.right < password.left, `Encoder blocks fit at ${width}px`);
    await page.click('#nav-arena-btn');
    await page.click('#arena-manage-link');
    const dialog = page.locator('#room-dialog[open]');
    assert.equal(await dialog.count(), 1);
    const profileRows = await dialog.locator('.profile-row').evaluateAll(rows => rows.map(row => {
      const avatar = row.querySelector('.chip-avatar').getBoundingClientRect();
      const details = row.querySelector('.profile-row-left');
      return { width: avatar.width, height: avatar.height, overflow: details.scrollWidth - details.clientWidth };
    }));
    assert.equal(profileRows.length, 5);
    for (const avatar of profileRows) {
      assert.ok(avatar.width > 0 && Math.abs(avatar.width - avatar.height) <= 0.25,
        `Profile avatar stays circular at ${width}px (got ${avatar.width} × ${avatar.height}px)`);
      assert.ok(avatar.overflow <= 1, `Profile details fit at ${width}px`);
    }
    await page.click('#close-room-btn');
  }
  await page.setViewportSize({ ...viewport, width: 320 });
  await page.click('#lang-ko-btn');
  assert.equal(await page.getAttribute('#arena-korean-study-link', 'href'), './korean-study.html');
  await page.evaluate(() => window.__keyprintTestApi.roomSync.deleteProfile('responsive_korean_fixture'));
  await page.setViewportSize(viewport);

  // Benign in-progress fixture, used to check workspace reset without typing a long sequence.
  await page.evaluate(() => {
    const s = window.__keyprintTestApi.state;
    s.arenaEnrollBuffer.items = [{ example: true }];
    s.arenaBlindBuffer = [{ example: true }];
    s.pwEnrollBuffer.items = [{ example: true }];
    document.getElementById('guard-enroll-input').value = 'temporary demo';
  });
  await page.click('#open-room-btn');
  await page.check('#share-room-consent');
  await page.locator('#change-room-form button[type="submit"]').click();
  assert.deepEqual(await page.evaluate(() => { const a = window.__keyprintTestApi; return { shared: a.roomSync.shared, profiles: a.state.profiles.length, progress: [a.state.arenaEnrollBuffer.items.length, a.state.arenaBlindBuffer.length, a.state.pwEnrollBuffer.items.length], input: document.getElementById('guard-enroll-input').value }; }), { shared: true, profiles: 0, progress: [0, 0, 0], input: '' });
  await page.click('#use-local-btn');
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.state.profiles.length), 4, 'Local profiles survive entering and leaving a separate shared workspace');
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.roomSync.shared), false);
  await page.click('#close-room-btn');
  await page.click('#arena-sample-start-btn');
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.state.lang), 'ko');
  await page.setViewportSize({ ...viewport, width: 320 });
  await assertArenaResult(page, 4, 'touch');
  await page.click('#lang-en-btn');
  await assertArenaResult(page, 4, 'keyboard');
  await page.evaluate(() => window.__keyprintTestApi.roomSync.removeSamples());

  // Fix only the query embeddings to cover every gate with production ranking and real input submissions.
  const writer = { id: 'arena_writer_fixture', name: 'LongArenaCandidate123456', vector: [1, 0] };
  const other = { id: 'arena_other_fixture', name: '다른 후보', vector: [-1, 0] };
  const similar = { ...other, vector: [0.99, Math.sqrt(1 - 0.99 ** 2)] };
  const cases = [
    { profiles: [other, writer], query: writer.vector, winner: writer, rejectedReason: null, inputMode: 'keyboard' },
    { profiles: [writer, similar], query: [Math.SQRT1_2, Math.SQRT1_2], winner: similar, rejectedReason: 'margin_hold', inputMode: 'touch' },
    { profiles: [other, writer], query: [0, 1], winner: writer, rejectedReason: 'below_threshold', inputMode: 'keyboard' },
    { profiles: [writer], query: writer.vector, winner: writer, rejectedReason: null, inputMode: 'keyboard' },
    { profiles: [writer], query: [0, 1], winner: writer, rejectedReason: 'below_threshold', inputMode: 'keyboard' },
  ];
  for (const { profiles, query, winner, rejectedReason, inputMode } of cases) {
    await page.evaluate(({ profiles, query, inputMode, otherId }) => {
      const { roomSync, engine } = window.__keyprintTestApi;
      const toEmbedding = vector => [...vector, ...Array(62).fill(0)];
      if (profiles.length === 1) roomSync.deleteProfile(otherId);
      for (const { id, name, vector } of profiles) {
        const centroid = toEmbedding(vector);
        roomSync.upsertProfile({
          id, name, isSample: false, color: '#2563eb',
          arena: { en: { centroid, exemplars: [centroid], count: 5, inputMode } },
        });
      }
      engine.embedStroke = async () => ({ embedding: toEmbedding(query), latencyMs: 0 });
    }, { profiles, query, inputMode, otherId: other.id });
    await page.click('#arena-retry-btn');
    const verdict = await assertArenaResult(page, profiles.length, inputMode);
    assert.equal(verdict.candidates[0].name, winner.name);
    assert.equal(verdict.accepted, rejectedReason === null);
    assert.equal(verdict.rejectedReason, rejectedReason);
    if (rejectedReason === 'margin_hold') {
      await page.click('#arena-inspect-viz-btn');
      const snapshot = await page.textContent('#viz-verdict-status');
      assert.ok(snapshot.includes(`예상 작성자 1위 ${winner.name} · 판정 보류`));
      await page.click('#nav-arena-btn');
    }
  }

  await page.evaluate(writerId => {
    const { roomSync, engine } = window.__keyprintTestApi;
    roomSync.deleteProfile(writerId);
    delete engine.embedStroke;
  }, writer.id);
  await page.locator('#arena-blind-input').press('Enter');
  assert.match(await page.textContent('#arena-blind-feedback'), /현재 언어로 등록된 참가자가 없습니다/);
  assert.equal(await page.locator('#arena-result-card').isVisible(), false);
  await page.click('#arena-sample-start-btn');
  await page.setViewportSize(viewport);

  // A normal submission stays pending while a mock inference awaits its result.
  await page.click('#nav-guard-btn');
  await page.evaluate(() => {
    document.getElementById('guard-sub-enroll').click();
    document.getElementById('guard-enroll-pw-target').focus();
  });
  await page.waitForTimeout(60);
  assert.equal(await page.evaluate(() => document.activeElement.id), 'guard-enroll-pw-target',
    'Changing panels must not steal focus from a field the user selected');
  await page.fill('#guard-enroll-name', 'Pending user');
  const enrollmentText = 'sampleflowabc';
  await page.fill('#guard-enroll-pw-target', enrollmentText);
  await page.evaluate(() => {
    const e = window.__keyprintTestApi.engine;
    window.__originalEmbedAll = e.embedAll.bind(e);
    e.embedAll = () => new Promise(resolve => { window.__finishInference = () => resolve({ password: [1, ...Array(63).fill(0)], sentence: [1, ...Array(63).fill(0)], latencyMs: 0 }); });
  });
  const enrollInput = page.locator('#guard-enroll-input');
  await enrollInput.focus();
  await typeWithHold(page, enrollmentText);
  assert.equal(await enrollInput.inputValue(), enrollmentText);
  await enrollInput.press('Enter');
  await page.waitForFunction(() => typeof window.__finishInference === 'function');
  assert.equal(await page.getAttribute('#guard-enroll-input', 'readonly'), '');
  await page.click('#open-room-btn');
  await page.check('#share-room-consent');
  await page.locator('#change-room-form button[type="submit"]').click();
  await page.evaluate(() => window.__finishInference());
  await page.waitForFunction(() => !document.getElementById('guard-enroll-input').readOnly);
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.state.pwEnrollBuffer.items.length), 0, 'An in-flight result does not return to the new workspace');
  await page.click('#use-local-btn');
  await page.click('#close-room-btn');
  await page.fill('#guard-enroll-name', 'Error handling');
  await page.fill('#guard-enroll-pw-target', enrollmentText);
  await page.evaluate(() => { window.__keyprintTestApi.engine.embedAll = async () => { throw new Error('simulated model failure'); }; });
  await enrollInput.focus();
  await typeWithHold(page, enrollmentText);
  await enrollInput.press('Enter');
  await page.waitForFunction(() => document.getElementById('guard-enroll-feedback').textContent.includes('다시 입력해 주세요'));
  assert.equal(await page.getAttribute('#guard-enroll-input', 'readonly'), null);
  await page.evaluate(() => { window.__keyprintTestApi.engine.embedAll = window.__originalEmbedAll; });

  const verifyInput = page.locator('#guard-verify-input');
  for (const mutation of ['update', 'delete']) {
    await page.evaluate(() => {
      const { engine, roomSync, state } = window.__keyprintTestApi;
      const centroid = [1, ...Array(63).fill(0)];
      const embedding = { centroid, exemplars: [centroid] };
      roomSync.upsertProfile({
        id: 'inference_fixture', name: 'Inference fixture', isSample: false, color: '#2563eb', arena: {},
        password: { ...embedding, enc17: embedding, target: 'sampleflowab', count: 1, inputMode: 'keyboard' },
      });
      state.selectedGuardId = 'inference_fixture';
      window.__finishVerification = null;
      window.__scoreCalls = 0;
      window.__originalScore = engine.scoreTemplate.bind(engine);
      engine.embedAll = () => new Promise((resolve) => {
        window.__finishVerification = () => resolve({ password: centroid, passwordAux: centroid, sentence: centroid, latencyMs: 0 });
      });
      engine.scoreTemplate = () => {
        window.__scoreCalls++;
        return { accepted: true, adapt: true, similarity: 1, threshold: 0.5, method: 'fixture' };
      };
      document.getElementById('guard-sub-verify').click();
    });
    await verifyInput.focus();
    await typeWithHold(page, 'sampleflowab');
    await verifyInput.press('Enter');
    await page.waitForFunction(() => typeof window.__finishVerification === 'function');
    await page.evaluate((action) => {
      const { roomSync } = window.__keyprintTestApi;
      const profile = roomSync.getProfiles().find((p) => p.id === 'inference_fixture');
      if (action === 'delete') roomSync.deleteProfile(profile.id);
      else roomSync.upsertProfile({ ...profile, password: { ...profile.password, count: 2 } });
      window.__finishVerification();
    }, mutation);
    await page.waitForFunction(() => !document.getElementById('guard-verify-input').readOnly);
    const result = await page.evaluate(() => {
      const { roomSync, state, engine } = window.__keyprintTestApi;
      engine.embedAll = window.__originalEmbedAll;
      engine.scoreTemplate = window.__originalScore;
      return {
        scoreCalls: window.__scoreCalls,
        sampleCount: roomSync.getProfiles().find((p) => p.id === 'inference_fixture')?.password.count ?? null,
        verdict: state.lastVerdict,
        feedback: document.getElementById('guard-verify-feedback').textContent,
      };
    });
    assert.equal(result.scoreCalls, 0, 'A pending result cannot adapt a profile after it changes');
    assert.equal(result.sampleCount, mutation === 'delete' ? null : 2);
    assert.equal(result.verdict, null);
    assert.match(result.feedback, /프로필이 변경되었습니다/);
  }
  assert.deepEqual(errors, []);

  async function joinOfflineRoom(tab) {
    await tab.goto(origin);
    await tab.waitForFunction(() => window.__keyprintTestApi?.state.ready);
    if (await tab.locator('#room-dialog[open]').count() === 0) await tab.click('#open-room-btn');
    await tab.fill('#room-code-input', 'OFFLINE-TABS');
    await tab.check('#share-room-consent');
    await tab.locator('#change-room-form button[type="submit"]').click();
  }
  const idle = await ctx.newPage();
  await Promise.all([joinOfflineRoom(page), joinOfflineRoom(idle)]);
  await page.evaluate(() => {
    const centroid = [1, ...Array(63).fill(0)];
    window.__keyprintTestApi.roomSync.upsertProfile({
      id: 'offline_tab', name: 'Offline tab', isSample: false, color: '#2563eb',
      arena: { en: { centroid, exemplars: [centroid], count: 1, inputMode: 'keyboard' } },
    });
  });
  await idle.waitForFunction(() => window.__keyprintTestApi.roomSync.pendingUpdates.has('offline_tab'));
  await Promise.all([page.close(), idle.close()]);
  const reopened = await ctx.newPage();
  await joinOfflineRoom(reopened);
  assert.equal(await reopened.evaluate(() => window.__keyprintTestApi.roomSync.pendingUpdates.has('offline_tab')), true);

  await reopened.goto(`${origin}recorder-test`);
  await reopened.waitForFunction(() => window.recorder);
  const input = reopened.locator('#test');
  await input.focus();
  await typeWithHold(reopened, 'aabcd');
  const readRecorder = () => reopened.evaluate(() => {
    const { stroke, typedText, backspaceCount } = window.recorder.extractCleanStroke();
    const timings = window.recorder.stack.map(({ char, downTime, upTime }) => ({ char, downTime, upTime }));
    return { stroke, typedText, backspaceCount, timings };
  });
  const original = await readRecorder();
  assert.equal(original.stroke.dds.length, 4);
  assert.ok(original.stroke.holdValid.every(Boolean));
  await input.press('Delete');
  assert.deepEqual(await readRecorder(), original, 'Forward Delete at the end leaves timing and counts intact');

  await input.evaluate((el) => el.setSelectionRange(0, 0));
  await input.press('Delete');
  const deletedFirst = await readRecorder();
  assert.equal(deletedFirst.typedText, 'abcd');
  assert.equal(deletedFirst.backspaceCount, 1);
  assert.deepEqual(deletedFirst.timings, original.timings.slice(1), 'Forward Delete removes the first hit even with repeated characters');

  await input.evaluate((el) => el.setSelectionRange(1, 3));
  await typeWithHold(reopened, 'x');
  const replaced = await readRecorder();
  assert.equal(replaced.typedText, 'axd');
  assert.deepEqual(replaced.timings.map((stroke) => stroke.char), ['a', 'x', 'd']);
  assert.deepEqual(replaced.timings.at(-1), original.timings.at(-1), 'Selection replacement preserves untouched suffix timing');
  assert.deepEqual(replaced.stroke.dds, [], 'Selection replacement drops transitions across the edit');
  await input.evaluate((el) => el.setSelectionRange(el.value.length, el.value.length));
  await input.press('Backspace');
  const deletedLast = await readRecorder();
  assert.equal(deletedLast.typedText, 'ax');
  assert.equal(deletedLast.backspaceCount, 2);
  assert.deepEqual(deletedLast.timings, replaced.timings.slice(0, -1));
  assert.deepEqual(outside, [], 'Asset and demo requests stay within the local deployment prefix');
  console.log('Offline browser checks passed for prefixed deployment, real inference, responsive arena and encoder layouts, Arena ranking and verdicts, consent, workspace isolation, samples, stale inference, offline tabs, physical Delete, and selection edits.');
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
