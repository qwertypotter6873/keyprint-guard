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
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.css': 'text/css', '.json': 'application/json' };
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

async function typeWithHold(tab, text) {
  for (const key of text) {
    await tab.keyboard.down(key);
    await tab.waitForTimeout(20);
    await tab.keyboard.up(key);
    await tab.waitForTimeout(20);
  }
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
  assert.equal(await page.locator('#arena-roster-list [data-sample="1"]').count(), 0, 'English samples are excluded from Korean comparisons');
  assert.match(await page.textContent('#arena-language-notice'), /별도 검증/);
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
  assert.equal(await page.evaluate(() => window.__keyprintTestApi.state.lang), 'en');
  // A normal submission stays pending while a mock inference awaits its result.
  await page.click('#nav-guard-btn');
  await page.click('#guard-sub-enroll');
  await page.fill('#guard-enroll-name', 'Pending user');
  await page.fill('#guard-enroll-pw-target', 'SampleFlow123');
  await page.evaluate(() => {
    const e = window.__keyprintTestApi.engine;
    window.__originalEmbedAll = e.embedAll.bind(e);
    e.embedAll = () => new Promise(resolve => { window.__finishInference = () => resolve({ password: [1, ...Array(63).fill(0)], sentence: [1, ...Array(63).fill(0)], latencyMs: 0 }); });
  });
  await page.locator('#guard-enroll-input').pressSequentially('SampleFlow123', { delay: 25 });
  await page.press('#guard-enroll-input', 'Enter');
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
  await page.fill('#guard-enroll-pw-target', 'SampleFlow123');
  await page.evaluate(() => { window.__keyprintTestApi.engine.embedAll = async () => { throw new Error('simulated model failure'); }; });
  await page.locator('#guard-enroll-input').pressSequentially('SampleFlow123', { delay: 25 });
  await page.press('#guard-enroll-input', 'Enter');
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
  console.log('Functional browser flows passed: prefixed deployment and real inference, consent, workspace isolation, samples, stale inference, offline tabs, physical Delete and selection edits. No external traffic.');
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
