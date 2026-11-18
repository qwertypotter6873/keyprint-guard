/** Normal workspace and sample flows, with all external networking disabled. */
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { findChromeExecutable } from './browser_helpers.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.css': 'text/css', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const file = path.resolve(root, '.' + decodeURIComponent(req.url.split('?')[0] === '/' ? '/index.html' : req.url.split('?')[0]));
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
  fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ executablePath: findChromeExecutable(chromium), headless: true });
try {
  const ctx = await browser.newContext();
  await ctx.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort());
  await ctx.addInitScript(() => { window.__connectionAttempts = 0; window.WebSocket = function () { window.__connectionAttempts++; throw new Error('Mocked offline transport'); }; });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/?room=INVITED`);
  await page.waitForFunction(() => window.__keyprintTestApi?.state.ready);
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
  assert.deepEqual(errors, []);
  console.log('Functional browser flows passed: invitation consent, local storage isolation, English samples, language exclusion and workspace resets. No external traffic.');
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
