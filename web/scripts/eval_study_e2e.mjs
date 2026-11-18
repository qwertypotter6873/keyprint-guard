/**
 * End-to-end Playwright browser verification for the Cross-Device & Smartphone Evaluation Logger
 * (`?eval=1`), 4-device counterbalanced protocol, Hangul 2-beolsik layout mismatch warning,
 * template adaptation lock (`lockAdaptation`), custom-password privacy redaction vs 24 public
 * experiment presets, side-by-side string cohort calibration UI, `DeviceMotion` sensor capture,
 * and full `.jsonl` export/merge round-trip into `src/analyze_field_study.py`.
 */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { findChromeExecutable } from './browser_helpers.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, '..');
const repoRoot = path.resolve(web, '..');
const root = path.resolve(process.env.KEYPRINT_WEB_DIST || path.join(web, 'dist'));
const prefix = '/grading/keyprint-guard/';

const mime = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
};

const server = http.createServer((req, res) => {
  const { pathname } = new URL(req.url, 'http://localhost');
  if (!pathname.startsWith(prefix)) {
    res.writeHead(404).end();
    return;
  }
  const relative = decodeURIComponent(pathname.slice(prefix.length));
  const file = path.resolve(root, relative || 'index.html');
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
  fs.createReadStream(file).pipe(res);
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
const baseUrl = `http://127.0.0.1:${port}${prefix}`;

const browser = await chromium.launch({
  executablePath: findChromeExecutable(chromium),
  headless: true,
});

async function typeWithHold(page, text, holdMs = 22, gapMs = 18) {
  for (const ch of text) {
    const isUpper = ch >= 'A' && ch <= 'Z';
    if (isUpper) await page.keyboard.down('ShiftLeft');
    await page.keyboard.down(ch);
    await page.waitForTimeout(holdMs);
    await page.keyboard.up(ch);
    if (isUpper) await page.keyboard.up('ShiftLeft');
    await page.waitForTimeout(gapMs);
  }
}

try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  // 1) Default load (without ?eval=1): evaluation panel stays hidden, zero visual clutter
  await page.goto(baseUrl);
  await page.waitForFunction(() => window.__keyprintTestApi?.engine?.session !== null);
  assert.equal(await page.locator('#eval-study-panel').isHidden(), true, 'Eval panel hidden by default');
  assert.equal(await page.locator('#toggle-eval-btn').getAttribute('aria-expanded'), 'false');

  // Verify companion calibration bundle loaded cleanly in KeyPrintEngine
  const hasCalibStudy = await page.evaluate(() => Boolean(window.__keyprintTestApi.engine.calibrationStudy?.models?.touch));
  assert.equal(hasCalibStudy, true, 'KeyPrintEngine loaded string_cohort_calibration.json');

  // 2) ?eval=1 auto-opens panel, enables logging, and renders Williams 4-device counterbalanced order
  await page.goto(`${baseUrl}?eval=1`);
  await page.waitForFunction(() => window.__keyprintTestApi?.engine?.session !== null);
  assert.equal(await page.locator('#eval-study-panel').isVisible(), true, '?eval=1 opens #eval-study-panel');
  assert.equal(await page.locator('#eval-enabled-chk').isChecked(), true, '?eval=1 auto-enables logging');

  const bannerP01 = await page.locator('#eval-order-banner').textContent();
  await page.fill('#eval-participant-id', 'P02');
  const bannerP02 = await page.locator('#eval-order-banner').textContent();
  assert.notEqual(bannerP01, bannerP02, 'Counterbalanced 4-device order updates dynamically per participantId');

  // 3) Hangul non-2-beolsik mismatch warning when Korean Arena + 천지인 selected
  await page.click('#lang-ko-btn');
  await page.selectOption('#eval-hangul-layout', 'mobile_cheonjiin');
  assert.equal(await page.locator('#eval-layout-warning').isVisible(), true, '천지인 triggers 2-beolsik mismatch warning in Korean mode');
  await page.selectOption('#eval-hangul-layout', 'pc_2beolsik');
  assert.equal(await page.locator('#eval-layout-warning').isHidden(), true, 'pc_2beolsik hides mismatch warning');

  // Switch back to English Arena and opt in to raw timing & diagnostics
  await page.click('#lang-en-btn');
  await page.check('#eval-raw-consent-chk');
  await page.selectOption('#eval-keyboard-type', 'mechanical_linear');
  await page.selectOption('#eval-session-slot', 'day1_slot1');

  // 4) Arena Enrollment (5 sentences on mechanical_linear)
  await page.fill('#arena-name-input', '실측참가자A');
  const enrollShots = await page.evaluate(() => window.__keyprintTestApi.engine.arenaProtocol().enroll);
  for (let s = 0; s < enrollShots; s++) {
    const targetSentence = (await page.locator('#arena-enroll-prompt').textContent()).trim();
    await page.locator('#arena-enroll-input').focus();
    await typeWithHold(page, targetSentence, 24, 18);
    await page.locator('#arena-enroll-input').press('Enter');
    if (s < enrollShots - 1) {
      await page.waitForFunction(
        (prev) => document.getElementById('arena-enroll-prompt').textContent.trim() !== prev,
        targetSentence
      );
    } else {
      await page.waitForFunction(() =>
        document.getElementById('arena-enroll-feedback').textContent.includes('등록되었습니다')
      );
      await page.click('#arena-sub-blind');
    }
  }

  const enrolledProfileId = await page.evaluate(() => {
    const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === '실측참가자A');
    return p?.id || null;
  });
  assert.ok(enrolledProfileId, 'Enrolled Arena profile created');

  // 5) Cross-device Arena Query (pantograph_external, longer hold 52ms) + simulated DeviceMotion samples
  await page.selectOption('#eval-keyboard-type', 'pantograph_external');
  await page.selectOption('#eval-session-slot', 'day1_slot2');
  await page.selectOption('#eval-gt-role', `enrolled:${enrolledProfileId}`);

  const queryShots = await page.evaluate(() => window.__keyprintTestApi.engine.arenaProtocol().query);
  for (let q = 0; q < queryShots; q++) {
    const qSentence = (await page.locator('#arena-blind-prompt').textContent()).trim();
    await page.locator('#arena-blind-input').focus();
    // Inject DeviceMotion samples continuously around typing timestamps
    await page.evaluate(() => {
      const m = window.__keyprintTestApi.evalStudy.motion;
      const now = performance.now();
      for (let k = -4; k < 180; k++) {
        m.ingestSample({
          tMs: now + k * 16.67,
          ax: 0.03,
          ay: 0.02,
          az: k % 9 === 0 ? 0.48 : 0.02,
          agx: 0.04,
          agy: 0.08,
          agz: -9.8,
          rAlpha: 0.2,
          rBeta: 0.1,
          rGamma: 0.1,
        });
      }
    });
    await typeWithHold(page, qSentence, 52, 18);
    await page.locator('#arena-blind-input').press('Enter');
    await page.waitForFunction(
      (prev) =>
        !document.getElementById('arena-blind-input').readOnly &&
        document.getElementById('arena-blind-prompt').textContent.trim() !== prev,
      qSentence
    );
  }
  assert.equal(await page.locator('#arena-result-card').isVisible(), true, 'Arena query verdict card displayed');

  // 6) Guard Password Enrollment + Verification:
  //    (a) Public experiment preset EXP12_REP_A ('CoffeeTree88') + lockAdaptation check + side-by-side calib display
  await page.click('#nav-guard-btn');
  await page.click('#guard-sub-enroll');

  const presetCount = await page.locator('#guard-exp-preset-select option').count();
  assert.equal(presetCount, 25, '1 default option + 24 public experiment presets');
  await page.selectOption('#guard-exp-preset-select', 'CoffeeTree88');
  assert.equal(await page.locator('#guard-enroll-pw-target').inputValue(), 'CoffeeTree88', 'Preset autofills CoffeeTree88');

  await page.fill('#guard-enroll-name', '실측암호계정A');
  for (let i = 0; i < 4; i++) {
    await page.locator('#guard-enroll-input').focus();
    await typeWithHold(page, 'CoffeeTree88', 28, 20);
    await page.locator('#guard-enroll-input').press('Enter');
    if (i < 3) {
      await page.waitForFunction((step) => document.getElementById('guard-step-text').textContent.includes(`${step} / 4`), i + 2);
    } else {
      await page.waitForFunction(() => document.getElementById('guard-verify-panel').hidden === false);
    }
  }

  const guardAccId = await page.evaluate(() => {
    const p = window.__keyprintTestApi.state.profiles.find((x) => x.name === '실측암호계정A');
    return p?.id || null;
  });
  assert.ok(guardAccId, 'Guard password profile created');
  await page.selectOption('#eval-gt-role', `enrolled:${guardAccId}`);

  await page.selectOption('#eval-string-calib-mode', 'force_on');
  await page.locator('#guard-verify-input').focus();
  await typeWithHold(page, 'CoffeeTree88', 28, 20);
  await page.locator('#guard-verify-input').press('Enter');
  await page.waitForFunction(() => document.getElementById('guard-result-card').hidden === false);

  const calibCompareText = await page.locator('#guard-calib-compare').textContent();
  assert.ok(
    calibCompareText.includes('길이 기준') && calibCompareText.includes('문자열 보정') && !calibCompareText.includes('w=0.00'),
    `Side-by-side calibration line rendered with active support weight: ${calibCompareText}`
  );
  await page.selectOption('#eval-string-calib-mode', 'auto');
  const templateCountAfterLockedVerify = await page.evaluate(
    (id) => window.__keyprintTestApi.state.profiles.find((x) => x.id === id)?.password?.count,
    guardAccId
  );
  assert.equal(templateCountAfterLockedVerify, 4, 'lockAdaptation prevented template count mutation (stayed 4)');

  //    (b) Custom private password ('MyPrivateKey9876') -> must be redacted in exported JSONL!
  await page.click('#guard-sub-enroll');
  await page.selectOption('#guard-exp-preset-select', '');
  await page.fill('#guard-enroll-name', '비공개암호계정B');
  await page.fill('#guard-enroll-pw-target', 'MyPrivateKey9876');
  for (let i = 0; i < 4; i++) {
    await page.locator('#guard-enroll-input').focus();
    await typeWithHold(page, 'MyPrivateKey9876', 26, 20);
    await page.locator('#guard-enroll-input').press('Enter');
    if (i < 3) {
      await page.waitForFunction((step) => document.getElementById('guard-step-text').textContent.includes(`${step} / 4`), i + 2);
    } else {
      await page.waitForFunction(() => document.getElementById('guard-verify-panel').hidden === false);
    }
  }
  await page.selectOption('#eval-gt-role', 'outsider');
  await page.locator('#guard-verify-input').focus();
  await typeWithHold(page, 'MyPrivateKey9876', 65, 45);
  await page.locator('#guard-verify-input').press('Enter');
  await page.waitForFunction(() => document.getElementById('guard-result-card').hidden === false);

  // 7) Export JSONL, verify custom-password redaction & round-trip into browser merge + Python analyzer
  const jsonlText = await page.evaluate(() => window.__keyprintTestApi.evalStudy.toJsonl());
  assert.equal(jsonlText.includes('MyPrivateKey9876'), false, 'Custom password plaintext NEVER appears in JSONL');
  assert.equal(jsonlText.includes('CoffeeTree88'), true, 'Public allowlist password tagged in JSONL');

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kp-eval-e2e-'));
  const jsonlPath = path.join(tmpDir, 'browser_session.jsonl');
  const reportPath = path.join(tmpDir, 'field_report.json');
  fs.writeFileSync(jsonlPath, jsonlText, 'utf8');

  // Verify browser multi-file JSONL import deduplicates cleanly
  await page.setInputFiles('#eval-import-jsonl-input', jsonlPath);
  const countAfterDupImport = await page.evaluate(() => window.__keyprintTestApi.evalStudy.records.length);
  assert.equal(countAfterDupImport, 6, '3 enrollments + 1 arena query + 2 password queries = 6 deduplicated records');

  // Run standalone Python offline analyzer on the real browser-exported JSONL!
  const pyBin = fs.existsSync(path.join(repoRoot, '.venv', 'bin', 'python'))
    ? path.join(repoRoot, '.venv', 'bin', 'python')
    : 'python3';
  const stdout = execFileSync(pyBin, [path.join(repoRoot, 'src', 'analyze_field_study.py'), jsonlPath, '--out', reportPath], {
    encoding: 'utf8',
  });
  const pyReport = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  assert.equal(pyReport.meta.total_records, 6);
  assert.equal(pyReport.environment_and_errors.protocol_integrity.custom_password_redaction_ok, true);
  assert.equal(pyReport.environment_and_errors.protocol_integrity.template_lock_ok, true);
  assert.equal(pyReport.environment_and_errors.arena.cross_device.n_total, 1);
  assert.equal(pyReport.hold_calibration.rescoring.available, true);
  assert.equal(
    pyReport.hold_calibration.rescoring.arena_cross_switch.physics_switch_plus_shrink.n_cross_switch_queries,
    1
  );
  assert.equal(pyReport.touch_and_motion.device_motion_audit.n_records_with_motion >= 1, true);
  assert.equal(pyReport.password_string_calibration.overall.n_dual_queries, 2);

  fs.rmSync(tmpDir, { recursive: true, force: true });
  console.log('Eval Study E2E passed: ?eval=1 UI, Williams 4-device order, 2-beolsik guard, cross-switch hold rescoring, DeviceMotion, string calibration & privacy redaction.');
  console.log(stdout.trim());
} finally {
  await browser.close();
  server.close();
}
