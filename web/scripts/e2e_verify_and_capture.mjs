import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, '..');
const DIST_DIR = path.join(WEB_DIR, 'dist');
const SCREENSHOT_DIR = path.resolve(WEB_DIR, '..', 'assets', 'screenshots');

const MIME_TYPES = {
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

function startStaticServer(rootDir) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = decodeURIComponent((req.url || '/').split('?')[0]);
      const relPath = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
      const fullPath = path.join(rootDir, relPath);

      if (!fullPath.startsWith(rootDir) || !fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const ext = path.extname(fullPath).toLowerCase();
      res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
      fs.createReadStream(fullPath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, url: `http://127.0.0.1:${addr.port}` });
    });
  });
}

/**
 * Type an English string with a distinct human-like cadence (holdMs, ddMs, rollover)
 * and optionally inject real typos + Backspace repairs mid-sentence!
 */
async function typeWithBiometricCadence(page, selector, text, profile, injectBackspace = false) {
  await page.focus(selector);
  const chars = Array.from(text);
  const typoAt = injectBackspace ? Math.floor(chars.length * 0.42) : -1;

  const codeForChar = (ch) => {
    if (ch === ' ') return 'Space';
    if (ch === '.') return 'Period';
    if (/^[a-zA-Z]$/.test(ch)) return `Key${ch.toUpperCase()}`;
    if (/^[0-9]$/.test(ch)) return `Digit${ch}`;
    return 'KeyA';
  };

  for (let i = 0; i < chars.length; i++) {
    if (i === typoAt) {
      // Inject 2 mistyped characters followed by 2 Backspaces!
      for (const bad of ['x', 'z']) {
        await page.dispatchEvent(selector, 'keydown', { key: bad, code: codeForChar(bad) });
        await page.evaluate(
          ({ sel, ch }) => {
            const el = document.querySelector(sel);
            el.value += ch;
            el.dispatchEvent(new Event('input', { bubbles: true }));
          },
          { sel: selector, ch: bad }
        );
        await page.waitForTimeout(38);
        await page.dispatchEvent(selector, 'keyup', { key: bad, code: codeForChar(bad) });
        await page.waitForTimeout(55);
      }
      for (let b = 0; b < 2; b++) {
        await page.dispatchEvent(selector, 'keydown', { key: 'Backspace', code: 'Backspace' });
        await page.evaluate((sel) => {
          const el = document.querySelector(sel);
          el.value = el.value.slice(0, -1);
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }, selector);
        await page.waitForTimeout(35);
        await page.dispatchEvent(selector, 'keyup', { key: 'Backspace', code: 'Backspace' });
        await page.waitForTimeout(60);
      }
    }

    const ch = chars[i];
    const code = codeForChar(ch);
    const wave = 1.0 + profile.waveAmp * Math.sin((i / Math.max(1, chars.length)) * Math.PI * 2);
    const hold = Math.round(profile.holdMs * wave);
    const dd = Math.round(profile.ddMs * wave);

    await page.dispatchEvent(selector, 'keydown', { key: ch, code });
    await page.evaluate(
      ({ sel, c }) => {
        const el = document.querySelector(sel);
        el.value += c;
        el.dispatchEvent(new Event('input', { bubbles: true }));
      },
      { sel: selector, c: ch }
    );

    if (dd < hold) {
      // Rollover typing (UD < 0): next key pressed before releasing current key!
      await page.waitForTimeout(dd);
      setTimeout(() => {
        page.dispatchEvent(selector, 'keyup', { key: ch, code }).catch(() => {});
      }, hold - dd);
    } else {
      await page.waitForTimeout(hold);
      await page.dispatchEvent(selector, 'keyup', { key: ch, code });
      await page.waitForTimeout(Math.max(8, dd - hold));
    }
  }
  await page.waitForTimeout(60);
}

async function run() {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const { server, url } = await startStaticServer(DIST_DIR);
  const testRoom = `E2E${Math.floor(1000 + Math.random() * 8999)}`;
  const roomUrl = `${url}/?room=${testRoom}`;
  console.log(`[E2E] Serving MVP build at ${roomUrl}`);

  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  // Two distinct human cadence profiles
  const cadenceDaehyeon = { holdMs: 108, ddMs: 84, waveAmp: 0.12 }; // Fast rollover typist (UD < 0!)
  const cadenceMinsu = { holdMs: 62, ddMs: 195, waveAmp: 0.08 }; // Deliberate staccato typist (UD > 0)

  try {
    // Launch TWO simultaneous independent browser contexts (Laptop A & Laptop B)
    const contextA = await browser.newContext({ viewport: { width: 1280, height: 880 }, deviceScaleFactor: 2 });
    const contextB = await browser.newContext({ viewport: { width: 1280, height: 880 }, deviceScaleFactor: 2 });
    const pageA = await contextA.newPage();
    const pageB = await contextB.newPage();

    const errors = [];
    for (const p of [pageA, pageB]) {
      p.on('pageerror', (e) => errors.push(e.message));
      p.on('console', (msg) => {
        if (msg.type() === 'error') errors.push(msg.text());
      });
    }

    await Promise.all([
      pageA.goto(roomUrl, { waitUntil: 'networkidle' }),
      pageB.goto(roomUrl, { waitUntil: 'networkidle' }),
    ]);

    await Promise.all([
      pageA.waitForFunction(() => document.getElementById('engine-status-text')?.textContent?.includes('ONNX 준비 완료'), { timeout: 15000 }),
      pageB.waitForFunction(() => document.getElementById('engine-status-text')?.textContent?.includes('ONNX 준비 완료'), { timeout: 15000 }),
    ]);
    console.log('[E2E] Both Laptop A & Laptop B initialized ONNX WASM!');

    // Switch both to English sentences for deterministic fast cadence synthesis + test Korean display too
    await pageA.click('#lang-en-btn');
    await pageB.click('#lang-en-btn');

    // 1. Laptop A enrolls '김대현' across 3 DIFFERENT sentences WITH Backspace repairs!
    await pageA.fill('#arena-name-input', '김대현');
    const seenSentences = new Set();
    for (let s = 0; s < 3; s++) {
      const promptTxt = await pageA.evaluate(() => window.__keyprintTestApi.state.arenaEnrollSentence);
      seenSentences.add(promptTxt);
      await typeWithBiometricCadence(pageA, '#arena-enroll-input', promptTxt, cadenceDaehyeon, true);
      if (s === 1) {
        // Capture clean enrollment in-progress screenshot
        await pageA.screenshot({
          path: path.join(SCREENSHOT_DIR, '01_enrollment_template_ready.png'),
          fullPage: false,
        });
      }
      await pageA.keyboard.press('Enter');
      await pageA.waitForTimeout(160);
    }
    if (seenSentences.size !== 3) {
      throw new Error(`Expected 3 diverse sentences, got ${seenSentences.size}`);
    }
    console.log('[E2E] Laptop A enrolled 김대현 across 3 diverse sentences WITH Backspace!');

    // Verify Laptop B received 김대현's profile in real time over WSS!
    await pageB.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '김대현'),
      { timeout: 8000 }
    );
    console.log('[E2E] Real-time WSS Sync verified: Laptop B automatically received 김대현!');

    // 2. Laptop B enrolls '박민수' across 3 DIFFERENT sentences!
    await pageB.fill('#arena-name-input', '박민수');
    for (let s = 0; s < 3; s++) {
      const promptTxt = await pageB.evaluate(() => window.__keyprintTestApi.state.arenaEnrollSentence);
      await typeWithBiometricCadence(pageB, '#arena-enroll-input', promptTxt, cadenceMinsu, s === 0);
      await pageB.keyboard.press('Enter');
      await pageB.waitForTimeout(160);
    }

    // Verify Laptop A now sees BOTH 김대현 and 박민수!
    await pageA.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.length >= 2,
      { timeout: 8000 }
    );
    console.log('[E2E] Laptop A now has live synced profiles:', await pageA.evaluate(() => window.__keyprintTestApi.state.profiles.map((p) => p.name)));

    // 3. Blind Identification on a BRAND NEW sentence WITH Backspace!
    await pageA.click('#arena-sub-blind');
    const blindSentence = await pageA.evaluate(() => window.__keyprintTestApi.state.arenaBlindSentence);
    await typeWithBiometricCadence(pageA, '#arena-blind-input', blindSentence, cadenceDaehyeon, true);
    await pageA.keyboard.press('Enter');
    await pageA.waitForSelector('#arena-result-card:not([hidden])');
    await pageA.waitForTimeout(300);
    const blindTitle = await pageA.textContent('#arena-result-title');
    const blindMeta = await pageA.textContent('#arena-result-meta');
    console.log(`[E2E] Blind Identification Result: ${blindTitle} | ${blindMeta}`);

    await pageA.screenshot({
      path: path.join(SCREENSHOT_DIR, '05_typing_arena_who_is_typing.png'),
      fullPage: false,
    });

    // 4. Mode B: Password Guard — Laptop A registers password for 김대현 (with Backspace on rep 1!)
    await pageA.click('#nav-guard-btn');
    await pageA.click('#guard-sub-enroll');
    await pageA.fill('#guard-enroll-name', '김대현');
    await pageA.fill('#guard-enroll-pw-target', 'gachon2026');
    for (let r = 0; r < 4; r++) {
      await typeWithBiometricCadence(pageA, '#guard-enroll-input', 'gachon2026', cadenceDaehyeon, r === 1);
      await pageA.keyboard.press('Enter');
      await pageA.waitForTimeout(140);
    }
    console.log('[E2E] Password profile registered for 김대현!');

    // Laptop A logs in as 김대현 -> ACCESS GRANTED
    await typeWithBiometricCadence(pageA, '#guard-verify-input', 'gachon2026', cadenceDaehyeon, true);
    await pageA.keyboard.press('Enter');
    await pageA.waitForSelector('#guard-result-card.is-pass');
    await pageA.waitForTimeout(300);
    console.log('[E2E] Mode 1 Genuine Login:', await pageA.textContent('#guard-result-title'), await pageA.textContent('#guard-result-score'));
    await pageA.screenshot({
      path: path.join(SCREENSHOT_DIR, '02_genuine_verification_granted.png'),
      fullPage: false,
    });

    // Laptop B (박민수) switches to Password Guard, sees 김대현's synced account, and types exact password 'gachon2026' -> BLOCKED!
    await pageB.click('#nav-guard-btn');
    await pageB.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '김대현' && p.password),
      { timeout: 8000 }
    );
    await typeWithBiometricCadence(pageB, '#guard-verify-input', 'gachon2026', cadenceMinsu, false);
    await pageB.keyboard.press('Enter');
    await pageB.waitForSelector('#guard-result-card.is-block');
    await pageB.waitForTimeout(300);
    console.log('[E2E] Mode 1 Cross-User Intrusion on Laptop B:', await pageB.textContent('#guard-result-title'), await pageB.textContent('#guard-result-score'));
    await pageB.screenshot({
      path: path.join(SCREENSHOT_DIR, '03_hacker_intrusion_blocked.png'),
      fullPage: false,
    });

    // 5. Open Simultaneous Room & Profile Management Dialog for screenshot 04
    await pageA.click('#open-room-btn');
    await pageA.waitForTimeout(300);
    await pageA.screenshot({
      path: path.join(SCREENSHOT_DIR, '04_exception_handling_guard.png'),
      fullPage: false,
    });
    await pageA.click('#toggle-samples-btn');
    await pageA.waitForTimeout(150);
    await pageA.click('#close-room-btn');
    await pageA.waitForTimeout(200);

    // 6. Interactive Principles & Profile Structure Visualizer Tab (#view-principles)
    await pageA.click('#nav-principles-btn');
    await pageA.waitForSelector('#view-principles:not([hidden])');
    await pageA.click('#viz-preset-fast-btn');
    await pageA.waitForTimeout(220);

    // Verify live sandbox typing updates Step 1 & Step 3 in real time, then restore Korean preset for clean visual
    await typeWithBiometricCadence(pageA, '#viz-sandbox-input', 'keyprint live rhythm', cadenceDaehyeon, true);
    await pageA.waitForTimeout(280);
    const liveBadge = await pageA.textContent('#viz-signal-badge');
    if (!liveBadge || liveBadge.includes('예시 파형')) {
      throw new Error(`Expected live sandbox badge update, got: ${liveBadge}`);
    }
    await pageA.click('#viz-preset-fast-btn');
    await pageA.waitForTimeout(200);

    // Verify zero-latency custom hover tooltip across Step 2 64-D barcode columns
    const barcodeHit = pageA.locator('#viz-step2-svg .viz-col-hit').nth(12);
    await barcodeHit.scrollIntoViewIfNeeded();
    await barcodeHit.hover();
    await pageA.waitForTimeout(120);
    const tooltipVisible = await pageA.locator('#viz-tooltip').isVisible();
    if (!tooltipVisible) {
      throw new Error('Expected #viz-tooltip to be visible when hovering Step 2 barcode column');
    }
    console.log('[E2E] Principles & Profile Structure tab + hover tooltips verified!');

    await pageA.evaluate(() => window.scrollTo(0, 0));
    await pageA.mouse.move(4, 4);
    await pageA.waitForTimeout(150);
    await pageA.screenshot({
      path: path.join(SCREENSHOT_DIR, '06_principles_and_profile_viz.png'),
      fullPage: true,
    });

    // Clean up test room retained profiles on broker
    await pageA.evaluate(() => {
      for (const p of [...window.__keyprintTestApi.state.profiles]) {
        if (!p.isSample) {
          window.__keyprintTestApi.roomSync.deleteProfile(p.id);
        }
      }
    });
    await pageA.waitForTimeout(300);

    await contextA.close();
    await contextB.close();

    if (errors.length > 0) {
      console.error('[E2E] Console/Page errors:', errors);
      process.exitCode = 1;
    } else {
      console.log('[E2E] ALL MULTI-USER + BACKSPACE + DIVERSE SENTENCE CHECKS PASSED WITH ZERO ERRORS!');
    }
  } finally {
    await browser.close();
    server.close();
  }
}

run();
