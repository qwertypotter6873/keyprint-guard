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

async function run() {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  const { server, url } = await startStaticServer(DIST_DIR);
  console.log(`[E2E] Serving static build at ${url}`);

  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 980 },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();

    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForFunction(
      () => document.getElementById('engine-status')?.textContent?.includes('ONNX WASM 활성'),
      { timeout: 15000 }
    );
    const engineTxt = await page.textContent('#engine-status');
    console.log(`[E2E] Engine Ready: ${engineTxt}`);

    // 1. Mode 1: Click first Unseen Test Subject Preset -> Capture Enrollment Screenshot
    await page.click('#benchmark-preset-buttons .preset-user-btn');
    await page.waitForSelector('#template-summary-card:not([hidden])');
    await page.waitForTimeout(300);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '01_enrollment_template_ready.png'),
      fullPage: false,
    });
    console.log('[E2E] Saved 01_enrollment_template_ready.png');

    // 2. Mode 1: Genuine User Verification -> ACCESS GRANTED
    await page.click('#sim-genuine-btn');
    await page.waitForSelector('#verdict-card.state-granted');
    await page.waitForTimeout(300);
    const genuineScore = await page.textContent('#verdict-score');
    console.log(`[E2E] Genuine Verification Score: ${genuineScore}`);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '02_genuine_verification_granted.png'),
      fullPage: false,
    });
    console.log('[E2E] Saved 02_genuine_verification_granted.png');

    // 3. Mode 1: Human Hacker + Macro Bot Intrusions -> INTRUSION BLOCKED
    await page.click('#sim-macro-bot-btn');
    await page.waitForTimeout(180);
    await page.click('#sim-human-hacker-btn');
    await page.waitForSelector('#verdict-card.state-blocked');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(300);
    const hackerScore = await page.textContent('#verdict-score');
    console.log(`[E2E] Skilled Human Hacker Score: ${hackerScore} (BLOCKED)`);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '03_hacker_intrusion_blocked.png'),
      fullPage: true,
    });
    console.log('[E2E] Saved 03_hacker_intrusion_blocked.png');

    // 4. Mode 1: Exception Handling Verification (Backspace / Wrong Password)
    await page.focus('#verify-input');
    await page.keyboard.type('.tie5Wrong', { delay: 55 });
    await page.keyboard.press('Backspace');
    await page.keyboard.press('Enter');
    await page.waitForSelector('#verdict-card.state-exception');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(250);
    const excBadge = await page.textContent('#verdict-badge');
    console.log(`[E2E] Exception Handling Verified: ${excBadge}`);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '04_exception_handling_guard.png'),
      fullPage: false,
    });
    console.log('[E2E] Saved 04_exception_handling_guard.png');

    // 5. Mode 2: Switch to 1:N Typing-Practice Person Classifier ("Who Is Typing?")
    await page.click('#tab-mode-arena');
    await page.waitForSelector('#arena-view:not([hidden])');
    await page.click('#arena-preset-gallery-btns .arena-gal-btn');
    await page.waitForFunction(
      () => Number(document.getElementById('arena-gallery-count')?.textContent || '0') >= 8
    );
    // Also type partial characters in enroll box to show live green typing-practice highlight in screenshot!
    await page.focus('#arena-enroll-input');
    const targetSentence = await page.inputValue('#arena-sentence-input');
    await page.keyboard.type(targetSentence.slice(0, Math.min(11, targetSentence.length)), { delay: 30 });

    // Trigger blind typist identification!
    await page.click('#arena-sim-random-btn');
    await page.waitForSelector('#arena-ranking-list .rank-row');
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(350);
    const arenaTitle = await page.textContent('#arena-verdict-title');
    console.log(`[E2E] Mode 2 1:N Identification Result: ${arenaTitle}`);
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, '05_typing_arena_who_is_typing.png'),
      fullPage: true,
    });
    console.log('[E2E] Saved 05_typing_arena_who_is_typing.png');

    if (errors.length > 0) {
      console.error('[E2E] Console/Page errors:', errors);
      process.exitCode = 1;
    } else {
      console.log('[E2E] Zero console/page errors! All 5 interactive scenarios passed.');
    }
  } finally {
    await browser.close();
    server.close();
  }
}

run();
