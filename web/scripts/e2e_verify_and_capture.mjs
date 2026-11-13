import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.resolve(__dirname, '..');
const DIST_DIR = path.join(WEB_DIR, 'dist');
const MODEL_CONFIG_PATH = path.join(WEB_DIR, 'public', 'models', 'model_config.json');
const DEFAULT_SCREENSHOT_DIR = path.resolve(WEB_DIR, '..', 'assets', 'screenshots');
const CHROME_PATH =
  process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEMO_ROOM = 'DEMO_CAPTURE';

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

function parseCliArgs(argv) {
  let outDir = DEFAULT_SCREENSHOT_DIR;
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--out-dir' || arg === '-o') {
      if (!args[i + 1]) {
        throw new Error(`Missing directory path after ${arg}`);
      }
      outDir = path.resolve(process.cwd(), args[i + 1]);
      i++;
    } else if (arg.startsWith('--out-dir=')) {
      outDir = path.resolve(process.cwd(), arg.slice('--out-dir='.length));
    }
  }
  return { outDir };
}

function loadProtocolConfig() {
  const cfgPath = fs.existsSync(MODEL_CONFIG_PATH)
    ? MODEL_CONFIG_PATH
    : path.join(DIST_DIR, 'models', 'model_config.json');
  const raw = JSON.parse(fs.readFileSync(cfgPath, 'utf-8'));
  const passwordMinLength = raw.password_min_length ?? 12;
  const arenaEnrollSentences = raw.arena_enroll_sentences ?? 5;
  const arenaQuerySentences = raw.arena_query_sentences ?? 2;
  const passwordEnrollReps = raw.password_enroll_reps ?? 4;

  const basePw = 'GachonGuard2026';
  const targetPassword =
    basePw.length >= passwordMinLength
      ? basePw
      : basePw.padEnd(passwordMinLength, 'Key9').slice(0, passwordMinLength);

  return {
    passwordMinLength,
    arenaEnrollSentences,
    arenaQuerySentences,
    passwordEnrollReps,
    targetPassword,
  };
}

function startStaticServer(rootDir) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const reqPath = decodeURIComponent((req.url || '/').split('?')[0]);
      const relPath = reqPath === '/' ? 'index.html' : reqPath.replace(/^\/+/, '');
      const fullPath = path.resolve(rootDir, relPath);

      if (
        !fullPath.startsWith(rootDir + path.sep) ||
        !fs.existsSync(fullPath) ||
        fs.statSync(fullPath).isDirectory()
      ) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const ext = path.extname(fullPath).toLowerCase();
      res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
      fs.createReadStream(fullPath).pipe(res);
    });
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, url: `http://127.0.0.1:${addr.port}` });
    });
  });
}

// Deterministic PRNG + per-typist digraph rhythm factor so each user has a distinct 64-D rhythm shape
let prngSeed = 42;
function nextRand() {
  prngSeed = (prngSeed * 16807) % 2147483647;
  return prngSeed / 2147483647;
}

function digraphFactor(seedStr, key) {
  let x = 2166136261;
  for (const c of `${seedStr}|${key}`) {
    x = Math.imul(x ^ c.codePointAt(0), 16777619) >>> 0;
  }
  return 0.65 + 0.7 * ((x % 10007) / 10007);
}

const SPECIAL_CODES = {
  ' ': 'Space',
  '.': 'Period',
  ',': 'Comma',
  "'": 'Quote',
  '-': 'Minus',
  '!': 'Digit1',
  '?': 'Slash',
};

function codeForChar(ch) {
  if (SPECIAL_CODES[ch]) return SPECIAL_CODES[ch];
  if (/^[a-z]$/i.test(ch)) return `Key${ch.toUpperCase()}`;
  if (/^[0-9]$/.test(ch)) return `Digit${ch}`;
  return 'KeyA';
}

/**
 * Build an in-page timed keyboard event plan (keydown -> input -> keyup) with per-digraph rhythm,
 * optional mid-stream typo + Backspace repair, and optional Enter submission.
 */
function buildKeyboardPlan(text, cadence, { injectBackspace = false, submitEnter = true } = {}) {
  const plan = [];
  let t = 18;
  const chars = Array.from(text);
  const typoAt = injectBackspace ? Math.floor(chars.length * 0.42) : -1;

  const pushStroke = (ch) => {
    const code = codeForChar(ch);
    const isUpper = /^[A-Z]$/.test(ch);
    const hold =
      cadence.holdMs *
      digraphFactor(cadence.seed, `h${ch}`) *
      (1 + cadence.jitter * (nextRand() - 0.5));
    const dd =
      cadence.ddMs *
        digraphFactor(cadence.seed, `d${ch}`) *
        (1 + cadence.jitter * (nextRand() - 0.5)) +
      (ch === ' ' ? cadence.spaceExtraMs : 0);

    plan.push({
      t,
      type: 'keydown',
      key: ch,
      code,
      shift: isUpper && !cadence.capsLock,
      caps: Boolean(cadence.capsLock) && isUpper,
    });
    plan.push({ t, type: 'input', op: 'insert', data: ch, inputType: 'insertText' });
    plan.push({ t: t + hold, type: 'keyup', key: ch, code });
    t += dd;
  };

  chars.forEach((ch, idx) => {
    if (idx === typoAt) {
      pushStroke('x');
      t += cadence.detectMs;
      plan.push({ t, type: 'keydown', key: 'Backspace', code: 'Backspace' });
      plan.push({ t, type: 'input', op: 'delete', inputType: 'deleteContentBackward' });
      plan.push({ t: t + cadence.bsHoldMs, type: 'keyup', key: 'Backspace', code: 'Backspace' });
      t += cadence.resumeMs;
    }
    pushStroke(ch);
  });

  if (submitEnter) {
    plan.push({ t: t + 110, type: 'keydown', key: 'Enter', code: 'Enter' });
  }
  return plan.sort((a, b) => a.t - b.t);
}

async function runKeyboardPlan(page, selector, plan) {
  await page.evaluate(
    async ({ sel, events }) => {
      const el = document.querySelector(sel);
      if (!el) throw new Error(`Missing input element: ${sel}`);
      el.focus();
      const t0 = performance.now();
      for (const ev of events) {
        const wait = t0 + ev.t - performance.now();
        if (wait > 1) {
          await new Promise((r) => setTimeout(r, wait));
        }
        if (ev.type === 'keydown' || ev.type === 'keyup') {
          el.dispatchEvent(
            new KeyboardEvent(ev.type, {
              key: ev.key,
              code: ev.code,
              bubbles: true,
              cancelable: true,
              shiftKey: Boolean(ev.shift),
              modifierCapsLock: Boolean(ev.caps),
            })
          );
        } else {
          if (ev.op === 'delete') {
            el.value = Array.from(el.value).slice(0, -1).join('');
          } else {
            el.value += ev.data;
          }
          el.dispatchEvent(
            new InputEvent('input', {
              inputType: ev.inputType,
              data: ev.data ?? null,
              bubbles: true,
            })
          );
        }
      }
    },
    { sel: selector, events: plan }
  );
  await page.waitForTimeout(220);
}

async function selectGuardAccountByName(page, targetName) {
  const profileId = await page.evaluate((name) => {
    const found = window.__keyprintTestApi.state.profiles.find(
      (p) => p.name === name && p.password
    );
    return found ? found.id : null;
  }, targetName);
  if (!profileId) {
    throw new Error(`Password account not found for ${targetName}`);
  }
  await page.click(`#guard-account-list [data-guard-id="${profileId}"]`);
  await page.waitForFunction(
    (name) => document.getElementById('guard-target-name')?.textContent?.trim() === name,
    targetName,
    { timeout: 5000 }
  );
}

async function run() {
  const { outDir } = parseCliArgs(process.argv);
  const protocol = loadProtocolConfig();

  if (!fs.existsSync(path.join(DIST_DIR, 'index.html'))) {
    throw new Error(`Missing production build at ${DIST_DIR}. Run "npm run build" in web/ first.`);
  }

  fs.mkdirSync(outDir, { recursive: true });
  const { server, url } = await startStaticServer(DIST_DIR);
  const roomUrl = `${url}/?room=${DEMO_ROOM}`;
  console.log(
    `[E2E] Serving v2 build at ${roomUrl} | protocol: pw>=${protocol.passwordMinLength} ("${protocol.targetPassword}"), arena=${protocol.arenaEnrollSentences} enroll / ${protocol.arenaQuerySentences} query -> ${outDir}`
  );

  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
  });

  // Distinct biometric cadences: fast rollover typist (UD < 0, Shift key) vs slow staccato typist (UD > 0, CapsLock)
  const cadenceDaehyeon = {
    seed: 'daehyeon-rollover-v2',
    holdMs: 112,
    ddMs: 84,
    jitter: 0.14,
    spaceExtraMs: 24,
    detectMs: 220,
    resumeMs: 190,
    bsHoldMs: 68,
    capsLock: false,
  };
  const cadenceMinsu = {
    seed: 'minsu-staccato-v2',
    holdMs: 62,
    ddMs: 182,
    jitter: 0.14,
    spaceExtraMs: 135,
    detectMs: 420,
    resumeMs: 390,
    bsHoldMs: 105,
    capsLock: true,
  };

  try {
    // Shared local BrowserContext so pageA <-> pageB sync deterministically via BroadcastChannel + localStorage
    // while ALL external HTTP/MQTT traffic (broker.emqx.io, broker.hivemq.com) is 100% blocked!
    const context = await browser.newContext({
      viewport: { width: 1280, height: 880 },
      deviceScaleFactor: 2,
    });

    const blockedExternal = { http: [], ws: [] };
    await context.route('**/*', (route) => {
      const reqUrl = new URL(route.request().url());
      if (
        reqUrl.hostname === '127.0.0.1' ||
        reqUrl.protocol === 'data:' ||
        reqUrl.protocol === 'blob:'
      ) {
        return route.continue();
      }
      blockedExternal.http.push(reqUrl.host);
      return route.abort();
    });
    await context.routeWebSocket(/.*/, (ws) => {
      const wsHost = new URL(ws.url()).host;
      blockedExternal.ws.push(wsHost);
      // Close cleanly after handshake so roomSync settles to local mode without DevTools failed-handshake errors
      setTimeout(() => {
        ws.close({ code: 1000, reason: 'Deterministic local E2E mode' }).catch?.(() => {});
      }, 25);
    });

    const pageA = await context.newPage();
    const pageB = await context.newPage();

    const errors = [];
    for (const p of [pageA, pageB]) {
      p.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
      p.on('console', (msg) => {
        if (msg.type() === 'error') errors.push(`console: ${msg.text()}`);
      });
    }

    await Promise.all([
      pageA.goto(roomUrl, { waitUntil: 'load' }),
      pageB.goto(roomUrl, { waitUntil: 'load' }),
    ]);

    await Promise.all([
      pageA.waitForFunction(
        () => document.getElementById('engine-status-text')?.textContent?.includes('ONNX 준비 완료'),
        { timeout: 20000 }
      ),
      pageB.waitForFunction(
        () => document.getElementById('engine-status-text')?.textContent?.includes('ONNX 준비 완료'),
        { timeout: 20000 }
      ),
    ]);

    // Clear any stale localStorage profiles in DEMO_CAPTURE, keeping built-in sample profiles intact
    await pageA.evaluate(() => {
      for (const p of [...window.__keyprintTestApi.state.profiles]) {
        if (!p.isSample) {
          window.__keyprintTestApi.roomSync.deleteProfile(p.id);
        }
      }
    });
    await pageA.waitForTimeout(150);

    // Verify dynamic protocol values in browser engine match model_config.json
    const browserProtocol = await pageA.evaluate(() => ({
      arena: window.__keyprintTestApi.engine.arenaProtocol(),
      pwMinLen: window.__keyprintTestApi.engine.config?.password_min_length ?? 12,
    }));
    if (
      browserProtocol.arena.enroll !== protocol.arenaEnrollSentences ||
      browserProtocol.arena.query !== protocol.arenaQuerySentences ||
      browserProtocol.pwMinLen !== protocol.passwordMinLength
    ) {
      throw new Error(
        `Browser protocol mismatch: ${JSON.stringify(browserProtocol)} vs ${JSON.stringify(protocol)}`
      );
    }

    // Switch both tabs to English sentences for deterministic fast cadence synthesis
    await pageA.click('#lang-en-btn');
    await pageB.click('#lang-en-btn');

    // 1. Tab A enrolls '김대현' across arena_enroll_sentences (5) diverse sentences WITH Backspace repairs!
    await pageA.fill('#arena-name-input', '김대현');
    const seenSentences = new Set();
    const midShotIdx = Math.min(2, protocol.arenaEnrollSentences - 1);

    for (let s = 0; s < protocol.arenaEnrollSentences; s++) {
      const promptTxt = await pageA.evaluate(
        () => window.__keyprintTestApi.state.arenaEnrollSentence
      );
      seenSentences.add(promptTxt);
      const captureMidShot = s === midShotIdx;
      const plan = buildKeyboardPlan(promptTxt, cadenceDaehyeon, {
        injectBackspace: s <= midShotIdx,
        submitEnter: !captureMidShot,
      });
      await runKeyboardPlan(pageA, '#arena-enroll-input', plan);

      if (captureMidShot) {
        await pageA.waitForTimeout(180);
        await pageA.screenshot({
          path: path.join(outDir, '01_enrollment_template_ready.png'),
          fullPage: false,
        });
        await pageA.keyboard.press('Enter');
        await pageA.waitForTimeout(240);
      }
    }

    if (seenSentences.size !== protocol.arenaEnrollSentences) {
      throw new Error(
        `Expected ${protocol.arenaEnrollSentences} diverse sentences, got ${seenSentences.size}`
      );
    }
    console.log(
      `[E2E] Tab A enrolled 김대현 across ${protocol.arenaEnrollSentences} diverse sentences WITH Backspace!`
    );

    // Verify Tab B received 김대현's profile in real time via local BroadcastChannel sync!
    await pageB.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '김대현' && p.arena?.en),
      { timeout: 5000 }
    );
    console.log('[E2E] Deterministic local BroadcastChannel sync verified on Tab B!');

    // 2. Tab B enrolls '박민수' across arena_enroll_sentences (5) diverse sentences!
    await pageB.fill('#arena-name-input', '박민수');
    for (let s = 0; s < protocol.arenaEnrollSentences; s++) {
      const promptTxt = await pageB.evaluate(
        () => window.__keyprintTestApi.state.arenaEnrollSentence
      );
      const plan = buildKeyboardPlan(promptTxt, cadenceMinsu, {
        injectBackspace: s === 0,
        submitEnter: true,
      });
      await runKeyboardPlan(pageB, '#arena-enroll-input', plan);
    }

    await pageA.waitForFunction(
      () =>
        window.__keyprintTestApi.state.profiles.filter((p) => !p.isSample && p.arena?.en).length >=
        2,
      { timeout: 5000 }
    );

    // 3. Mode B: Password Guard (12+ char dual-encoder password enrollment + genuine pass + hacker block)
    await pageA.click('#nav-guard-btn');
    await pageA.click('#guard-sub-enroll');
    await pageA.fill('#guard-enroll-name', '김대현');
    await pageA.fill('#guard-enroll-pw-target', protocol.targetPassword);
    for (let r = 0; r < protocol.passwordEnrollReps; r++) {
      const plan = buildKeyboardPlan(protocol.targetPassword, cadenceDaehyeon, {
        injectBackspace: r === 1,
        submitEnter: true,
      });
      await runKeyboardPlan(pageA, '#guard-enroll-input', plan);
    }
    console.log(
      `[E2E] Registered ${protocol.targetPassword.length}-char dual-encoder password profile for 김대현!`
    );

    // Genuine owner verification on Tab A -> ACCESS GRANTED
    await pageA.click('#guard-sub-verify');
    await selectGuardAccountByName(pageA, '김대현');
    await runKeyboardPlan(
      pageA,
      '#guard-verify-input',
      buildKeyboardPlan(protocol.targetPassword, cadenceDaehyeon, {
        injectBackspace: true,
        submitEnter: true,
      })
    );
    await pageA.waitForSelector('#guard-result-card.is-pass:not([hidden])', { timeout: 5000 });
    await pageA.waitForTimeout(250);
    console.log(
      '[E2E] Mode B Genuine Login:',
      (await pageA.textContent('#guard-result-title'))?.trim(),
      (await pageA.textContent('#guard-result-score'))?.trim()
    );
    await pageA.screenshot({
      path: path.join(outDir, '02_genuine_verification_granted.png'),
      fullPage: false,
    });

    // Cross-user intrusion on Tab B with exact leaked 15-char password -> INTRUSION BLOCKED
    await pageB.click('#nav-guard-btn');
    await pageB.waitForFunction(
      () => window.__keyprintTestApi.state.profiles.some((p) => p.name === '김대현' && p.password),
      { timeout: 5000 }
    );
    await selectGuardAccountByName(pageB, '김대현');
    await runKeyboardPlan(
      pageB,
      '#guard-verify-input',
      buildKeyboardPlan(protocol.targetPassword, cadenceMinsu, {
        injectBackspace: false,
        submitEnter: true,
      })
    );
    await pageB.waitForSelector('#guard-result-card.is-block:not([hidden])', { timeout: 5000 });
    await pageB.waitForTimeout(250);
    console.log(
      '[E2E] Mode B Cross-User Intrusion Blocked:',
      (await pageB.textContent('#guard-result-title'))?.trim(),
      (await pageB.textContent('#guard-result-score'))?.trim()
    );
    await pageB.screenshot({
      path: path.join(outDir, '03_hacker_intrusion_blocked.png'),
      fullPage: false,
    });

    // 4. Room Sync & Profile Management Dialog (screenshot 04)
    await pageA.click('#open-room-btn');
    await pageA.waitForSelector('#room-dialog[open]', { timeout: 5000 });
    const hasSamples = await pageA.evaluate(() =>
      window.__keyprintTestApi.state.profiles.some((p) => p.isSample)
    );
    if (!hasSamples) {
      await pageA.click('#toggle-samples-btn');
      await pageA.waitForTimeout(180);
    }
    await pageA.waitForTimeout(220);
    await pageA.screenshot({
      path: path.join(outDir, '04_exception_handling_guard.png'),
      fullPage: false,
    });
    await pageA.click('#close-room-btn');
    await pageA.waitForTimeout(180);

    // 5. Mode A: Blind 1:N Identification across arena_query_sentences (2) unseen sentences!
    await pageA.click('#nav-arena-btn');
    await pageA.click('#arena-sub-blind');
    await pageA.evaluate(() => {
      document.getElementById('arena-result-card').hidden = true;
    });

    for (let q = 0; q < protocol.arenaQuerySentences; q++) {
      const blindSentence = await pageA.evaluate(
        () => window.__keyprintTestApi.state.arenaBlindSentence
      );
      const isLastQuery = q === protocol.arenaQuerySentences - 1;
      await runKeyboardPlan(
        pageA,
        '#arena-blind-input',
        buildKeyboardPlan(blindSentence, cadenceDaehyeon, {
          injectBackspace: isLastQuery,
          submitEnter: true,
        })
      );
      if (!isLastQuery) {
        const cardStillHidden = await pageA.evaluate(
          () => document.getElementById('arena-result-card').hidden
        );
        if (!cardStillHidden) {
          throw new Error(
            `Arena result card appeared early on sentence ${q + 1}/${protocol.arenaQuerySentences}`
          );
        }
      }
    }

    await pageA.waitForSelector('#arena-result-card:not([hidden])', { timeout: 5000 });
    await pageA.waitForTimeout(280);
    const blindTitle = (await pageA.textContent('#arena-result-title'))?.trim() || '';
    const blindMeta = (await pageA.textContent('#arena-result-meta'))?.trim() || '';
    console.log(`[E2E] Mode A ${protocol.arenaQuerySentences}-Sentence Blind Result: ${blindTitle} | ${blindMeta}`);
    if (!blindTitle.includes('김대현')) {
      throw new Error(`Expected Blind 1:N winner to be 김대현, got: ${blindTitle}`);
    }

    await pageA.screenshot({
      path: path.join(outDir, '05_typing_arena_who_is_typing.png'),
      fullPage: false,
    });

    // 6. Mode C: Principles & Profile Structure tab directly from the active verdict (#arena-inspect-viz-btn)
    // Capture screenshot BEFORE touching presets/sandbox so Step 1, Step 2 (64-D Barcode + Mean Pairwise Cosine),
    // and Step 3 (2D PCA Hypersphere Map) reflect the exact active 김대현 verdict!
    await pageA.click('#arena-inspect-viz-btn');
    await pageA.waitForSelector('#view-principles:not([hidden])', { timeout: 5000 });
    await pageA.waitForTimeout(300);

    // Ensure 김대현's profile chip is selected in Step 2 so the 64-D barcode & stability match the active winner
    await pageA.evaluate(() => {
      const chips = Array.from(document.querySelectorAll('#viz-profile-chips button'));
      const daehyeonChip = chips.find((btn) => btn.textContent?.includes('김대현'));
      if (daehyeonChip && !daehyeonChip.classList.contains('is-selected')) {
        daehyeonChip.click();
      }
    });
    await pageA.waitForTimeout(180);

    const vizCheck = await pageA.evaluate(() => ({
      step1HasSvg: Boolean(document.querySelector('#viz-step1-svg svg')),
      step2HasSvg: Boolean(document.querySelector('#viz-step2-svg svg')),
      step3HasSvg: Boolean(document.querySelector('#viz-step3-svg svg')),
      stabilityText: document.getElementById('viz-stat-stability')?.textContent?.trim() || '',
      step3Text: document.getElementById('viz-step3-svg')?.textContent || '',
    }));
    if (!vizCheck.step1HasSvg || !vizCheck.step2HasSvg || !vizCheck.step3HasSvg) {
      throw new Error(`Missing SVG in Mode C Principles tab: ${JSON.stringify(vizCheck)}`);
    }
    if (!vizCheck.stabilityText || !vizCheck.step3Text.includes('김대현')) {
      throw new Error(
        `Mode C Step 2/3 did not reflect active 김대현 verdict: ${JSON.stringify(vizCheck)}`
      );
    }

    await pageA.evaluate(() => window.scrollTo(0, 0));
    await pageA.mouse.move(4, 4);
    await pageA.waitForTimeout(200);
    await pageA.screenshot({
      path: path.join(outDir, '06_principles_and_profile_viz.png'),
      fullPage: true,
    });

    // Post-screenshot interactive verification: Tempo-Invariance tab, 64-D hover tooltip, and live sandbox typing
    await pageA.click('#viz-wave-tab-tempo');
    await pageA.waitForTimeout(120);
    const tempoSvgOk = await pageA.evaluate(() =>
      Boolean(document.querySelector('#viz-step1-svg svg'))
    );
    if (!tempoSvgOk) {
      throw new Error('Expected Tempo-Invariance SVG to render in Step 1');
    }
    await pageA.click('#viz-wave-tab-piano');

    const barcodeHit = pageA.locator('#viz-step2-svg .viz-col-hit').nth(12);
    await barcodeHit.scrollIntoViewIfNeeded();
    await barcodeHit.hover();
    await pageA.waitForTimeout(120);
    if (!(await pageA.locator('#viz-tooltip').isVisible())) {
      throw new Error('Expected #viz-tooltip to be visible when hovering Step 2 barcode column');
    }

    await runKeyboardPlan(
      pageA,
      '#viz-sandbox-input',
      buildKeyboardPlan('keyprint live rhythm', cadenceDaehyeon, {
        injectBackspace: true,
        submitEnter: false,
      })
    );
    const liveBadge = (await pageA.textContent('#viz-signal-badge')) || '';
    if (!liveBadge || liveBadge.includes('예시 파형')) {
      throw new Error(`Expected live sandbox badge update, got: ${liveBadge}`);
    }

    console.log(
      `[E2E] Captured all 6 v2 screenshots in ${outDir} (blocked external WS: ${[...new Set(blockedExternal.ws)].join(', ') || 'none'})`
    );

    await context.close();

    if (errors.length > 0) {
      console.error('[E2E] Console/Page errors:', errors);
      process.exitCode = 1;
    } else {
      console.log('[E2E] ALL PROTOCOL-DRIVEN V2 CHECKS & SCREENSHOTS PASSED WITH ZERO ERRORS!');
    }
  } finally {
    await browser.close();
    server.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
