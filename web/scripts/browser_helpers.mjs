import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

/** Find an installed Chrome/Chromium, or a Playwright-managed Chromium on any supported OS. */
export function findChromeExecutable(chromium) {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    path.join(os.homedir(), 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/snap/bin/chromium',
    ...['PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'].flatMap((key) => {
      const base = process.env[key];
      return base ? [path.join(base, 'Google/Chrome/Application/chrome.exe'), path.join(base, 'Chromium/Application/chrome.exe')] : [];
    }),
    ...(process.env.PATH || '').split(path.delimiter).filter(Boolean).flatMap((dir) =>
      ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome.exe'].map((name) => path.join(dir, name))),
    chromium?.executablePath(),
  ].filter(Boolean);
  if (process.env.CHROME_PATH && !fs.existsSync(process.env.CHROME_PATH)) {
    throw new Error(`CHROME_PATH does not exist: ${process.env.CHROME_PATH}`);
  }
  for (const candidate of candidates) {
    try {
      if (!fs.statSync(candidate).isFile()) continue;
      fs.accessSync(candidate, process.platform === 'win32' ? fs.constants.F_OK : fs.constants.X_OK);
      return candidate;
    } catch {
      // Continue through platform-specific paths that do not exist on this machine.
    }
  }
  throw new Error('Chrome/Chromium was not found. Install Chrome, set CHROME_PATH to its executable, or run "npx playwright-core install chromium" in web/.');
}
