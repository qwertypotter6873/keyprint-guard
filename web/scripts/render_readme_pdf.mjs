import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..', '..');
const README_PATH = path.join(ROOT_DIR, 'README.md');
const DEFAULT_PDF_OUT_PATH = path.join(ROOT_DIR, '포트폴리오_201935607_김대현.pdf');
const GITHUB_REPO_BASE = 'https://github.com/qwertypotter6873/keyprint-guard';
const GITHUB_BLOB_BASE = `${GITHUB_REPO_BASE}/blob/main`;
const CHROME_PATH =
  process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// A4 = 210mm x 297mm; margins = 15mm top/bottom, 14mm left/right -> printable = 182mm x 267mm.
// At CSS 96 DPI: 182mm = 687.87px, 267mm = 1009.13px. Use 686px for strict overflow checking.
const A4_PRINTABLE_WIDTH_PX = 686;
const A4_PRINTABLE_HEIGHT_PX = 1009;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.css': 'text/css; charset=utf-8',
};

function parseCliOutPath(argv) {
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--out' || arg === '-o') {
      if (!args[i + 1]) {
        throw new Error(`Missing value after ${arg}`);
      }
      return path.resolve(process.cwd(), args[i + 1]);
    }
    if (arg.startsWith('--out=')) {
      return path.resolve(process.cwd(), arg.slice('--out='.length));
    }
    if (!arg.startsWith('-')) {
      return path.resolve(process.cwd(), arg);
    }
  }
  return DEFAULT_PDF_OUT_PATH;
}

/**
 * Rewrite repository-relative (or accidental local file://) Markdown hrefs to GitHub blob URLs,
 * preserving #fragments and converting :L10-L20 / :10-20 suffixes cleanly.
 */
function rewriteRepositoryHref(rawHref) {
  if (!rawHref) return rawHref;
  const trimmed = rawHref.trim();
  if (
    trimmed.startsWith('#') ||
    trimmed.startsWith('http://') ||
    trimmed.startsWith('https://') ||
    trimmed.startsWith('mailto:')
  ) {
    return trimmed;
  }

  let working = trimmed;

  // If a file:// URL pointing inside ROOT_DIR ever appeared, strip ROOT_DIR prefix cleanly
  if (working.startsWith('file://')) {
    try {
      const u = new URL(working);
      const decodedPathname = decodeURIComponent(u.pathname);
      const rootNorm = ROOT_DIR.normalize('NFC');
      const pathNorm = decodedPathname.normalize('NFC');
      if (pathNorm.startsWith(rootNorm)) {
        working = pathNorm.slice(rootNorm.length).replace(/^\/+/, '') + (u.hash || '');
      } else {
        working = path.posix.basename(decodedPathname) + (u.hash || '');
      }
    } catch {
      working = working.replace(/^file:\/\/+/, '');
    }
  }

  const hashIdx = working.indexOf('#');
  let pathPart = hashIdx >= 0 ? working.slice(0, hashIdx) : working;
  let fragment = hashIdx >= 0 ? working.slice(hashIdx) : '';

  // Remove query string if present on relative file links
  const queryIdx = pathPart.indexOf('?');
  if (queryIdx >= 0) {
    pathPart = pathPart.slice(0, queryIdx);
  }

  // Handle :L10-L20 or :10-20 line-range suffixes when no #fragment was given
  const colonLineMatch = pathPart.match(/^(.*?):L?(\d+)(?:-L?(\d+))?$/i);
  if (colonLineMatch) {
    pathPart = colonLineMatch[1];
    if (!fragment) {
      fragment = colonLineMatch[3]
        ? `#L${colonLineMatch[2]}-L${colonLineMatch[3]}`
        : `#L${colonLineMatch[2]}`;
    }
  }

  // Normalize ./, leading /, and trailing /
  let cleanRel = path.posix.normalize(pathPart.replace(/\\/g, '/')).replace(/^(?:\.\/)+/, '');
  cleanRel = cleanRel.replace(/^\/+/, '').replace(/\/+$/, '');
  if (cleanRel === '.' || cleanRel === '') {
    return `${GITHUB_REPO_BASE}${fragment}`;
  }

  const encodedPath = cleanRel
    .split('/')
    .map((seg) => {
      try {
        return encodeURIComponent(decodeURIComponent(seg));
      } catch {
        return encodeURIComponent(seg);
      }
    })
    .join('/');

  return `${GITHUB_BLOB_BASE}/${encodedPath}${fragment}`;
}

function startPortfolioServer(htmlContent) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const reqPath = decodeURIComponent((req.url || '/').split('?')[0]);
      if (reqPath === '/' || reqPath === '/__portfolio__.html') {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(htmlContent);
        return;
      }

      const relPath = reqPath.replace(/^\/+/, '');
      const fullPath = path.resolve(ROOT_DIR, relPath);
      if (
        !fullPath.startsWith(ROOT_DIR + path.sep) ||
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
      resolve({ server, url: `http://127.0.0.1:${addr.port}/__portfolio__.html` });
    });
  });
}

async function generatePdf() {
  const pdfOutPath = parseCliOutPath(process.argv);
  fs.mkdirSync(path.dirname(pdfOutPath), { recursive: true });

  const mdRaw = fs.readFileSync(README_PATH, 'utf-8');
  const htmlBody = marked.parse(mdRaw, { gfm: true, breaks: false });

  const fullHtml = `<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>KeyPrint Guard - 머신러닝 포트폴리오 (201935607 김대현)</title>
  <style>
    @page {
      size: A4;
      margin: 15mm 14mm 15mm 14mm;
    }
    *, *::before, *::after {
      box-sizing: border-box;
    }
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      color: #0f172a;
      line-height: 1.6;
      font-size: 10pt;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    h1 {
      font-size: 18.5pt;
      font-weight: 800;
      color: #0f172a;
      border-bottom: 2.5px solid #2563eb;
      padding-bottom: 8px;
      margin-top: 0;
      margin-bottom: 14px;
    }
    h2 {
      font-size: 13.2pt;
      font-weight: 800;
      color: #1e3a8a;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 20px;
      margin-bottom: 10px;
      break-after: avoid;
    }
    h3 {
      font-size: 11pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 14px;
      margin-bottom: 6px;
      break-after: avoid;
    }
    h4 {
      font-size: 10.2pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 11px;
      margin-bottom: 5px;
      break-after: avoid;
    }
    p, ul, ol {
      margin-top: 4px;
      margin-bottom: 8px;
      max-width: 100%;
    }
    ul, ol {
      padding-left: 20px;
    }
    li {
      margin-bottom: 3px;
    }
    table {
      width: 100%;
      max-width: 100%;
      table-layout: auto;
      border-collapse: collapse;
      margin: 10px 0 14px 0;
      font-size: 8.3pt;
      word-break: break-word;
    }
    thead {
      display: table-header-group;
    }
    tr {
      break-inside: avoid;
    }
    th, td {
      border: 1px solid #cbd5e1;
      padding: 5px 6px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
      min-width: 0;
    }
    th {
      background-color: #f1f5f9;
      font-weight: 700;
      color: #0f172a;
    }
    tr:nth-child(even) td {
      background-color: #f8fafc;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 8.4pt;
      color: #0f172a;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    td code, th code {
      font-size: 7.9pt;
    }
    pre {
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 12px;
      border-radius: 8px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      max-width: 100%;
      overflow-x: hidden;
      font-size: 7.8pt;
      line-height: 1.45;
      break-inside: avoid;
    }
    pre code {
      background: transparent;
      color: inherit;
      padding: 0;
      font-size: inherit;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    blockquote {
      border-left: 4px solid #2563eb;
      background: #eff6ff;
      margin: 10px 0;
      padding: 8px 12px;
      border-radius: 0 6px 6px 0;
      color: #1e3a8a;
      max-width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      break-inside: avoid;
    }
    img {
      max-width: 100%;
      max-height: 215mm;
      width: auto;
      height: auto;
      display: block;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      margin: 8px auto;
      break-inside: avoid;
    }
    a {
      color: #2563eb;
      text-decoration: none;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
  </style>
</head>
<body>
${htmlBody}
</body>
</html>`;

  const { server, url } = await startPortfolioServer(fullHtml);
  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
  });

  try {
    const context = await browser.newContext({
      viewport: { width: A4_PRINTABLE_WIDTH_PX, height: A4_PRINTABLE_HEIGHT_PX },
    });
    // Block any accidental external image/font network fetch while rendering local README + assets
    await context.route('**/*', (route) => {
      const reqUrl = new URL(route.request().url());
      if (reqUrl.hostname === '127.0.0.1' || reqUrl.protocol === 'data:') {
        return route.continue();
      }
      return route.abort();
    });

    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle' });

    // Assign GitHub-compatible heading IDs so #section links navigate internally in the PDF,
    // and rewrite every relative <a href> to https://github.com/qwertypotter6873/keyprint-guard/blob/main/<clean_rel_path>.
    const rawHrefs = await page.evaluate(() =>
      Array.from(document.querySelectorAll('a[href]'), (a) => a.getAttribute('href') || '')
    );
    const rewrittenHrefs = rawHrefs.map((h) => rewriteRepositoryHref(h));

    await page.evaluate(
      ({ rewritten, repoBase }) => {
        const slugCounts = new Map();
        for (const heading of document.querySelectorAll('h1, h2, h3, h4, h5, h6')) {
          if (heading.id) continue;
          const baseSlug = (heading.textContent || '')
            .trim()
            .toLowerCase()
            .replace(/[^\p{L}\p{N}\s\-_]/gu, '')
            .replace(/\s/g, '-');
          if (!baseSlug) continue;
          const seen = slugCounts.get(baseSlug) || 0;
          slugCounts.set(baseSlug, seen + 1);
          heading.id = seen === 0 ? baseSlug : `${baseSlug}-${seen}`;
        }

        const anchors = Array.from(document.querySelectorAll('a[href]'));
        anchors.forEach((a, idx) => {
          const nextHref = rewritten[idx];
          if (nextHref.startsWith('#')) {
            const rawTarget = nextHref.slice(1);
            let decodedTarget = rawTarget;
            try {
              decodedTarget = decodeURIComponent(rawTarget);
            } catch {
              // ignore
            }
            const targetEl =
              document.getElementById(decodedTarget) || document.getElementById(rawTarget);
            if (!targetEl) {
              // Unmatched fragment: point to GitHub README anchor rather than local server URL
              a.setAttribute('href', `${repoBase}${nextHref}`);
            } else if (targetEl.id !== rawTarget) {
              a.setAttribute('href', `#${targetEl.id}`);
            }
          } else {
            a.setAttribute('href', nextHref);
          }
        });
      },
      { rewritten: rewrittenHrefs, repoBase: GITHUB_REPO_BASE }
    );

    // Switch to print media at A4 printable width BEFORE running layout/link assertions
    await page.emulateMedia({ media: 'print' });

    const audit = await page.evaluate(() => {
      const fileLinks = [];
      const badLocalLinks = [];
      for (const a of document.querySelectorAll('a[href]')) {
        const raw = a.getAttribute('href') || '';
        const resolved = a.href || '';
        if (raw.startsWith('file:') || resolved.startsWith('file:')) {
          fileLinks.push({ raw, resolved, text: (a.textContent || '').trim().slice(0, 60) });
        }
        if (!raw.startsWith('#') && resolved.includes('127.0.0.1')) {
          badLocalLinks.push({ raw, resolved, text: (a.textContent || '').trim().slice(0, 60) });
        }
      }

      const docScrollWidth = document.documentElement.scrollWidth;
      const docClientWidth = document.documentElement.clientWidth;
      const overflowingElements = [];

      for (const el of document.querySelectorAll('pre, table, img, blockquote')) {
        const parent = el.parentElement || document.documentElement;
        const elRect = el.getBoundingClientRect();
        const parentRect = parent.getBoundingClientRect();
        const exceedsContainer = elRect.width > parentRect.width + 1;
        const exceedsViewport = elRect.right > docClientWidth + 1;
        const internalScrollOverflow =
          (el.tagName === 'PRE' || el.tagName === 'TABLE') && el.scrollWidth > el.clientWidth + 1;
        if (exceedsContainer || exceedsViewport || internalScrollOverflow) {
          overflowingElements.push({
            tag: el.tagName,
            elWidth: Math.round(elRect.width * 10) / 10,
            parentWidth: Math.round(parentRect.width * 10) / 10,
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth,
            sample: (el.textContent || el.getAttribute('src') || '').trim().slice(0, 80),
          });
        }
      }

      const brokenImages = [];
      for (const img of document.querySelectorAll('img')) {
        if (!img.complete || img.naturalWidth === 0) {
          brokenImages.push(img.getAttribute('src') || '');
        }
      }

      return {
        fileLinks,
        badLocalLinks,
        docScrollWidth,
        docClientWidth,
        overflowingElements,
        brokenImages,
      };
    });

    if (audit.fileLinks.length > 0) {
      throw new Error(
        `[PDF] Found ${audit.fileLinks.length} file:// link(s): ${JSON.stringify(audit.fileLinks)}`
      );
    }
    if (audit.badLocalLinks.length > 0) {
      throw new Error(
        `[PDF] Found ${audit.badLocalLinks.length} unrewritten relative link(s): ${JSON.stringify(audit.badLocalLinks)}`
      );
    }
    if (audit.brokenImages.length > 0) {
      throw new Error(
        `[PDF] Found ${audit.brokenImages.length} broken image(s): ${JSON.stringify(audit.brokenImages)}`
      );
    }
    if (audit.docScrollWidth > audit.docClientWidth + 1 || audit.overflowingElements.length > 0) {
      throw new Error(
        `[PDF] Horizontal overflow detected (doc ${audit.docScrollWidth}px > ${audit.docClientWidth}px): ${JSON.stringify(audit.overflowingElements.slice(0, 5))}`
      );
    }

    await page.pdf({
      path: pdfOutPath,
      format: 'A4',
      printBackground: true,
      displayHeaderFooter: false,
    });

    const pdfBytes = fs.readFileSync(pdfOutPath);
    if (pdfBytes.includes(Buffer.from('file://'))) {
      throw new Error(`[PDF] Raw PDF bytes contain forbidden "file://" URI in ${pdfOutPath}`);
    }
    if (pdfBytes.includes(Buffer.from('127.0.0.1'))) {
      throw new Error(`[PDF] Raw PDF bytes contain local server URI "127.0.0.1" in ${pdfOutPath}`);
    }

    console.log(
      `[PDF] Generated ${pdfOutPath} (${Math.round(pdfBytes.length / 1024)} KB, 0 file:// links, 0 horizontal overflows)`
    );
  } finally {
    await browser.close();
    server.close();
  }
}

generatePdf().catch((err) => {
  console.error(err);
  process.exit(1);
});
