import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { marked } from 'marked';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..', '..');
const README_PATH = path.join(ROOT_DIR, 'README.md');
const PDF_OUT_PATH = path.join(ROOT_DIR, '포트폴리오_201935607_김대현.pdf');

async function generatePdf() {
  const mdRaw = fs.readFileSync(README_PATH, 'utf-8');
  const htmlBody = marked.parse(mdRaw, { gfm: true, breaks: false });

  // Rewrite relative image src="assets/..." to absolute file:// URLs so Chrome loads all local images cleanly
  const resolvedHtml = htmlBody.replace(/src="(?!https?:|data:|file:)([^"]+)"/g, (_, rel) => {
    const absPath = path.resolve(ROOT_DIR, rel);
    return `src="${pathToFileURL(absPath).href}"`;
  });

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
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      color: #0f172a;
      line-height: 1.62;
      font-size: 10.2pt;
      max-width: 100%;
    }
    h1 {
      font-size: 19pt;
      font-weight: 800;
      color: #0f172a;
      border-bottom: 2.5px solid #2563eb;
      padding-bottom: 8px;
      margin-top: 0;
      margin-bottom: 14px;
    }
    h2 {
      font-size: 13.5pt;
      font-weight: 800;
      color: #1e3a8a;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 22px;
      margin-bottom: 10px;
      break-after: avoid;
    }
    h3 {
      font-size: 11.2pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 14px;
      margin-bottom: 6px;
      break-after: avoid;
    }
    p, ul, ol {
      margin-top: 4px;
      margin-bottom: 8px;
    }
    ul, ol {
      padding-left: 20px;
    }
    li {
      margin-bottom: 3px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px 0;
      font-size: 9.1pt;
      break-inside: avoid;
    }
    th, td {
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
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
      font-size: 8.8pt;
      color: #0f172a;
    }
    pre {
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 12px;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 8.4pt;
      line-height: 1.45;
      break-inside: avoid;
    }
    pre code {
      background: transparent;
      color: inherit;
      padding: 0;
    }
    blockquote {
      border-left: 4px solid #2563eb;
      background: #eff6ff;
      margin: 10px 0;
      padding: 8px 12px;
      border-radius: 0 6px 6px 0;
      color: #1e3a8a;
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
    }
  </style>
</head>
<body>
${resolvedHtml}
</body>
</html>`;

  const tempHtmlPath = path.join(ROOT_DIR, '.tmp_portfolio_render.html');
  fs.writeFileSync(tempHtmlPath, fullHtml, 'utf-8');

  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });

  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(tempHtmlPath).href, { waitUntil: 'networkidle' });
    await page.pdf({
      path: PDF_OUT_PATH,
      format: 'A4',
      printBackground: true,
      displayHeaderFooter: false,
    });
    console.log(`[PDF] Generated ${PDF_OUT_PATH} (${Math.round(fs.statSync(PDF_OUT_PATH).size / 1024)} KB)`);
  } finally {
    await browser.close();
    fs.rmSync(tempHtmlPath, { force: true });
  }
}

generatePdf();
