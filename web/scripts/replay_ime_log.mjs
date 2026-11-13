/**
 * Replay a log captured with public/ime-debug.html through the keystroke recorder.
 *
 *   node scripts/replay_ime_log.mjs <log.json> [path/to/recorder.js]
 *
 * Prints the Backspace count, clean digraphs and synthetic strokes the recorder derives from the
 * real browser event sequence (pass an older recorder.js to compare before/after a fix).
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const [, , logPath, recorderPath = fileURLToPath(new URL('../src/recorder.js', import.meta.url))] = process.argv;
if (!logPath) {
  console.error('usage: node scripts/replay_ime_log.mjs <log.json> [recorder.js]');
  process.exit(1);
}
const { SmartKeystrokeRecorder } = await import(pathToFileURL(resolve(recorderPath)).href);
const log = JSON.parse(readFileSync(logPath, 'utf8'));

class FakeInput extends EventTarget {
  constructor() {
    super();
    this.value = '';
  }
}
const el = new FakeInput();
const rec = new SmartKeystrokeRecorder(el, {});
for (const r of log.rows) {
  if (!['keydown', 'keyup', 'input'].includes(r.type)) continue;
  if (r.type === 'keydown' && r.key === 'Enter') continue;
  const e = new Event(r.type, { cancelable: true });
  Object.assign(e, {
    key: r.key ?? '',
    code: r.code ?? '',
    repeat: Boolean(r.repeat),
    isComposing: Boolean(r.isComposing),
    shiftKey: Boolean(r.shift),
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    inputType: r.inputType ?? '',
    getModifierState: () => false,
  });
  Object.defineProperty(e, 'timeStamp', { value: r.t });
  if (r.type === 'input') el.value = r.value;
  el.dispatchEvent(e);
}
const out = rec.extractCleanStroke();
const typesSeen = [...new Set(log.rows.filter((r) => r.type === 'input').map((r) => r.inputType))];
console.log(`browser: ${log.ua}`);
console.log(`input event types: ${typesSeen.join(', ')}`);
console.log(`text: "${out.typedText}"`);
console.log(
  `backspaces ${out.backspaceCount} | clean digraphs ${out.cleanDigraphCount} | strokes ${rec.stack.length} (synthetic ${rec.stack.filter((s) => s.synthetic).length}) | stitches ${out.stroke.stitchFlags.filter(Boolean).length} | mode ${out.inputMode}`,
);
