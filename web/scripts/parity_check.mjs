/**
 * Offline JS <-> Python parity + recorder unit tests (no browser, no network).
 *
 *   node scripts/parity_check.mjs
 *
 * 1) Replays the DOM event scripts in scripts/fixtures/feature_parity.json (exported by
 *    src/visualize_and_run.py from real KeyRecs typo-laden segments, keyboard + touch) through
 *    the browser SmartKeystrokeRecorder and checks that the clean stroke, the 17-channel model
 *    tensor, the behaviour statistics and the behaviour LLR / fused score match the Python
 *    reference implementation (src/capture_sim.py, src/behavior_profile.py).
 * 2) Unit-tests the mobile input paths that used to fail ("추출된 연속 리듬 구간이 부족합니다"):
 *    soft-keyboard events without key codes, Korean IME composition, autocorrect replacement,
 *    OS auto-repeat and Ctrl+Backspace.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { SmartKeystrokeRecorder } from '../src/recorder.js';
import { behaviorLlr, buildInputTensor, fuseScore, textToKeystrokes } from '../src/features.js';
import { validateSentenceCapture } from '../src/validation.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixtures', 'feature_parity.json'), 'utf8'));
const config = JSON.parse(readFileSync(join(here, '..', 'public', 'models', 'model_config.json'), 'utf8'));

let failures = 0;
let checks = 0;
function check(cond, msg) {
  checks += 1;
  if (!cond) {
    failures += 1;
    console.error(`  ✗ ${msg}`);
  }
}
const close = (a, b, tol) => (a === null || b === null ? a === b : Math.abs(a - b) <= tol);

// ---------------------------------------------------------------------------------------
// Minimal DOM stand-ins
// ---------------------------------------------------------------------------------------
class FakeInput extends EventTarget {
  constructor() {
    super();
    this.value = '';
  }
}

function makeEvent(type, props = {}, t = 0) {
  const e = new Event(type, { cancelable: true });
  Object.assign(e, { key: '', code: '', repeat: false, isComposing: false, ctrlKey: false, metaKey: false, altKey: false, inputType: '' }, props);
  Object.defineProperty(e, 'timeStamp', { value: t });
  return e;
}

function newRecorder() {
  const el = new FakeInput();
  const rec = new SmartKeystrokeRecorder(el, {});
  return { el, rec };
}

const keydown = (el, t, code, key, extra = {}) => el.dispatchEvent(makeEvent('keydown', { code, key, ...extra }, t));
const keyup = (el, t, code, key) => el.dispatchEvent(makeEvent('keyup', { code, key }, t));
function setValue(el, t, value, inputType) {
  el.value = value;
  el.dispatchEvent(makeEvent('input', { inputType }, t));
}

/** Physical keyboard: keydown -> value change -> keyup (holdMs later). */
function typeKey(el, t, code, ch, holdMs = 90) {
  keydown(el, t, code, ch);
  setValue(el, t, el.value + ch, 'insertText');
  keyup(el, t + holdMs, code, ch);
}

/** Soft keyboard: keydown(Unidentified, code '') -> value change, no usable keyup. */
function tapValue(el, t, nextValue, inputType = 'insertText') {
  keydown(el, t, '', 'Unidentified');
  setValue(el, t, nextValue, inputType);
  keyup(el, t + 1, '', 'Unidentified');
}

// ---------------------------------------------------------------------------------------
// 1) Fixture parity
// ---------------------------------------------------------------------------------------
console.log(`Parity fixture: ${fixture.cases.length} replay cases, ${fixture.llr_cases.length} LLR cases`);
const jsBehavior = [];
for (const c of fixture.cases) {
  const { el, rec } = newRecorder();
  for (const ev of c.script) {
    if (ev.type === 'keydown') keydown(el, ev.t, ev.code, ev.key);
    else if (ev.type === 'keyup') keyup(el, ev.t, ev.code, ev.key);
    else if (ev.op === 'insert') setValue(el, ev.t, el.value + ev.data, 'insertText');
    else if (ev.op === 'delete') setValue(el, ev.t, [...el.value].slice(0, -1).join(''), 'deleteContentBackward');
  }
  const out = rec.extractCleanStroke();
  const exp = c.expected;
  const s = out.stroke;
  const before = failures;

  check(out.cleanDigraphCount === exp.stroke.clean_digraphs, `${c.name}: clean digraphs JS=${out.cleanDigraphCount} PY=${exp.stroke.clean_digraphs}`);
  check(out.inputMode === exp.stroke.input_mode, `${c.name}: inputMode JS=${out.inputMode} PY=${exp.stroke.input_mode}`);
  const n = Math.min(s.dds.length, exp.stroke.dds.length);
  for (let i = 0; i < n; i++) {
    check(close(s.dds[i], exp.stroke.dds[i], 1e-9), `${c.name}: dd[${i}] ${s.dds[i]} vs ${exp.stroke.dds[i]}`);
    check(close(s.uds[i], exp.stroke.uds[i], 1e-9), `${c.name}: ud[${i}] ${s.uds[i]} vs ${exp.stroke.uds[i]}`);
    check(s.stitchFlags[i] === exp.stroke.stitch_flags[i], `${c.name}: stitch[${i}]`);
    check(
      s.pairKeys[i][0] === exp.stroke.pair_keys[i][0] && s.pairKeys[i][1] === exp.stroke.pair_keys[i][1],
      `${c.name}: pairKeys[${i}] ${s.pairKeys[i]} vs ${exp.stroke.pair_keys[i]}`,
    );
  }
  for (let i = 0; i < Math.min(s.holds.length, exp.stroke.holds.length); i++) {
    check(close(s.holds[i], exp.stroke.holds[i], 1e-9), `${c.name}: hold[${i}] ${s.holds[i]} vs ${exp.stroke.holds[i]}`);
    check(s.holdValid[i] === exp.stroke.hold_valid[i], `${c.name}: holdValid[${i}]`);
  }

  // Model tensor
  if (out.cleanDigraphCount > 0 && out.cleanDigraphCount === exp.stroke.clean_digraphs) {
    const { tensorData, dims } = buildInputTensor(s, config);
    const [, nCh, m] = dims;
    check(nCh === exp.tensor.length, `${c.name}: tensor channels ${nCh} vs ${exp.tensor.length}`);
    let maxErr = 0;
    for (let ch = 0; ch < nCh; ch++) {
      for (let t = 0; t < m; t++) maxErr = Math.max(maxErr, Math.abs(tensorData[ch * m + t] - exp.tensor[ch][t]));
    }
    check(maxErr < 2e-5, `${c.name}: tensor max |Δ| = ${maxErr.toExponential(2)}`);
  }

  // Behaviour statistics
  for (const [k, [sumPy, nPy]] of Object.entries(exp.behavior)) {
    const [sumJs, nJs] = out.behavior[k] || [0, 0];
    check(nJs === nPy && close(sumJs, sumPy, 1e-8), `${c.name}: behavior ${k} JS=[${sumJs}, ${nJs}] PY=[${sumPy}, ${nPy}]`);
  }
  jsBehavior.push(out.behavior);
  console.log(`  ${failures === before ? '✓' : '✗'} ${c.name}: ${out.cleanDigraphCount} digraphs, mode=${out.inputMode}, backspaces=${out.backspaceCount}`);
}

for (const lc of fixture.llr_cases) {
  const { total, parts } = behaviorLlr(jsBehavior[lc.query], jsBehavior[lc.enroll], config.behavior_profile);
  check(close(total, lc.llr, 1e-5), `LLR case ${lc.query}->${lc.enroll}: JS=${total} PY=${lc.llr}`);
  for (const [k, v] of Object.entries(lc.parts)) check(close(parts[k] ?? 0, v, 1e-5), `LLR part ${k}: JS=${parts[k]} PY=${v}`);
  const fused = fuseScore(0.6, total, fixture.beta);
  check(close(fused, lc.fused_at_0_6, 1e-6), `fused score JS=${fused} PY=${lc.fused_at_0_6}`);
  console.log(`  ✓ LLR ${lc.query}->${lc.enroll}: ${total.toFixed(4)} (fused@0.6 = ${fused.toFixed(4)})`);
}

// ---------------------------------------------------------------------------------------
// 2) Unit tests
// ---------------------------------------------------------------------------------------
console.log('Unit tests');
function unit(name, fn) {
  const before = failures;
  try {
    fn();
  } catch (err) {
    check(false, `${name}: threw ${err.stack || err}`);
  }
  console.log(`  ${failures === before ? '✓' : '✗'} ${name}`);
}

unit('Hangul syllables expand to 2-beolsik keystrokes', () => {
  check(textToKeystrokes('가각').join('') === 'ㄱㅏㄱㅏㄱ', '가각');
  check(textToKeystrokes('값').join('') === 'ㄱㅏㅂㅅ', '값 (compound final)');
  check(textToKeystrokes('ㅘ').join('') === 'ㅗㅏ', 'ㅘ (compound vowel)');
  check(textToKeystrokes('뭐').join('') === 'ㅁㅜㅓ', '뭐');
});

unit('Korean soft-keyboard composition (ㅇ→아→안→안ㄴ→안녀→안녕)', () => {
  const { el, rec } = newRecorder();
  const steps = ['ㅇ', '아', '안', '안ㄴ', '안녀', '안녕'];
  steps.forEach((v, i) => tapValue(el, 1000 + i * 180, v));
  const out = rec.extractCleanStroke();
  const codes = out.stroke.pairKeys.map((p) => p[0]).concat(out.stroke.pairKeys.length ? [out.stroke.pairKeys.at(-1)[1]] : []);
  const resolved = codes.map((k) => (k.startsWith('char:') ? config.keyboard_layout.char_to_code[k.slice(5)] : k));
  check(JSON.stringify(resolved) === JSON.stringify(['KeyD', 'KeyK', 'KeyS', 'KeyS', 'KeyU', 'KeyD']), `codes ${JSON.stringify(resolved)}`);
  check(out.cleanDigraphCount === 5, `digraphs ${out.cleanDigraphCount}`);
  check(out.inputMode === 'touch', `mode ${out.inputMode}`);
  check(out.stroke.holdValid.every((v) => v === false), 'holds flagged unknown');
});

unit('Korean sentence on a soft keyboard passes sentence validation', () => {
  const { el, rec } = newRecorder();
  const target = '오늘은 날씨가 맑아서 산책하기 좋다';
  // Re-compose syllable by syllable through the IME's intermediate states
  let t = 5000;
  let committed = '';
  for (const ch of target) {
    const cp = ch.codePointAt(0);
    if (cp >= 0xac00 && cp <= 0xd7a3) {
      const idx = cp - 0xac00;
      const cho = Math.floor(idx / 588);
      const jung = Math.floor((idx % 588) / 28);
      const jong = idx % 28;
      const states = [String.fromCodePoint(0x3131 + [0, 1, 3, 6, 7, 8, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29][cho])];
      states.push(String.fromCodePoint(0xac00 + cho * 588 + jung * 28));
      if (jong) states.push(ch);
      for (const st of states) {
        t += 170 + ((t / 7) % 60);
        tapValue(el, t, committed + st);
      }
    } else {
      t += 200;
      tapValue(el, t, committed + ch);
    }
    committed += ch;
  }
  check(el.value === target, `value ${el.value}`);
  const out = rec.extractCleanStroke();
  const validation = validateSentenceCapture(out, target);
  check(out.cleanDigraphCount >= 8, `digraphs ${out.cleanDigraphCount}`);
  check(validation.ok, `validation ${JSON.stringify(validation)}`);
  const { dims } = buildInputTensor(out.stroke, config);
  check(dims[1] === 17 && dims[2] === out.cleanDigraphCount, `tensor dims ${dims}`);
});

unit('English sentence on a soft keyboard (no key codes, keyup with keydown)', () => {
  const { el, rec } = newRecorder();
  const target = 'the quick brown fox jumps over the lazy dog';
  let t = 100;
  for (const ch of target) {
    t += 160 + ((t / 11) % 70);
    tapValue(el, t, el.value + ch);
  }
  const out = rec.extractCleanStroke();
  check(out.cleanDigraphCount >= 30, `digraphs ${out.cleanDigraphCount}`);
  check(out.inputMode === 'touch', `mode ${out.inputMode}`);
  check(validateSentenceCapture(out, target).ok, 'validation ok');
});

unit('Soft-keyboard Backspace (no Backspace keydown) is logged once and stitched', () => {
  const { el, rec } = newRecorder();
  let t = 0;
  for (const ch of 'helo') tapValue(el, (t += 180), el.value + ch);
  tapValue(el, (t += 400), 'hel', 'deleteContentBackward');
  for (const ch of 'lo world') tapValue(el, (t += 180), el.value + ch);
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 1, `backspaces ${out.backspaceCount}`);
  check(out.behavior.bs_burst_len[1] === 1, 'one correction burst');
  check(out.stroke.stitchFlags.some(Boolean), 'stitch flag set after correction');
});

unit('Autocorrect replacement → synthetic strokes, no Backspace event', () => {
  const { el, rec } = newRecorder();
  let t = 0;
  for (const ch of 'teh') tapValue(el, (t += 180), el.value + ch);
  tapValue(el, (t += 200), 'the', 'insertReplacementText');
  for (const ch of ' end') tapValue(el, (t += 180), el.value + ch);
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 0, `backspaces ${out.backspaceCount}`);
  check(rec.stack.filter((s) => s.synthetic).length === 2, `synthetic ${rec.stack.filter((s) => s.synthetic).length}`);
  // t | h* e* | ␣ e n d -> only "␣e", "en", "nd" survive (no digraph may touch a synthetic stroke)
  check(out.cleanDigraphCount === 3, `digraphs ${out.cleanDigraphCount}`);
});

unit('OS auto-repeat is not a keystroke', () => {
  const { el, rec } = newRecorder();
  typeKey(el, 0, 'KeyA', 'a');
  typeKey(el, 150, 'KeyB', 'b');
  keydown(el, 300, 'KeyC', 'c');
  setValue(el, 300, el.value + 'c', 'insertText');
  keydown(el, 800, 'KeyC', 'c', { repeat: true });
  setValue(el, 800, el.value + 'c', 'insertText');
  keydown(el, 833, 'KeyC', 'c', { repeat: true });
  setValue(el, 833, el.value + 'c', 'insertText');
  keyup(el, 850, 'KeyC', 'c');
  const out = rec.extractCleanStroke();
  check(rec.stack.length === 5 && rec.stack.filter((s) => s.synthetic).length === 2, 'repeats are synthetic');
  check(out.cleanDigraphCount === 2, `digraphs ${out.cleanDigraphCount}`);
  check(out.inputMode === 'keyboard', `mode ${out.inputMode}`);
});

unit('Ctrl+Backspace (word delete) → one Backspace event', () => {
  const { el, rec } = newRecorder();
  let t = 0;
  for (const [code, ch] of [['KeyH', 'h'], ['KeyI', 'i'], ['Space', ' '], ['KeyY', 'y'], ['KeyO', 'o']]) typeKey(el, (t += 150), code, ch);
  keydown(el, (t += 300), 'Backspace', 'Backspace', { ctrlKey: true });
  setValue(el, t, 'hi ', 'deleteWordBackward');
  keyup(el, t + 80, 'Backspace', 'Backspace');
  const out = rec.extractCleanStroke();
  const bsEvents = rec.rawEvents.filter((r) => r.kind === 'bs').length;
  check(bsEvents === 1, `bs events ${bsEvents}`);
  check(rec.stack.length === 3, `stack ${rec.stack.length}`);
  check(out.behavior.bs_burst_len[1] === 1, 'one burst');
});

unit('Physical keyboard Hangul (KeyboardEvent.code + composition)', () => {
  const { el, rec } = newRecorder();
  // 한 = ㅎ(KeyG) ㅏ(KeyK) ㄴ(KeyS); 글 = ㄱ(KeyR) ㅡ(KeyM) ㄹ(KeyF)
  const seq = [['KeyG', 'ㅎ'], ['KeyK', '하'], ['KeyS', '한'], ['KeyR', '한ㄱ'], ['KeyM', '한그'], ['KeyF', '한글']];
  let t = 0;
  for (const [code, value] of seq) {
    keydown(el, (t += 140), code, 'Process', { isComposing: true });
    setValue(el, t, value, 'insertCompositionText');
    keyup(el, t + 85, code, 'Process');
  }
  const out = rec.extractCleanStroke();
  check(out.cleanDigraphCount === 5, `digraphs ${out.cleanDigraphCount}`);
  check(out.inputMode === 'keyboard', `mode ${out.inputMode}`);
  check(JSON.stringify(out.stroke.pairKeys.map((p) => p[0])) === JSON.stringify(['KeyG', 'KeyK', 'KeyS', 'KeyR', 'KeyM']), 'codes');
  check(rec.stack.every((s) => !s.synthetic), 'no synthetic strokes');
});

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.error(`${failures} parity/unit check(s) FAILED`);
  process.exit(1);
}
