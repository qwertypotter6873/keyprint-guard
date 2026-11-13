/**
 * Offline JS <-> Python parity + recorder unit tests (no browser, no network).
 *
 *   node scripts/parity_check.mjs
 *
 * 1) Replays the DOM event scripts in scripts/fixtures/feature_parity.json (exported by
 *    src/visualize_and_run.py from real KeyRecs typo-laden segments, keyboard + touch) through
 *    the browser SmartKeystrokeRecorder and checks that the clean stroke, the 17-channel model
 *    tensor, the behaviour statistics and the behaviour LLR / fused score match the Python
 *    reference implementation (src/capture_sim.py, src/behavior_profile.py), and that the score rules
 *    (dual-encoder password score, multi-sentence query, open-set margin) match Python.
 * 2) Unit-tests the mobile input paths that used to fail ("추출된 연속 리듬 구간이 부족합니다"):
 *    soft-keyboard events without key codes, Korean IME composition, autocorrect replacement,
 *    OS auto-repeat and Ctrl+Backspace.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { SmartKeystrokeRecorder } from '../src/recorder.js';
import {
  behaviorLlr,
  buildInputTensor,
  capitalMethodLlr,
  combineEmbeddings,
  dualParamsForLength,
  dualPasswordScore,
  fuseScore,
  openSetAccept,
  textToKeystrokes,
} from '../src/features.js';
import { DEFAULT_PREPROCESS_VERSION, SCHEMA_VERSION, sha256Hex, sha256HexPureJs } from '../src/biometrics.js';
import { normalizeProfileName, validateProfileSchema } from '../src/roomSync.js';
import { composingTailIndex, computeLiveAccuracy, validateSentenceCapture } from '../src/validation.js';
import { analyzeProfileConvergence, createPresetCapture } from '../src/visualizer.js';

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

  const isV4Config = config.preprocess_version === 'kp-preproc-v4';
  const hasV4StrokeFields =
    Array.isArray(exp.stroke.h_curr) &&
    Array.isArray(exp.stroke.h_next) &&
    Array.isArray(exp.stroke.hv_curr) &&
    Array.isArray(exp.stroke.hv_next);
  if (isV4Config) {
    check(hasV4StrokeFields, `${c.name}: kp-preproc-v4 fixture must export h_curr/h_next/hv_curr/hv_next`);
  }

  check(out.cleanDigraphCount === exp.stroke.clean_digraphs, `${c.name}: clean digraphs JS=${out.cleanDigraphCount} PY=${exp.stroke.clean_digraphs}`);
  check(out.inputMode === exp.stroke.input_mode, `${c.name}: inputMode JS=${out.inputMode} PY=${exp.stroke.input_mode}`);
  const n = Math.min(s.dds.length, exp.stroke.dds.length);
  for (let i = 0; i < n; i++) {
    check(close(s.dds[i], exp.stroke.dds[i], 1e-9), `${c.name}: dd[${i}] ${s.dds[i]} vs ${exp.stroke.dds[i]}`);
    check(close(s.uds[i], exp.stroke.uds[i], 1e-9), `${c.name}: ud[${i}] ${s.uds[i]} vs ${exp.stroke.uds[i]}`);
    check(
      s.pairKeys[i][0] === exp.stroke.pair_keys[i][0] && s.pairKeys[i][1] === exp.stroke.pair_keys[i][1],
      `${c.name}: pairKeys[${i}] ${s.pairKeys[i]} vs ${exp.stroke.pair_keys[i]}`,
    );
  }

  if (hasV4StrokeFields) {
    for (let i = 0; i < n; i++) {
      check(s.stitchFlags[i] === exp.stroke.stitch_flags[i], `${c.name}: stitch[${i}] ${s.stitchFlags[i]} vs ${exp.stroke.stitch_flags[i]}`);
      check(close(s.hCurr[i], exp.stroke.h_curr[i], 1e-9), `${c.name}: hCurr[${i}] ${s.hCurr[i]} vs ${exp.stroke.h_curr[i]}`);
      check(close(s.hNext[i], exp.stroke.h_next[i], 1e-9), `${c.name}: hNext[${i}] ${s.hNext[i]} vs ${exp.stroke.h_next[i]}`);
      check(s.hvCurr[i] === exp.stroke.hv_curr[i], `${c.name}: hvCurr[${i}]`);
      check(s.hvNext[i] === exp.stroke.hv_next[i], `${c.name}: hvNext[${i}]`);
    }
    check(s.holds.length === exp.stroke.holds.length, `${c.name}: uniqueHolds len JS=${s.holds.length} PY=${exp.stroke.holds.length}`);
    for (let i = 0; i < Math.min(s.holds.length, exp.stroke.holds.length); i++) {
      check(close(s.holds[i], exp.stroke.holds[i], 1e-9), `${c.name}: hold[${i}] ${s.holds[i]} vs ${exp.stroke.holds[i]}`);
      check(s.holdValid[i] === exp.stroke.hold_valid[i], `${c.name}: holdValid[${i}]`);
    }
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
  } else {
    // Transition check when running against pre-Stage-5 legacy fixture (before visualize_and_run.py re-export)
    const legacyHolds = s.dds.length ? [s.hCurr[0], ...s.hNext] : [];
    const legacyHoldValid = s.dds.length ? [s.hvCurr[0], ...s.hvNext] : [];
    check(legacyHolds.length === exp.stroke.holds.length, `${c.name}: legacy endpoints len`);
    for (let i = 0; i < Math.min(legacyHolds.length, exp.stroke.holds.length); i++) {
      check(close(legacyHolds[i], exp.stroke.holds[i], 1e-9), `${c.name}: legacy hold[${i}]`);
      check(legacyHoldValid[i] === exp.stroke.hold_valid[i], `${c.name}: legacy holdValid[${i}]`);
    }
    if (out.cleanDigraphCount > 0 && out.cleanDigraphCount === exp.stroke.clean_digraphs) {
      const { tensorData: v4Data, dims: v4Dims } = buildInputTensor(s, config);
      check(
        v4Dims[1] === exp.tensor.length && v4Dims[2] === out.cleanDigraphCount && v4Data.every(Number.isFinite),
        `${c.name}: v4 tensor finite [${v4Dims.join('x')}]`,
      );
      const legacyStroke = {
        holds: exp.stroke.holds,
        holdValid: exp.stroke.hold_valid,
        dds: s.dds,
        uds: s.uds,
        stitchFlags: exp.stroke.stitch_flags,
        pairKeys: s.pairKeys,
      };
      const { tensorData, dims } = buildInputTensor(legacyStroke, config);
      const [, nCh, m] = dims;
      let maxErr = 0;
      for (let ch = 0; ch < nCh; ch++) {
        for (let t = 0; t < m; t++) maxErr = Math.max(maxErr, Math.abs(tensorData[ch * m + t] - exp.tensor[ch][t]));
      }
      check(maxErr < 2e-5, `${c.name}: legacy fallback tensor max |Δ| = ${maxErr.toExponential(2)}`);
    }
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

for (const sc of fixture.score_cases || []) {
  const r = dualPasswordScore(sc.q9, sc.q17, sc.t9, sc.t17, sc.prm, sc.slope);
  check(close(r.z, sc.z, 1e-6), `dual password z (${sc.mode}): JS=${r.z} PY=${sc.z}`);
  check(close(r.similarity, sc.similarity, 1e-7), `dual password similarity (${sc.mode}): JS=${r.similarity} PY=${sc.similarity}`);
}
if (fixture.score_cases?.length) console.log(`  ✓ dual password score: ${fixture.score_cases.length} cases (keyboard + touch parameters)`);
if (fixture.combine_case) {
  const got = combineEmbeddings(fixture.combine_case.inputs);
  check(got.every((v, i) => close(v, fixture.combine_case.expected[i], 1e-9)), 'two-sentence query embedding (mean, re-normalised)');
  console.log('  ✓ two-sentence query embedding');
}
for (const oc of fixture.open_set_cases || []) {
  check(openSetAccept(oc.scores, oc.threshold, oc.margin) === oc.accept, `open-set rule ${JSON.stringify(oc)}`);
}
if (fixture.open_set_cases?.length) console.log(`  ✓ open-set threshold + margin rule: ${fixture.open_set_cases.length} cases`);
for (const lc of fixture.length_cases || []) {
  const got = dualParamsForLength(config.password_dual, lc.mode === 'touch' ? 'touch' : 'keyboard', lc.length);
  for (const [k, v] of Object.entries(lc.expected)) check(close(got[k], v, 1e-9), `length ${lc.length} ${lc.mode}: ${k} JS=${got[k]} PY=${v}`);
}
if (fixture.length_cases?.length) console.log(`  ✓ length-aware dual parameters: ${fixture.length_cases.length} cases (interpolated / clamped)`);
for (const cc of fixture.capital_cases || []) {
  const got = capitalMethodLlr(cc.q_caps, cc.n_caps, cc.n_total, cc.caps_share);
  check(close(got, cc.llr, 1e-9), `capital-method LLR ${JSON.stringify(cc)} JS=${got}`);
}
if (fixture.capital_cases?.length) console.log(`  ✓ capital-letter (Shift vs CapsLock) LLR: ${fixture.capital_cases.length} cases`);
check(Boolean(config.password_dual?.keyboard && config.password_dual?.touch), 'model_config.json has dual password parameters');
check(config.password_min_length === 12 && Array.isArray(config.password_dual?.by_length), 'password: 12-char minimum + length-aware calibration in config');
check(config.arena_enroll_sentences === 5 && config.arena_query_sentences === 2, 'typing practice protocol: 5 enrolment / 2 query sentences');

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
  // (timestamps start at 1000: a timeStamp of 0 means "unknown" and falls back to performance.now())
  typeKey(el, 1000, 'KeyA', 'a');
  typeKey(el, 1150, 'KeyB', 'b');
  keydown(el, 1300, 'KeyC', 'c');
  setValue(el, 1300, el.value + 'c', 'insertText');
  keydown(el, 1800, 'KeyC', 'c', { repeat: true });
  setValue(el, 1800, el.value + 'c', 'insertText');
  keydown(el, 1833, 'KeyC', 'c', { repeat: true });
  setValue(el, 1833, el.value + 'c', 'insertText');
  keyup(el, 1850, 'KeyC', 'c');
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


// ---- Hangul composition must never look like a typo ------------------------------------------
// Value after each tap while typing "안녕하세요" (the final consonant moves on at 핫→하세, 셍→세요)
const KO_STEPS = ['ㅇ', '아', '안', '안ㄴ', '안녀', '안녕', '안녕ㅎ', '안녕하', '안녕핫', '안녕하세', '안녕하셍', '안녕하세요'];

unit('iOS / Samsung keyboard: each composition step as delete + insert is not a Backspace', () => {
  const { el, rec } = newRecorder();
  let t = 2000;
  let prevVal = '';
  for (const v of KO_STEPS) {
    t += 190;
    keydown(el, t, '', 'Unidentified');
    if (v.startsWith(prevVal) && v.length > prevVal.length) {
      setValue(el, t, v, 'insertText');
    } else {
      setValue(el, t, prevVal.slice(0, -1), 'deleteContentBackward'); // the IME removes the composing syllable…
      setValue(el, t + 0.6, v, 'insertText'); // …and writes the new one right away
    }
    keyup(el, t + 1, '', 'Unidentified');
    prevVal = v;
  }
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 0, `backspaces ${out.backspaceCount}`);
  check(rec.rawEvents.filter((e) => e.kind === 'bs').length === 0, 'no Backspace in the behaviour log');
  check(out.cleanDigraphCount === 11 && !out.stroke.stitchFlags.some(Boolean), `digraphs ${out.cleanDigraphCount}, stitches ${out.stroke.stitchFlags.filter(Boolean).length}`);
  check(rec.stack.every((x) => !x.synthetic), 'no synthetic strokes');
});

unit('Physical keyboard: final consonant moving on (간 → 가 → 가나) is not a deletion', () => {
  const { el, rec } = newRecorder();
  // 가나다 = ㄱ(KeyR) ㅏ(KeyK) ㄴ(KeyS) ㅏ(KeyK) ㄷ(KeyE) ㅏ(KeyK); the IME passes through "가" and "가나"
  const seq = [['KeyR', ['ㄱ']], ['KeyK', ['가']], ['KeyS', ['간']], ['KeyK', ['가', '가나']], ['KeyE', ['가낟']], ['KeyK', ['가나', '가나다']]];
  let t = 0;
  for (const [code, values] of seq) {
    keydown(el, (t += 150), code, 'Process', { isComposing: true });
    values.forEach((v, i) => setValue(el, t + i * 0.3, v, 'insertCompositionText'));
    keyup(el, t + 80, code, 'Process');
  }
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 0, `backspaces ${out.backspaceCount}`);
  check(rec.stack.length === 6 && rec.stack.every((x) => x.source === 'key' && !x.synthetic), `strokes ${rec.stack.map((x) => x.source).join(',')}`);
  check(rec.stack.map((x) => x.label).join('') === 'ㄱㅏㄴㅏㄷㅏ', `labels show the typed jamo, not the Latin key names: ${rec.stack.map((x) => x.label).join(' ')}`);
  check(out.cleanDigraphCount === 5 && out.inputMode === 'keyboard', `digraphs ${out.cleanDigraphCount}, mode ${out.inputMode}`);
});

unit('Safari: deleteCompositionText + insertFromComposition on commit is not a Backspace', () => {
  const { el, rec } = newRecorder();
  const seq = [['KeyG', 'ㅎ'], ['KeyK', '하'], ['KeyS', '한'], ['KeyR', '한ㄱ'], ['KeyM', '한그'], ['KeyF', '한글']];
  let t = 0;
  let prevVal = '';
  for (const [code, value] of seq) {
    keydown(el, (t += 140), code, 'Process', { isComposing: true });
    if (value.length > prevVal.length && prevVal.length > 0) {
      // composition of the previous syllable ends: Safari deletes and re-inserts it before the new one
      setValue(el, t, prevVal.slice(0, -1), 'deleteCompositionText');
      setValue(el, t + 0.2, prevVal, 'insertFromComposition');
    }
    setValue(el, t + 0.4, value, 'insertCompositionText');
    keyup(el, t + 85, code, 'Process');
    prevVal = value;
  }
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 0, `backspaces ${out.backspaceCount}`);
  check(out.cleanDigraphCount === 5 && rec.stack.every((x) => !x.synthetic), `digraphs ${out.cleanDigraphCount}`);
});

unit('Live preview (principles tab) peeking mid-composition does not settle the IME edit', () => {
  const { el, rec } = newRecorder();
  let t = 3000;
  let prevVal = '';
  for (const v of KO_STEPS) {
    t += 190;
    keydown(el, t, '', 'Unidentified');
    if (v.startsWith(prevVal) && v.length > prevVal.length) {
      setValue(el, t, v, 'insertText');
    } else {
      setValue(el, t, prevVal.slice(0, -1), 'deleteContentBackward');
      rec.extractCleanStroke({ resolvePending: false }); // what the live piano roll does on every input event
      setValue(el, t + 0.6, v, 'insertText');
    }
    rec.extractCleanStroke({ resolvePending: false });
    keyup(el, t + 1, '', 'Unidentified');
    prevVal = v;
  }
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 0 && out.cleanDigraphCount === 11, `backspaces ${out.backspaceCount}, digraphs ${out.cleanDigraphCount}`);
});

unit('A real soft-keyboard deletion followed by normal typing still counts once', () => {
  const { el, rec } = newRecorder();
  let t = 0;
  for (const ch of 'helo') tapValue(el, (t += 170), el.value + ch);
  setValue(el, (t += 160), 'hel', 'deleteContentBackward');
  for (const ch of 'lo') tapValue(el, (t += 170), el.value + ch);
  const out = rec.extractCleanStroke();
  check(out.backspaceCount === 1, `backspaces ${out.backspaceCount}`);
});

unit('Live display: a syllable still being composed is not a typo', () => {
  check(composingTailIndex('아', '안녕') === 0 && computeLiveAccuracy('아', '안녕') === 100, 'ㅇ+ㅏ on the way to 안');
  check(composingTailIndex('간', '가나다') === 0 && computeLiveAccuracy('간', '가나다') === 100, '간 before the ㄴ moves to 나');
  check(composingTailIndex('가나닥', '가나다라') === -1 && computeLiveAccuracy('가나닥', '가나다라') === 67, 'a real typo still counts');
  check(composingTailIndex('hel', 'hello') === -1 && computeLiveAccuracy('hex', 'hello') === 67, 'Latin text unchanged');
});

// ---- v4 Frozen Stitch, Tensor, Stability & Schema Security Contracts --------------------------
unit('v4 Contract (1): Backspace stitch preserves per-transition (hCurr, hNext) & zeroes Local_Accel', () => {
  const { el, rec } = newRecorder();
  // A(h=0.08) -> B(h=0.06), typo X + Backspace, C(h=0.20) -> D(h=0.09)
  typeKey(el, 1000, 'KeyA', 'a', 80);
  typeKey(el, 1180, 'KeyB', 'b', 60);
  typeKey(el, 1360, 'KeyX', 'x', 70);
  keydown(el, 1560, 'Backspace', 'Backspace');
  setValue(el, 1560, 'ab', 'deleteContentBackward');
  keyup(el, 1620, 'Backspace', 'Backspace');
  typeKey(el, 1820, 'KeyC', 'c', 200);
  typeKey(el, 2100, 'KeyD', 'd', 90);

  const out = rec.extractCleanStroke();
  const s = out.stroke;
  check(out.cleanDigraphCount === 2, `digraphs ${out.cleanDigraphCount}`);
  check(close(s.hCurr[0], 0.08, 1e-9) && close(s.hNext[0], 0.06, 1e-9), `transition 0 (${s.hCurr[0]}, ${s.hNext[0]})`);
  check(close(s.hCurr[1], 0.20, 1e-9) && close(s.hNext[1], 0.09, 1e-9), `transition 1 (${s.hCurr[1]}, ${s.hNext[1]})`);
  check(s.stitchFlags[0] === false && s.stitchFlags[1] === true, `stitchFlags ${JSON.stringify(s.stitchFlags)}`);
  check(
    s.holds.length === 4 &&
      close(s.holds[0], 0.08, 1e-9) &&
      close(s.holds[1], 0.06, 1e-9) &&
      close(s.holds[2], 0.20, 1e-9) &&
      close(s.holds[3], 0.09, 1e-9),
    `uniqueHolds ${JSON.stringify(s.holds)}`,
  );
  // UI holdMs uses observed uniqueHolds mean: (80+60+200+90)/4 = 107.5 -> 108
  check(out.holdMs === 108, `UI holdMs=${out.holdMs} expected 108`);

  const { tensorData, dims } = buildInputTensor(s, config);
  const m = dims[2];
  check(m === 2, `tensor m=${m}`);
  const expectedScaledZeroAccel = (0.0 - config.channel_mean[8]) / config.channel_std[8];
  check(close(tensorData[8 * m + 0], expectedScaledZeroAccel, 1e-6), 'Local_Accel[0] === 0.0 (unscaled)');
  check(close(tensorData[8 * m + 1], expectedScaledZeroAccel, 1e-6), 'Local_Accel[1] === 0.0 across Backspace stitch');
  const expectedScaledHC = (0.20 - config.channel_mean[0]) / config.channel_std[0];
  check(close(tensorData[0 * m + 1], expectedScaledHC, 1e-6), 'channel 0 at t=1 uses hC=0.20 (NOT hB=0.06)');

  // Preset capture in visualizer.js also preserves ㄴ (0.086s) after stitch
  const preset = createPresetCapture('normal');
  check(
    preset.stroke.hCurr.some((h) => close(h, 0.086, 1e-6)) && preset.stroke.holds.some((h) => close(h, 0.086, 1e-6)),
    'visualizer createPresetCapture preserves ㄴ hold after stitch',
  );
});

unit('v4 Contract (2): Skipped outlier transition (dd > 2.2s or hold > MAX_HOLD) sets stitchFlags=true & zeroes Local_Accel', () => {
  // Case A: long pause dd > 2.2s between B and C without Backspace
  const { el, rec } = newRecorder();
  typeKey(el, 1000, 'KeyA', 'a', 80);
  typeKey(el, 1180, 'KeyB', 'b', 70);
  typeKey(el, 4000, 'KeyC', 'c', 150); // dd(B->C) = 2.82s > 2.2s -> skipped!
  typeKey(el, 4240, 'KeyD', 'd', 85);  // dd(C->D) = 0.24s -> accepted as new segment!
  const out = rec.extractCleanStroke();
  const s = out.stroke;
  check(out.cleanDigraphCount === 2, `pause skip digraphs=${out.cleanDigraphCount}`);
  check(s.stitchFlags[0] === false && s.stitchFlags[1] === true, `pause skip stitchFlags=${JSON.stringify(s.stitchFlags)}`);
  check(close(s.hCurr[1], 0.15, 1e-9) && close(s.hNext[1], 0.085, 1e-9), `pause skip hCurr[1]=${s.hCurr[1]}`);
  const { tensorData, dims } = buildInputTensor(s, config);
  const zeroAccel = (0.0 - config.channel_mean[8]) / config.channel_std[8];
  check(close(tensorData[8 * dims[2] + 1], zeroAccel, 1e-6), 'Local_Accel[1] === 0.0 after skipped pause transition');

  // Case B: MAX_HOLD (> 1.2s) on middle key X skips B->X and X->C, setting stitchFlags=true on C->D
  const { el: el2, rec: rec2 } = newRecorder();
  typeKey(el2, 1000, 'KeyA', 'a', 80);
  typeKey(el2, 1180, 'KeyB', 'b', 75);
  typeKey(el2, 1360, 'KeyX', 'x', 1350); // hold = 1.35s > 1.2s -> B->X and X->C both skipped!
  typeKey(el2, 2800, 'KeyC', 'c', 120);
  typeKey(el2, 2990, 'KeyD', 'd', 90);
  const out2 = rec2.extractCleanStroke();
  check(out2.cleanDigraphCount === 2, `MAX_HOLD skip digraphs=${out2.cleanDigraphCount}`);
  check(out2.stroke.stitchFlags[1] === true, 'stitchFlags[1] === true after MAX_HOLD outlier');
  check(close(out2.stroke.hCurr[1], 0.12, 1e-9), `hCurr[1]=${out2.stroke.hCurr[1]}`);
});

unit('v4 Contract (3): Mixed keyboard+touch endpoint imputation, Hold_Valid mask, model meanH vs UI holdMs', () => {
  // Touch pure capture -> UI holdMs === 0
  const { el: touchEl, rec: touchRec } = newRecorder();
  ['a', 'ab', 'abc', 'abcd'].forEach((v, idx) => tapValue(touchEl, 1000 + idx * 170, v));
  const touchOut = touchRec.extractCleanStroke();
  check(touchOut.inputMode === 'touch' && touchOut.holdMs === 0, `pure touch holdMs=${touchOut.holdMs}`);

  // Mixed synthetic stroke: 2 transitions across 3 unique keys [k0(valid 0.15s), k1(touch null), k2(valid 0.21s)]
  // Transition 0: k0->k1 (hvCurr=true, hvNext=false)
  // Transition 1: k1->k2 (hvCurr=false, hvNext=true)
  const bounds = config.outlier_bounds || { H: [0.008, 0.45], DD: [0.008, 1.4], UD: [-0.35, 1.4] };
  const clipVal = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
  const hImp = clipVal(config.touch_hold_impute_sec ?? 0.09, bounds.H);

  const mixedStroke = {
    holds: [0.15, null, 0.21],
    holdValid: [true, false, true],
    hCurr: [0.15, null],
    hNext: [null, 0.21],
    hvCurr: [true, false],
    hvNext: [false, true],
    dds: [0.22, 0.25],
    uds: [0.07, null],
    stitchFlags: [false, false],
    pairKeys: [['KeyA', 'KeyB'], ['KeyB', 'KeyC']],
  };
  const { tensorData, dims, cleanStroke } = buildInputTensor(mixedStroke, config);
  const m = dims[2];
  check(m === 2, `mixed m=${m}`);
  check(close(cleanStroke.hCurr[0], 0.15, 1e-9) && close(cleanStroke.hNext[0], hImp, 1e-9), 't=0 imputes hNext only');
  check(close(cleanStroke.hCurr[1], hImp, 1e-9) && close(cleanStroke.hNext[1], 0.21, 1e-9), 't=1 imputes hCurr only');
  check(close(cleanStroke.uds[1], clipVal(0.25 - hImp, bounds.UD), 1e-9), 't=1 imputes UD as clip(dd - hImp)');

  // Hold_Valid (channel 16) requires BOTH endpoints observed -> [0.0, 0.0]
  const rawHoldValid0 = tensorData[16 * m + 0] * config.channel_std[16] + config.channel_mean[16];
  const rawHoldValid1 = tensorData[16 * m + 1] * config.channel_std[16] + config.channel_mean[16];
  check(close(rawHoldValid0, 0.0, 1e-6) && close(rawHoldValid1, 0.0, 1e-6), `Hold_Valid=[${rawHoldValid0}, ${rawHoldValid1}]`);

  // Model meanH is computed on imputed+clipped uniqueHolds: (0.15 + hImp + 0.21) / 3
  const expectedMeanH = (0.15 + hImp + 0.21) / 3;
  const rawRelH0 = tensorData[4 * m + 0] * config.channel_std[4] + config.channel_mean[4];
  check(close(rawRelH0, 0.15 / expectedMeanH, 1e-5), `Rel_H[0]=${rawRelH0} vs ${0.15 / expectedMeanH}`);

  // Empty m=0 stroke returns safe empty Float32Array
  const emptyRes = buildInputTensor({ holds: [], dds: [], uds: [] }, config);
  check(emptyRes.dims[2] === 0 && emptyRes.tensorData.length === 0, 'm=0 safe empty tensor');
});

unit('v4 Contract (4): Clipping boundary preserves raw UD = rawDD - rawH before UD clip', () => {
  // rawH = hiH + 0.15 (> bounds.H[1]), rawDD = rawH + 0.05 (within bounds.DD), so rawUD = 0.05.
  // Buggy clip(DD) - clip(H) would give (hiH + 0.20) - hiH = 0.20 instead of 0.05!
  const hiH = (config.outlier_bounds?.H || [0.008, 0.45])[1];
  const rawH = hiH + 0.15;
  const rawDD = rawH + 0.05;
  const boundaryStroke = {
    holds: [rawH, 0.08],
    holdValid: [true, true],
    hCurr: [rawH],
    hNext: [0.08],
    hvCurr: [true],
    hvNext: [true],
    dds: [rawDD],
    uds: [0.05],
    stitchFlags: [false],
    pairKeys: [['KeyA', 'KeyB']],
  };
  const { tensorData, cleanStroke } = buildInputTensor(boundaryStroke, config);
  check(close(cleanStroke.hCurr[0], hiH, 1e-9), `hCurr clipped to hiH=${hiH}: ${cleanStroke.hCurr[0]}`);
  check(close(cleanStroke.uds[0], 0.05, 1e-9), `uds preserved raw 0.05 (not 0.20): ${cleanStroke.uds[0]}`);
  const unscaledUd = tensorData[3] * config.channel_std[3] + config.channel_mean[3];
  check(close(unscaledUd, 0.05, 1e-6), `channel 3 UD=${unscaledUd}`);
});

unit('v4 Contract (5): analyzeProfileConvergence handles K=1, zero/NaN, opposite [v,-v], identical [v,v]', () => {
  const v = Array.from({ length: 64 }, (_, i) => (i === 0 ? 1.0 : 0.0));
  const negV = v.map((x) => -x);
  const zeroV = new Array(64).fill(0);
  const nanV = v.map((x, i) => (i === 3 ? Number.NaN : x));

  const r1 = analyzeProfileConvergence({ exemplars: [v] }, 1);
  check(
    r1.status === 'insufficient' &&
      r1.meanPairwiseCos === null &&
      r1.stabilityPct === null &&
      r1.stabilityText === '근거 부족 (2개 이상 필요)',
    `K=1: ${JSON.stringify(r1)}`,
  );

  const rZero = analyzeProfileConvergence({ exemplars: [v, zeroV] }, 2);
  check(
    rZero.status === 'invalid' && rZero.meanPairwiseCos === null && rZero.stabilityPct === null && rZero.stabilityText === '무효 벡터',
    `zero vector: ${JSON.stringify(rZero)}`,
  );

  const rNan = analyzeProfileConvergence([v, nanV]);
  check(rNan.status === 'invalid' && rNan.stabilityPct === null, `NaN vector: ${JSON.stringify(rNan)}`);

  const rOpp = analyzeProfileConvergence({ exemplars: [v, negV] }, 2);
  check(
    rOpp.status === 'ok' && close(rOpp.meanPairwiseCos, -1.0, 1e-9) && rOpp.stabilityPct === 0,
    `[v, -v]: ${JSON.stringify(rOpp)}`,
  );

  const rIdent = analyzeProfileConvergence([v, v]);
  check(
    rIdent.status === 'ok' && close(rIdent.meanPairwiseCos, 1.0, 1e-9) && rIdent.stabilityPct === 100,
    `[v, v]: ${JSON.stringify(rIdent)}`,
  );
});

unit('v4 Contract (6): validateProfileSchema & SHA-256 pure-JS parity', () => {
  // Pure-JS SHA-256 parity against node:crypto
  for (const msg of ['', 'abc', 'keyprint-v4:sha17:sha9', '가천대학교 KeyPrint Guard 2026!']) {
    const bytes = new TextEncoder().encode(msg);
    const expectedHex = createHash('sha256').update(bytes).digest('hex');
    const gotPure = sha256HexPureJs(bytes);
    check(gotPure === expectedHex, `sha256HexPureJs("${msg}") = ${gotPure}`);
  }

  const compat = {
    schemaVersion: SCHEMA_VERSION,
    preprocessVersion: DEFAULT_PREPROCESS_VERSION,
    modelHash: '0123456789abcdef',
  };
  const unitVec = Array.from({ length: 64 }, (_, i) => (i === 0 ? 0.8 : i === 1 ? 0.6 : 0.0));
  const zeroVec = new Array(64).fill(0);

  const validArenaOnly = {
    id: 'user_arena_01',
    name: '김가천',
    isSample: false,
    color: '#2563eb',
    createdAt: 1700000000000,
    updatedAt: 1700000001000,
    ...compat,
    arena: {
      ko: {
        centroid: unitVec,
        exemplars: [unitVec],
        count: 5,
        inputMode: 'keyboard',
        cpm: 320,
        holdMs: 85,
        rolloverPct: 24,
        behavior: { bs_rate: [0.05, 5] },
      },
      en: null,
    },
    password: null,
  };
  const resArena = validateProfileSchema(validArenaOnly, compat);
  check(resArena.ok === true && resArena.profile.name === '김가천', `valid arena-only: ${JSON.stringify(resArena.reason)}`);

  const validPasswordUser = {
    id: 'user_pw_01',
    name: '보안계정A',
    isSample: false,
    color: '#0d9488',
    createdAt: 1700000000000,
    updatedAt: 1700000002000,
    ...compat,
    arena: { ko: null, en: null },
    password: {
      target: 'GachonUniv2026!',
      centroid: unitVec,
      exemplars: [unitVec, unitVec],
      enc17: { centroid: unitVec, exemplars: [unitVec, unitVec] }, // minimal enc17 without count/inputMode
      count: 4,
      inputMode: 'keyboard',
      capsUse: [2, 2],
    },
  };
  const resPw = validateProfileSchema(validPasswordUser, compat);
  check(resPw.ok === true, `valid password user: ${resPw.reason}`);

  const validSampleShortPw = {
    ...validPasswordUser,
    id: 'sample_01',
    name: '서연 (샘플·고속연타)',
    isSample: true,
    password: { ...validPasswordUser.password, target: 'gachon2026' }, // 10 chars allowed for isSample
  };
  check(validateProfileSchema(validSampleShortPw, compat).ok === true, 'isSample allows 10-char demo password');

  // Adversarial / malformed cases must return { ok: false } and NEVER throw
  const badInputs = [
    [null, 'null root'],
    [undefined, 'undefined root'],
    [42, 'primitive root'],
    [[1, 2, 3], 'array root'],
    [{ ...validArenaOnly, name: null }, 'name: null'],
    [{ ...validArenaOnly, name: '   \x00\x07  ' }, 'control-char empty name'],
    [{ ...validArenaOnly, id: '../bad/id<script>' }, 'malicious id'],
    [{ ...validArenaOnly, color: 'red" onload="alert(1)' }, 'XSS color attribute'],
    [{ ...validPasswordUser, isSample: false, password: { ...validPasswordUser.password, target: 'short10chr' } }, 'user password < 12 chars'],
    [{ ...validArenaOnly, arena: { ko: { ...validArenaOnly.arena.ko, enc17: { centroid: unitVec, exemplars: [unitVec] } } } }, 'enc17 injected into arena'],
    [{ ...validArenaOnly, arena: { ko: { ...validArenaOnly.arena.ko, target: 'injected' } } }, 'target injected into arena'],
    [{ ...validArenaOnly, arena: { ko: { ...validArenaOnly.arena.ko, centroid: zeroVec } } }, 'zero-norm centroid'],
    [{ ...validArenaOnly, arena: { ko: { ...validArenaOnly.arena.ko, centroid: unitVec.map((x, i) => (i === 0 ? Number.NaN : x)) } } }, 'NaN in centroid'],
    [{ ...validArenaOnly, arena: { ko: null, en: null }, password: null }, 'zero templates'],
    [{ ...validArenaOnly, behavior: { unknown_key: [1, 1] }, arena: { ko: { ...validArenaOnly.arena.ko, behavior: { bad_feat: [1, 1] } } } }, 'unknown behavior key'],
    [{ ...validArenaOnly, schemaVersion: 3 }, 'wrong schemaVersion (v3)'],
    [{ ...validArenaOnly, preprocessVersion: 'kp-preproc-v3' }, 'wrong preprocessVersion'],
    [{ ...validArenaOnly, modelHash: 'deadbeefdeadbeef' }, 'wrong modelHash'],
    [JSON.parse('{"__proto__":{"polluted":true},"id":"x","name":"x"}'), 'prototype pollution JSON'],
  ];

  for (const [bad, label] of badInputs) {
    let res;
    try {
      res = validateProfileSchema(bad, compat);
    } catch (err) {
      check(false, `validateProfileSchema threw on ${label}: ${err}`);
      continue;
    }
    check(res && res.ok === false, `validateProfileSchema rejected ${label}`);
  }
  check(({}).polluted === undefined, 'Object.prototype not polluted');
  check(normalizeProfileName(null) === '' && normalizeProfileName('  홍길동\x00 ') === '홍길동', 'normalizeProfileName safe');
});

// Verify async sha256Hex matches Node crypto too
const asyncDigest = await sha256Hex(new TextEncoder().encode('keyprint-guard-v4'));
check(
  asyncDigest === createHash('sha256').update('keyprint-guard-v4').digest('hex'),
  `async sha256Hex parity: ${asyncDigest}`,
);

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.error(`${failures} parity/unit check(s) FAILED`);
  process.exit(1);
}
