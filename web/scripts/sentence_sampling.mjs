import assert from 'node:assert/strict';
import { textToKeystrokes } from '../src/features.js';
import {
  GENERAL_SENTENCES,
  PANGRAM_SENTENCES,
  RECENT_COOLDOWN,
  SENTENCE_POOLS,
  SentenceDeck,
} from '../src/sentences.js';

const CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
const JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
const JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
const BASIC_CONS = [...'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'];
const BASIC_VOWELS = [...'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'];
const ALL_26_2BEOLSIK = [...'ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ'];
const ALL_26_EN = [...'abcdefghijklmnopqrstuvwxyz'];

function directJamoSet(text) {
  const out = new Set();
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp >= 0xac00 && cp <= 0xd7a3) {
      const idx = cp - 0xac00;
      out.add(CHO[Math.floor(idx / 588)]);
      out.add(JUNG[Math.floor((idx % 588) / 28)]);
      const jong = JONG[idx % 28];
      if (jong) out.add(jong);
    }
  }
  return out;
}

function memoryStorage() {
  const records = new Map();
  return {
    getItem: (key) => records.get(key) ?? null,
    setItem: (key, value) => records.set(key, value),
  };
}

function drawMany(deck, lang, count, opts) {
  return Array.from({ length: count }, () => deck.next(lang, opts));
}

// 1. Catalog integrity: 96 general + 32 pangrams = 128 unique prompts per language
for (const lang of ['ko', 'en']) {
  assert.equal(GENERAL_SENTENCES[lang].length, 96, `${lang} general count`);
  assert.equal(PANGRAM_SENTENCES[lang].length, 32, `${lang} pangram count`);
  assert.equal(SENTENCE_POOLS[lang].length, 128, `${lang} total pool size`);
  assert.equal(new Set(SENTENCE_POOLS[lang]).size, 128, `${lang} pool is 100% unique`);

  const allowed = lang === 'ko' ? /^[가-힣 ]+$/ : /^[a-z ]+$/;
  for (const s of SENTENCE_POOLS[lang]) {
    assert.equal(s.trim(), s, `${lang} prompt trimmed`);
    assert.equal(/  /.test(s), false, `${lang} prompt has single spaces`);
    assert.equal(allowed.test(s), true, `${lang} prompt matches character policy: ${s}`);
  }
}

for (const s of PANGRAM_SENTENCES.ko) {
  const direct = directJamoSet(s);
  const keys = new Set(textToKeystrokes(s));
  assert.deepEqual(BASIC_CONS.filter((c) => !direct.has(c)), [], `KO pangram covers 14 basic consonants: ${s}`);
  assert.deepEqual(BASIC_VOWELS.filter((v) => !direct.has(v)), [], `KO pangram covers 10 basic vowels: ${s}`);
  assert.deepEqual(ALL_26_2BEOLSIK.filter((k) => !keys.has(k)), [], `KO pangram covers all 26 2-beolsik keys: ${s}`);
}

for (const s of PANGRAM_SENTENCES.en) {
  const chars = new Set(s);
  assert.deepEqual(ALL_26_EN.filter((c) => !chars.has(c)), [], `EN pangram covers a-z: ${s}`);
}

// 2. Full-cycle non-repetition across reloads, language switches, and multi-tab shared storage
const sharedStorage = memoryStorage();
let tabA = new SentenceDeck(sharedStorage);
let tabB = new SentenceDeck(sharedStorage);
const seen = {
  en: [...drawMany(tabA, 'en', 15), ...drawMany(tabB, 'en', 15)],
  ko: [...drawMany(tabA, 'ko', 10), ...drawMany(tabB, 'ko', 12)],
};
tabA = new SentenceDeck(sharedStorage);
for (const [lang, pool] of Object.entries(SENTENCE_POOLS)) {
  seen[lang].push(...drawMany(tabA, lang, pool.length - seen[lang].length));
  assert.equal(seen[lang].length, pool.length);
  assert.deepEqual(new Set(seen[lang]), new Set(pool), `${lang} deck survives multi-tab draws, language switches, and reload`);

  // Every 5-sentence profile enrollment window in the initial 128-card cycle mixes general + pangram prompts
  const pangramSet = new Set(PANGRAM_SENTENCES[lang]);
  for (let p = 0; p + 5 <= pool.length; p += 5) {
    const window5 = seen[lang].slice(p, p + 5);
    const pangramHits = window5.filter((s) => pangramSet.has(s)).length;
    assert.ok(pangramHits >= 1 && pangramHits <= 2, `${lang} 5-shot enrollment window has balanced pangrams (got ${pangramHits})`);
  }
}

// 3. Multi-cycle sliding cooldown guarantee across 5 consecutive 128-sentence cycles (640 draws)
for (const lang of ['ko', 'en']) {
  const deck = new SentenceDeck(memoryStorage());
  const pool = SENTENCE_POOLS[lang];
  const lastIndex = new Map();
  const minAllowedGap = pool.length - RECENT_COOLDOWN; // >= 80 indices apart => gap >= 49
  for (let step = 0; step < pool.length * 5; step++) {
    const picked = deck.next(lang);
    if (lastIndex.has(picked)) {
      const gap = step - lastIndex.get(picked);
      assert.ok(gap > RECENT_COOLDOWN, `${lang} repeat gap ${gap} must exceed cooldown ${RECENT_COOLDOWN}`);
    }
    lastIndex.set(picked, step);
  }
  assert.ok(minAllowedGap > RECENT_COOLDOWN);
}

// 4. Active sibling exclusion (`{ exclude }`) avoids simultaneous enroll/blind collision without losing prompts
{
  const deck = new SentenceDeck(memoryStorage());
  const held = deck.next('ko');
  const next = deck.next('ko', { exclude: [held] });
  assert.notEqual(next, held, 'Sibling sub-tab never receives the held prompt');
}

// 5. Boundary check under deterministic RNG + v2/v1 state migration & corrupted queue recovery
const originalRandom = Math.random;
try {
  Math.random = () => 0.9999;
  const pool = SENTENCE_POOLS.en;
  for (const key of ['keyprint_sentence_deck_v2_en', 'keyprint_sentence_deck_v1_en']) {
    for (const queue of [[], [pool[1], pool[1]], ['obsolete prompt']]) {
      const previous = memoryStorage();
      previous.setItem(key, JSON.stringify({ queue, lastPicked: pool[0] }));
      const cycle = drawMany(new SentenceDeck(previous), 'en', pool.length);
      assert.notEqual(cycle[0], pool[0], `Do not repeat the last prompt at a restored boundary (${key})`);
      assert.deepEqual(new Set(cycle), new Set(pool), `An exhausted or outdated queue starts a fresh 128-prompt cycle (${key})`);
    }
  }
} finally {
  Math.random = originalRandom;
}

const brokenState = memoryStorage();
brokenState.setItem('keyprint_sentence_deck_v2_ko', '{');
const deniedStorage = {
  getItem() { throw new Error('Storage access denied'); },
  setItem() { throw new Error('Storage writes denied'); },
};
for (const storage of [brokenState, deniedStorage, null]) {
  const pool = SENTENCE_POOLS.ko;
  const cycle = drawMany(new SentenceDeck(storage), 'ko', pool.length);
  assert.deepEqual(new Set(cycle), new Set(pool), 'A fresh deck works without valid storage');
}

console.log('Sentence sampling passed for 128 distinct prompts per language (96 general + 32 pangrams), multi-tab localStorage sync, 48-prompt cooldown, and storage fallbacks.');
