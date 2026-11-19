import assert from 'node:assert/strict';
import { SENTENCE_POOLS, SentenceDeck } from '../src/sentences.js';

function memoryStorage() {
  const records = new Map();
  return {
    getItem: (key) => records.get(key) ?? null,
    setItem: (key, value) => records.set(key, value),
  };
}

function drawMany(deck, lang, count) {
  return Array.from({ length: count }, () => deck.next(lang));
}

const storage = memoryStorage();
let deck = new SentenceDeck(storage);
const seen = {
  en: drawMany(deck, 'en', 13),
  ko: drawMany(deck, 'ko', 7),
};
seen.en.push(...drawMany(deck, 'en', 5));
deck = new SentenceDeck(storage);

for (const [lang, pool] of Object.entries(SENTENCE_POOLS)) {
  assert.equal(pool.length, 64, `${lang} pool size`);
  assert.equal(new Set(pool).size, pool.length, `${lang} pool is unique`);
  seen[lang].push(...drawMany(deck, lang, pool.length - seen[lang].length));
  assert.deepEqual(new Set(seen[lang]), new Set(pool), `${lang} deck survives language switches and reload`);
}

// A fixed shuffle puts the catalog's first entry at the head, forcing the boundary check.
const originalRandom = Math.random;
try {
  Math.random = () => 0.9999;
  const pool = SENTENCE_POOLS.en;
  const key = 'keyprint_sentence_deck_v1_en';
  for (const queue of [[], [pool[1], pool[1]], ['obsolete prompt']]) {
    const previous = memoryStorage();
    previous.setItem(key, JSON.stringify({ queue, lastPicked: pool[0] }));
    const cycle = drawMany(new SentenceDeck(previous), 'en', pool.length);
    assert.notEqual(cycle[0], pool[0], 'Do not repeat the last prompt at a restored boundary');
    assert.deepEqual(new Set(cycle), new Set(pool), 'An exhausted or outdated queue starts a fresh cycle');
  }
} finally {
  Math.random = originalRandom;
}

const brokenState = memoryStorage();
brokenState.setItem('keyprint_sentence_deck_v1_ko', '{');
const deniedStorage = {
  getItem() { throw new Error('Storage access denied'); },
  setItem() { throw new Error('Storage writes denied'); },
};
for (const storage of [brokenState, deniedStorage, null]) {
  const pool = SENTENCE_POOLS.ko;
  const cycle = drawMany(new SentenceDeck(storage), 'ko', pool.length);
  assert.deepEqual(new Set(cycle), new Set(pool), 'A fresh deck works without valid storage');
}

console.log('Sentence sampling passed for 64 distinct prompts per language, independent decks, reload, reshuffle, and unavailable storage.');
