/** Functional state transitions with in-memory storage and a mocked broker; no external traffic. */
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mock } from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { ActionGate } from '../src/asyncActions.js';
import { RoomProfileSync } from '../src/roomSync.js';

const gate = new ActionGate();
let release;
const pending = new Promise((resolve) => { release = resolve; });
let applied = 0;
const first = gate.run(async (isCurrent) => { await pending; if (isCurrent()) applied += 1; });
assert.equal(await gate.run(async () => { applied += 1; }), false);
gate.invalidate();
release();
await first;
assert.equal(applied, 0, 'Changing workspace discards the previous asynchronous result');
await gate.run(async (isCurrent) => { if (isCurrent()) applied += 1; });
assert.equal(applied, 1);
await assert.rejects(gate.run(async () => { throw new Error('simulated inference failure'); }));
assert.equal(gate.pending, false, 'An inference error releases the submission gate');

const storage = new Map();
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    get length() { return storage.size; },
    key: (i) => Array.from(storage.keys())[i] ?? null,
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
  },
});
const compat = { schemaVersion: 4, preprocessVersion: 'kp-preproc-v4', modelHash: 'functional-fixture' };
const vec = Array.from({ length: 64 }, (_, i) => i === 0 ? 1 : 0);
const profile = (id, count = 1) => ({ id, name: id, isSample: false, color: '#2563eb', arena: { en: { centroid: vec, exemplars: [vec], count, inputMode: 'keyboard' } } });
const matches = (filter, topic) => {
  const path = topic.split('/');
  const parts = filter.split('/');
  return parts.length === path.length && parts.every((part, i) => part === '+' || part === path[i]);
};

class MockBroker {
  clients = [];
  retained = new Map();
  published = [];

  open = (url) => {
    const client = new EventEmitter();
    client.url = url;
    client.connected = false;
    client.accept = () => { client.connected = true; client.emit('connect'); };
    client.end = () => { client.connected = false; client.emit('close'); };
    client.subscribe = (filters, options, callback) => {
      client.filters = filters;
      client.finishSubscribe = () => {
        for (const [topic, payload] of this.retained) {
          if (filters.some((filter) => matches(filter, topic))) client.emit('message', topic, payload, { retain: true });
        }
        callback();
      };
      if (!client.holdSubscribe) client.finishSubscribe();
    };
    client.publish = (topic, payload, options, callback) => {
      assert.ok(client.connected);
      this.published.push({ topic, payload, options });
      if (options.retain && payload) this.retained.set(topic, payload);
      if (options.retain && !payload) this.retained.delete(topic);
      for (const other of this.clients) {
        if (other.connected && other.filters?.some((filter) => matches(filter, topic))) {
          other.emit('message', topic, payload, { retain: false });
        }
      }
      if (client.holdAcknowledgment && callback && topic.includes('/profiles/')) client.finishPublish = callback;
      else callback?.();
    };
    this.clients.push(client);
    return client;
  };

  get newest() { return this.clients.at(-1); }
}

async function eventually(predicate, message) {
  const cutoff = Date.now() + 2500;
  while (!predicate() && Date.now() < cutoff) await setImmediate();
  assert.ok(predicate(), message);
}

const instances = [];
const makeRoom = (openMqtt) => {
  const sync = new RoomProfileSync({ ...compat, openMqtt, brokers: ['ws://mock.test/first', 'ws://mock.test/second'] });
  instances.push(sync);
  return sync;
};
mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
try {
  const broker = new MockBroker();
  const firstTab = makeRoom(broker.open);
  firstTab.connect('INVITED');
  assert.equal(broker.clients.length, 0);
  assert.equal(firstTab.shared, false);
  firstTab.upsertProfile(profile('local_person'));
  firstTab.connect('TEAM', { shared: true });
  assert.equal(firstTab.getProfiles().length, 0, 'Local profiles are not copied into a shared room');

  const secondTab = makeRoom(broker.open);
  secondTab.connect('TEAM', { shared: true });
  firstTab.upsertProfile(profile('shared_person'));
  await eventually(() => secondTab.pendingUpdates.has('shared_person'), 'An idle tab learns that the edit is queued');
  assert.equal(secondTab.getProfiles().length, 1);
  firstTab.disconnect();
  secondTab.disconnect();

  const reload = makeRoom(broker.open);
  reload.connect('TEAM', { shared: true });
  assert.ok(reload.pendingUpdates.has('shared_person'), 'The local queue survives both offline tabs closing');
  broker.newest.accept();
  assert.equal(reload.pendingUpdates.size, 0);
  assert.ok([...broker.retained.keys()].some((topic) => topic.includes('/profiles/shared_person/')));
  reload.disconnect();

  const live = makeRoom(broker.open);
  live.connect('TEAM', { shared: true });
  assert.equal(live.pendingUpdates.size, 0, 'A broker acknowledgment survives reopening');
  broker.newest.accept();
  live.upsertProfile(profile('shared_person', 2));
  const current = live.upsertProfile(profile('shared_person', 3));
  assert.deepEqual([...broker.retained.keys()].filter((topic) => topic.includes('/profiles/shared_person/')), [live._profileRevisionTopic(current)]);
  live.connect('LOCAL');
  assert.deepEqual(live.getProfiles().map((p) => p.id), ['local_person']);

  const alreadyOnlineA = makeRoom(broker.open);
  const alreadyOnlineB = makeRoom(broker.open);
  alreadyOnlineA.connect('LIVE_EDIT', { shared: true });
  broker.newest.accept();
  alreadyOnlineB.connect('LIVE_EDIT', { shared: true });
  broker.newest.accept();
  const oldLive = alreadyOnlineA.upsertProfile(profile('live_edit'));
  assert.deepEqual(alreadyOnlineB.getProfiles(), [oldLive]);
  const newLive = alreadyOnlineB.upsertProfile(profile('live_edit', 2));
  assert.deepEqual([...broker.retained.keys()].filter((topic) => topic.includes('/profiles/live_edit/')), [alreadyOnlineB._profileRevisionTopic(newLive)]);

  const sameMillisecondA = makeRoom(broker.open);
  const sameMillisecondB = makeRoom(broker.open);
  sameMillisecondA.connect('SAME_MS', { shared: true });
  sameMillisecondB.connect('SAME_MS', { shared: true });
  const actualNow = Date.now;
  Date.now = () => 50000;
  let a, b;
  try {
    a = sameMillisecondA.upsertProfile(profile('same_id', 2));
    b = sameMillisecondB.upsertProfile(profile('same_id', 3));
  } finally {
    Date.now = actualNow;
  }
  assert.equal(a.updatedAt, b.updatedAt);
  assert.notEqual(a.revisionId, b.revisionId);
  const winner = a.revisionId > b.revisionId ? a : b;
  await eventually(() => [sameMillisecondA, sameMillisecondB].every((tab) => tab.getProfiles()[0]?.revisionId === winner.revisionId), 'Both tabs agree on the equal-timestamp revision');
  sameMillisecondA.disconnect();
  sameMillisecondB.disconnect();
  const tieReload = makeRoom(broker.open);
  tieReload.connect('SAME_MS', { shared: true });
  assert.deepEqual(tieReload.getProfiles(), [winner]);
  assert.ok(tieReload.pendingUpdates.has(winner.id));

  const deleting = makeRoom(broker.open);
  deleting.connect('DELETE', { shared: true });
  broker.newest.accept();
  const beforeDelete = deleting.upsertProfile(profile('deferred'));
  const onlyDeleted = deleting.upsertProfile(profile('only_deleted'));
  const oldTopic = deleting._profileRevisionTopic(beforeDelete);
  broker.newest.end();
  deleting.deleteProfile(beforeDelete.id);
  deleting.deleteProfile(onlyDeleted.id);
  const deletedAt = deleting.tombstones.get(beforeDelete.id);
  assert.ok(broker.retained.has(oldTopic), 'Offline delete has yet to reach the broker');
  deleting.disconnect();
  const restored = makeRoom(broker.open);
  restored.connect('DELETE', { shared: true });
  assert.equal(restored.getProfiles().length, 0);
  assert.equal(restored._mergeIncomingProfile(beforeDelete), false, 'An older cache cannot resurrect a deletion');

  const newer = { ...beforeDelete, updatedAt: deletedAt + 1, revisionId: 'second_device' };
  const newTopic = restored._profileRevisionTopic(newer);
  const legacyTopic = restored._profileTopic(beforeDelete.id);
  broker.retained.set(newTopic, JSON.stringify(newer));
  broker.retained.set(legacyTopic, JSON.stringify(beforeDelete));
  broker.newest.accept();
  assert.deepEqual(restored.getProfiles(), [newer]);
  assert.ok(broker.retained.has(restored._deletionTopic(beforeDelete.id, deletedAt)));
  assert.ok(!broker.retained.has(oldTopic), 'The old revision is cleared after the retained delete is acknowledged');
  assert.ok(!broker.retained.has(restored._profileRevisionTopic(onlyDeleted)), 'A delete can clean a revision even with no replacement');
  assert.ok(broker.retained.has(newTopic), 'A later revision remains on the broker');
  assert.ok(broker.retained.has(legacyTopic), 'A mutable legacy topic is never cleared without a conditional broker API');

  const legacy = makeRoom(broker.open);
  legacy.connect('LEGACY', { shared: true });
  const oldSnapshot = { ...beforeDelete, id: 'old_snapshot', revisionId: undefined };
  storage.set(legacy._storageKey(), JSON.stringify({ ...compat, profiles: [oldSnapshot], tombstones: {}, pendingUpdates: [oldSnapshot.id] }));
  legacy.connect('LEGACY', { shared: true });
  assert.ok(legacy.pendingUpdates.has(oldSnapshot.id));
  broker.newest.accept();
  assert.ok(broker.retained.has(legacy._profileRevisionTopic(oldSnapshot)), 'Pending v4 cache snapshots can still be replayed');
  legacy.disconnect();

  const retryBroker = new MockBroker();
  const attempts = [];
  const retrying = makeRoom((url) => {
    attempts.push(url);
    if (attempts.length === 1) throw new Error('Simulated WebSocket startup failure');
    return retryBroker.open(url);
  });
  retrying.connect('RETRY', { shared: true });
  mock.timers.tick(1);
  assert.equal(attempts.length, 2, 'A synchronous first startup failure tries the next broker');
  const timedOutClient = retryBroker.newest;
  mock.timers.tick(5000);
  assert.equal(retrying.client, null, 'A silent startup timeout closes the failed transport');
  assert.equal(retrying.status, 'local');
  retrying.upsertProfile(profile('held_for_retry'));
  mock.timers.tick(4000);
  assert.equal(attempts.length, 3, 'All initial brokers failing still schedules a fresh retry cycle');
  timedOutClient.accept();
  assert.equal(retrying.status, 'connecting', 'A callback from an abandoned broker cannot claim a connection');
  retryBroker.newest.accept();
  assert.equal(retrying.status, 'online');
  assert.equal(retrying.pendingUpdates.size, 0);

  const staleAckClient = retryBroker.newest;
  staleAckClient.holdAcknowledgment = true;
  retrying.upsertProfile(profile('wait_for_ack'));
  staleAckClient.emit('offline');
  mock.timers.tick(1);
  assert.equal(attempts.length, 4, 'An established connection failing also switches brokers');
  staleAckClient.finishPublish();
  assert.ok(retrying.pendingUpdates.has('wait_for_ack'), 'A stale acknowledgment cannot clear the active queue');
  const staleSubscription = retryBroker.newest;
  staleSubscription.holdSubscribe = true;
  staleSubscription.accept();
  retrying.connect('LOCAL');
  staleSubscription.finishSubscribe();
  mock.timers.tick(60000);
  assert.equal(attempts.length, 4, 'Leaving a shared room cancels its connection attempts');
  assert.equal(retrying.status, 'local');
  assert.deepEqual(retrying.getProfiles().map((p) => p.id), ['local_person']);

  // Outdated storage keys (v1..v3, mismatched v4 snapshots, mismatched v4 upsert/delete records)
  // and outdated retained MQTT profile/deletion/peer messages are permanently deleted without triggering isolation warnings.
  const purgeBroker = new MockBroker();
  const outdatedCompat = { schemaVersion: 4, preprocessVersion: 'kp-preproc-v4', modelHash: 'outdated-00000000' };
  const outdatedPrefix = `keyprint_mvp_v4_room_PURGE_records_${encodeURIComponent(
    JSON.stringify([outdatedCompat.schemaVersion, outdatedCompat.preprocessVersion, outdatedCompat.modelHash])
  )}/`;
  const outdatedKeys = [
    'keyprint_mvp_profiles',
    'keyprint_mvp_v3_room_PURGE',
    'keyprint_mvp_v4_room_OTHER_OLD',
    'keyprint_mvp_v4_room_PURGE',
    `${outdatedPrefix}old_p/upsert/100/legacy`,
    `${outdatedPrefix}old_d/delete/200`,
  ];
  storage.set('keyprint_mvp_profiles', JSON.stringify([{ id: 'legacy_v0' }]));
  storage.set('keyprint_mvp_v3_room_PURGE', JSON.stringify([{ id: 'legacy_v3' }]));
  storage.set(
    'keyprint_mvp_v4_room_OTHER_OLD',
    JSON.stringify({ ...outdatedCompat, profiles: [{ ...profile('old_other'), ...outdatedCompat, createdAt: 10, updatedAt: 10 }], tombstones: { old_t: 20 } })
  );
  storage.set(
    'keyprint_mvp_v4_room_PURGE',
    JSON.stringify({ ...outdatedCompat, profiles: [{ ...profile('old_room_p'), ...outdatedCompat, createdAt: 10, updatedAt: 10 }], tombstones: { old_t2: 30 } })
  );
  storage.set(
    `${outdatedPrefix}old_p/upsert/100/legacy`,
    JSON.stringify({ type: 'upsert', profile: { ...profile('old_p'), ...outdatedCompat, createdAt: 100, updatedAt: 100 }, pending: false })
  );
  storage.set(`${outdatedPrefix}old_d/delete/200`, JSON.stringify({ type: 'delete', id: 'old_d', ts: 200 }));

  const outdatedTopics = [
    ['keyprint-agy-mvp-v4/room/PURGE/profiles/old_ret', JSON.stringify({ ...profile('old_ret'), ...outdatedCompat, createdAt: 50, updatedAt: 50 })],
    ['keyprint-agy-mvp-v4/room/PURGE/profiles/old_rev/60/rev_a', JSON.stringify({ ...profile('old_rev'), ...outdatedCompat, createdAt: 60, updatedAt: 60, revisionId: 'rev_a' })],
    ['keyprint-agy-mvp-v4/room/PURGE/deletions/old_del/70', JSON.stringify({ ...outdatedCompat, type: 'delete', id: 'old_del', ts: 70 })],
    ['keyprint-agy-mvp-v4/room/PURGE/peers/kp_oldpeer1', JSON.stringify({ ...outdatedCompat, cid: 'kp_oldpeer1', ts: 80 })],
  ];
  for (const [topic, payload] of outdatedTopics) purgeBroker.retained.set(topic, payload);

  const purger = makeRoom(purgeBroker.open);
  purger.connect('PURGE', { shared: true });
  purgeBroker.newest.accept();
  for (const k of outdatedKeys) {
    assert.equal(storage.has(k), false, `Outdated localStorage key ${k} was permanently deleted`);
  }
  for (const [topic] of outdatedTopics) {
    assert.equal(purgeBroker.retained.has(topic), false, `Outdated retained MQTT topic ${topic} was cleared`);
  }
  assert.equal(purger.getProfiles().length, 0, 'Outdated profiles were not imported');
  assert.equal(purger.tombstones.size, 0, 'Outdated deletion events were not imported');
  assert.equal(purger.compatSkipCount, 0, 'Purging outdated persisted/retained state does not raise compatibility warning');
} finally {
  for (const sync of instances) sync.disconnect();
  mock.timers.reset();
}
console.log('Functional state transitions passed: consent, async submissions, offline tabs, equal-timestamp revisions, legacy storage, safe deferred deletion, broker retries and stale callbacks.');
