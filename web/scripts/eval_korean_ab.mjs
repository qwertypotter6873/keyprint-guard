/**
 * Score the same Korean captures with two committed Arena releases.
 * After npm ci --prefix web, run from the repository root:
 * node web/scripts/eval_korean_ab.mjs --input /private/study.json > /tmp/korean-report.json
 * Each capture is the recorder's clean stroke and summary, plus a shared prompt ID.
 * The output contains aggregate counts only. Keep the input file private.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const root = fileURLToPath(new URL('../../', import.meta.url));
const upstream = 'https://github.com/qwertypotter6873/keyprint-guard.git';
const liveMain = '4afdaa2a4a3e6063a6cfbeb82b893481866dd471';
const reviewedCandidate = 'e7529d3f6be9b89f5f71a6958dadaa887e113c8f';
const schema = 'keyprint-korean-eval/v1';
const sha256 = (input) => createHash('sha256').update(input).digest('hex');
const readJson = async (path) => JSON.parse(await readFile(path, 'utf8'));
const identifier = (value) => typeof value === 'string' && value.length > 0 && value.length <= 128;
const finiteOrNull = (value) => value === null || Number.isFinite(value);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function validateCapture(capture, inputMode, label) {
  const bad = `${label}의 타건 특징이 올바르지 않습니다.`;
  const { stroke } = capture ?? {};
  const count = stroke?.dds?.length;
  requireCondition(Number.isInteger(count) && count >= 8 &&
    count === capture.cleanDigraphCount && capture.inputMode === inputMode, bad);
  const lengthIs = (field, size, valid) => Array.isArray(stroke[field]) && stroke[field].length === size &&
    stroke[field].every(valid);
  const isBool = (v) => typeof v === 'boolean';
  const validPair = (pair) => Array.isArray(pair) && pair.length === 2 &&
    pair.every((value) => value === null || identifier(value));
  requireCondition(lengthIs('dds', count, (d) => Number.isFinite(d) && d >= 0.006 && d <= 2.2) &&
    ['uds', 'hCurr', 'hNext'].every((key) => lengthIs(key, count, finiteOrNull)) &&
    ['hvCurr', 'hvNext', 'stitchFlags'].every((key) => lengthIs(key, count, isBool)) &&
    lengthIs('pairKeys', count, validPair) && stroke.stitchFlags[0] === false, bad);
  const holdCount = count + 1 + stroke.stitchFlags.filter(Boolean).length;
  requireCondition(lengthIs('holds', holdCount, finiteOrNull) && lengthIs('holdValid', holdCount, isBool) &&
    stroke.holds.every((hold, i) => stroke.holdValid[i] === (hold !== null)) &&
    stroke.hCurr.every((hold, i) => stroke.hvCurr[i] === (hold !== null)) &&
    stroke.hNext.every((hold, i) => stroke.hvNext[i] === (hold !== null)) &&
    stroke.uds.every((ud, i) => stroke.hvCurr[i] === (ud !== null)), bad);
  const validFraction = stroke.holdValid.filter(Boolean).length / holdCount;
  requireCondition(inputMode === 'keyboard' ? validFraction >= 0.8 : validFraction <= 0.2, bad);
  requireCondition(['cpm', 'holdMs', 'rolloverPct'].every((k) => Number.isFinite(capture[k]) && capture[k] >= 0), bad);
  if (capture.behavior !== null) {
    requireCondition(capture.behavior && typeof capture.behavior === 'object' && !Array.isArray(capture.behavior) &&
      Object.values(capture.behavior).every((stat) => Array.isArray(stat) && stat.length === 2 &&
        Number.isFinite(stat[0]) && Number.isInteger(stat[1]) && stat[1] >= 0), bad);
  }
}

function loadStudy(study, protocol) {
  const { participants, input_mode: mode } = study;
  const captureProtocol = study.capture_protocol ?? 'shared_prompts';
  const sharedPrompts = captureProtocol === 'shared_prompts';
  requireCondition(study.schema === schema && study.language === 'ko' && ['keyboard', 'touch'].includes(mode) &&
    ['shared_prompts', 'published_recording_split'].includes(captureProtocol), 'Unsupported Korean study format.');
  requireCondition(Array.isArray(participants) && participants.length >= 2, 'At least two writers are required.');
  const participantIds = new Set();
  const timingHashes = new Set();
  let commonPrompts = null;
  const writers = participants.map((participant, p) => {
    const row = `참가자 ${p + 1}`;
    const { id, sessions } = participant ?? {};
    requireCondition(identifier(id) && !participantIds.has(id), `${row}의 식별자가 올바르지 않습니다.`);
    participantIds.add(id);
    requireCondition(Array.isArray(sessions) && sessions.length === 2 && sessions[0]?.role === 'enroll' &&
      sessions[1]?.role === 'query' && sessions.every((s) => identifier(s.id)) && sessions[0].id !== sessions[1].id,
    `${row}의 등록과 판별에는 서로 다른 기록 세트가 필요합니다.`);
    const enroll = sessions[0].captures;
    const query = sessions[1].captures;
    requireCondition(Array.isArray(enroll) && enroll.length === protocol.enroll && Array.isArray(query) &&
      query.length >= protocol.query && query.length % protocol.query === 0,
    `${row}의 등록 ${protocol.enroll}개 및 판별 ${protocol.query}개 단위 기록을 확인하세요.`);
    const captures = [...enroll, ...query];
    const labels = captures.map((capture) => capture?.[sharedPrompts ? 'prompt_id' : 'recording_id']);
    requireCondition(labels.every(identifier) && new Set(labels).size === labels.length,
      `${row}의 기록별 식별자가 누락되었거나 중복되었습니다.`);
    if (sharedPrompts) {
      commonPrompts ??= labels;
      requireCondition(labels.length === commonPrompts.length && labels.every((label, i) => label === commonPrompts[i]),
        `${row}의 참가자 공통 문장 순서가 일치하지 않습니다.`);
    }
    captures.forEach((capture, i) => {
      validateCapture(capture, mode, `${row} 기록 ${i + 1}`);
      const { dds, hCurr, hNext } = capture.stroke;
      const timingHash = sha256(JSON.stringify([dds, hCurr, hNext]));
      requireCondition(!timingHashes.has(timingHash), `${row} 기록 ${i + 1}의 타건 타이밍이 다른 기록과 중복되었습니다.`);
      timingHashes.add(timingHash);
    });
    return { id, enroll, query };
  });
  const all = writers.flatMap(({ enroll, query }) => [...enroll, ...query]);
  return {
    writers,
    mode,
    captureProtocol,
    enrollCaptures: writers.length * protocol.enroll,
    queryCaptures: all.length - writers.length * protocol.enroll,
    behaviorCaptures: all.filter(({ behavior }) => behavior !== null).length,
  };
}

function git(directory, args, options = {}) {
  return execFileSync('git', ['-C', directory, ...args], options);
}

function optionalGit(directory, args) {
  try {
    return git(directory, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  } catch (error) {
    if (error.status === 1 || error.status === 128) return null;
    throw error;
  }
}

function snapshotSources(tempDir, refs) {
  const local = optionalGit(root, ['rev-parse', '--show-toplevel']);
  const resolveCommit = (directory, ref) =>
    optionalGit(directory, ['rev-parse', '--verify', '--quiet', '--end-of-options', `${ref}^{commit}`]);
  const resolved = refs.map((ref) => local && resolveCommit(local, ref));
  const missing = refs.filter((_, i) => !resolved[i]);
  if (!missing.length) return resolved.map((revision) => ({ directory: local, revision }));
  requireCondition(missing.every((ref) => /^[a-f0-9]{40}$/.test(ref)),
    'References missing locally must be supplied as full 40-character commit IDs.');

  // Standalone ZIPs and shallow clones can fetch the frozen releases privately.
  const cache = resolve(tempDir, 'frozen-releases.git');
  git(tempDir, ['init', '--quiet', '--bare', cache]);
  git(cache, ['remote', 'add', 'origin', upstream]);
  git(cache, ['config', 'remote.origin.promisor', 'true']);
  git(cache, ['fetch', '--quiet', '--no-tags', '--depth=1', '--filter=blob:none', 'origin', ...new Set(missing)],
    { timeout: 45000 });
  return refs.map((ref, i) => {
    if (resolved[i]) return { directory: local, revision: resolved[i] };
    const revision = resolveCommit(cache, ref);
    requireCondition(revision === ref, 'The fetched revision differs from the requested immutable commit.');
    return { directory: cache, revision };
  });
}

async function loadRelease(tempDir, name, { directory, revision }) {
  const releaseDir = resolve(tempDir, name);
  await mkdir(releaseDir);
  const archive = git(directory, ['archive', '--format=tar', revision, 'web/src/biometrics.js',
    'web/src/features.js', 'web/src/recorder.js', 'web/src/validation.js', 'web/package.json',
    'web/package-lock.json', 'web/public/models'], { maxBuffer: 32 * 1024 * 1024, timeout: 45000 });
  execFileSync('tar', ['-xf', '-', '-C', releaseDir], { input: archive });
  const web = resolve(releaseDir, 'web');
  const lock = await readJson(resolve(web, 'package-lock.json'));
  const nodeModules = resolve(root, 'web/node_modules');
  const ort = await readJson(resolve(nodeModules, 'onnxruntime-web/package.json'));
  requireCondition(ort.version === lock.packages['node_modules/onnxruntime-web'].version,
    `ONNX Runtime must match the release lockfile for ${name}. Run npm ci --prefix web.`);
  await symlink(nodeModules, resolve(web, 'node_modules'), 'dir');

  const assetRoot = pathToFileURL(resolve(web, 'public/models') + '/');
  const modelUrl = new URL('keyprint_net.onnx', assetRoot);
  const configUrl = new URL('model_config.json', assetRoot);
  const { KeyPrintEngine, mergeTemplateSample } = await import(pathToFileURL(resolve(web, 'src/biometrics.js')).href);
  const { combineEmbeddings, mergeBehavior } = await import(pathToFileURL(resolve(web, 'src/features.js')).href);
  const engine = new KeyPrintEngine();
  const normalFetch = globalThis.fetch;
  globalThis.fetch = async (resource, ...args) =>
    String(resource).startsWith(assetRoot.href)
      ? new Response(await readFile(new URL(resource)))
      : normalFetch(resource, ...args);
  try {
    await engine.init(modelUrl.href, configUrl.href);
  } finally {
    globalThis.fetch = normalFetch;
  }
  return {
    engine, mergeTemplateSample, combineEmbeddings, mergeBehavior, revision, runtime: ort.version,
    configSha256: sha256(await readFile(configUrl)),
  };
}

function makeRosters(writers, fivePersonRosters) {
  const indices = writers.map((_, index) => index);
  const full = indices.map((index) => ({
    enrolled: indices,
    unknown: indices.filter((candidate) => candidate !== index),
  }));
  if (!fivePersonRosters) return { primary: full, full };

  const size = 5;
  requireCondition(indices.length >= 2 * size && indices.length % size === 0,
    'Five-person rosters require at least two evenly sized groups of five writers.');
  const groups = Array.from({ length: indices.length / size }, (_, i) => indices.slice(i * size, (i + 1) * size));
  const primary = indices.map((index) => {
    const group = Math.floor(index / size);
    return { enrolled: groups[group], unknown: groups[(group + 1) % groups.length] };
  });
  return { primary, full };
}

async function evaluate(release, study, plan, fivePersonRosters) {
  const { engine, mergeTemplateSample, combineEmbeddings, mergeBehavior } = release;
  const { writers, mode } = study;
  const { query: queryShots } = engine.arenaProtocol();
  const roster = [];
  for (const writer of writers) {
    let template = null;
    for (const capture of writer.enroll) {
      const { embedding } = await engine.embedStroke(capture.stroke);
      template = mergeTemplateSample(template, { ...capture, embedding });
    }
    roster.push({ id: writer.id, name: writer.id, template });
  }
  const primary = { genuine: [], unknown: [] };
  const full = { genuine: [], unknown: [] };
  const score = (outcome, groups, combined, options, writer, writerIndex) => {
    const verdict = engine.identifyTypist(combined, groups.enrolled.map((i) => roster[i]), options);
    requireCondition(verdict.rankings.every((entry) => Number.isFinite(entry.similarity)),
      'The published scorer produced a nonfinite similarity.');
    outcome.genuine.push({
      writerIndex,
      top1: verdict.winner.id === writer.id,
      accepted: verdict.identified,
      marginRejection: verdict.rejectedByMargin,
    });
    if (groups.unknown.length >= 2) {
      const excluded = engine.identifyTypist(combined, groups.unknown.map((i) => roster[i]), options);
      outcome.unknown.push({ writerIndex, accepted: excluded.identified, marginRejection: excluded.rejectedByMargin });
    }
  };
  for (const [writerIndex, writer] of writers.entries()) {
    for (let start = 0; start < writer.query.length; start += queryShots) {
      const embeddings = [];
      let behavior = null;
      for (const capture of writer.query.slice(start, start + queryShots)) {
        const { embedding } = await engine.embedStroke(capture.stroke);
        embeddings.push(embedding);
        behavior = mergeBehavior(behavior, capture.behavior);
      }
      const options = { behavior, inputMode: mode };
      const combined = combineEmbeddings(embeddings);
      score(primary, plan.primary[writerIndex], combined, options, writer, writerIndex);
      if (fivePersonRosters) score(full, plan.full[writerIndex], combined, options, writer, writerIndex);
    }
  }
  return { primary, full };
}

const metric = (count, total) => ({
  count,
  rate: total ? Number((count / total).toFixed(6)) : null,
});

function summarize(known, unknown) {
  const count = (rows, predicate) => metric(rows.filter(predicate).length, rows.length);
  return {
    enrolled_probes: {
      trials: known.length,
      top1_before_rejection: count(known, (x) => x.top1),
      accepted_correct: count(known, (x) => x.accepted && x.top1),
      accepted_wrong: count(known, (x) => x.accepted && !x.top1),
      rejected: count(known, (x) => !x.accepted),
      rejected_below_threshold: count(known, (x) => !x.accepted && !x.marginRejection),
      rejected_by_margin: count(known, (x) => x.marginRejection),
    },
    unknown_writer_probes: {
      trials: unknown.length,
      false_accept: count(unknown, (x) => x.accepted),
      rejected: count(unknown, (x) => !x.accepted),
      rejected_below_threshold: count(unknown, (x) => !x.accepted && !x.marginRejection),
      rejected_by_margin: count(unknown, (x) => x.marginRejection),
    },
  };
}

function pairedChange(mainRows, candidateRows, predicate) {
  let mainOnly = 0;
  let candidateOnly = 0;
  const byWriter = new Map();
  mainRows.forEach((row, i) => {
    const a = predicate(row);
    const b = predicate(candidateRows[i]);
    mainOnly += Number(a && !b);
    candidateOnly += Number(b && !a);
    const writer = byWriter.get(row.writerIndex) ?? { delta: 0, total: 0 };
    writer.delta += Number(b) - Number(a);
    writer.total += 1;
    byWriter.set(row.writerIndex, writer);
  });
  const blocks = [...byWriter.values()];
  const bootstrap = [];
  let seed = 1729;
  if (blocks.length >= 10) {
    for (let sample = 0; sample < 2000; sample++) {
      let delta = 0;
      let total = 0;
      for (let j = 0; j < blocks.length; j++) {
        seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
        const block = blocks[Math.floor(seed * blocks.length / 0x100000000)];
        delta += block.delta;
        total += block.total;
      }
      bootstrap.push(100 * delta / total);
    }
    bootstrap.sort((a, b) => a - b);
  }
  return {
    main_only: mainOnly,
    candidate_only: candidateOnly,
    candidate_minus_main_pp: mainRows.length
      ? Number((100 * (candidateOnly - mainOnly) / mainRows.length).toFixed(4))
      : null,
    conditional_writer_bootstrap_95_ci_pp: bootstrap.length
      ? [bootstrap[49], bootstrap[1949]].map((value) => Number(value.toFixed(4)))
      : null,
  };
}

async function run({ input, 'main-ref': mainRef, 'candidate-ref': candidateRef, 'five-person-rosters': fivePersonRosters }) {
  const tempDir = await mkdtemp(resolve(tmpdir(), 'keyprint-korean-'));
  try {
    const [mainSource, candidateSource] = snapshotSources(tempDir, [mainRef, candidateRef]);
    const main = await loadRelease(tempDir, 'main', mainSource);
    const candidate = await loadRelease(tempDir, 'candidate', candidateSource);
    const protocol = main.engine.arenaProtocol();
    const otherProtocol = candidate.engine.arenaProtocol();
    requireCondition(protocol.enroll === otherProtocol.enroll && protocol.query === otherProtocol.query,
      'The committed releases require different shot counts.');
    const reports = [];
    for (const path of input) {
      const study = loadStudy(await readJson(resolve(path)), protocol);
      reports.push(await reportStudy(study, protocol, main, candidate, fivePersonRosters));
    }
    console.log(JSON.stringify(reports.length === 1 ? reports[0] : reports, null, 2));
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function reportStudy(study, protocol, main, candidate, fivePersonRosters) {
  const plan = makeRosters(study.writers, fivePersonRosters);
  const a = await evaluate(main, study, plan, fivePersonRosters);
  const b = await evaluate(candidate, study, plan, fivePersonRosters);
  const describe = (release, outcome) => ({
    revision: release.revision,
    combined_model_hash: release.engine.modelHash,
    config_sha256: release.configSha256,
    onnx_runtime_web: release.runtime,
    threshold: release.engine.arenaThreshold(study.mode),
    margin: release.engine.arenaMargin(study.mode),
    configured_behavior_weight: release.engine.behaviorWeight('sentence', study.mode),
    ...summarize(outcome.genuine, outcome.unknown),
  });
  const comparison = (groups, aResult, bResult) => ({
    enrolled_probe_roster_size: groups[0].enrolled.length,
    unknown_probe_roster_size: aResult.unknown.length ? groups[0].unknown.length : null,
    main: describe(main, aResult),
    candidate: describe(candidate, bResult),
    paired_change: {
      top1_before_rejection: pairedChange(aResult.genuine, bResult.genuine, (x) => x.top1),
      accepted_correct: pairedChange(aResult.genuine, bResult.genuine, (x) => x.accepted && x.top1),
      accepted_wrong: pairedChange(aResult.genuine, bResult.genuine, (x) => x.accepted && !x.top1),
      rejected: pairedChange(aResult.genuine, bResult.genuine, (x) => !x.accepted),
      unknown_false_accept: pairedChange(aResult.unknown, bResult.unknown, (x) => x.accepted),
    },
  });
  return {
    schema: 'keyprint-korean-ab-report/v1',
    language: 'ko',
    input_mode: study.mode,
    capture_protocol: study.captureProtocol,
    participants: study.writers.length,
    enroll_shots: protocol.enroll,
    query_shots: protocol.query,
    enroll_captures: study.enrollCaptures,
    query_captures: study.queryCaptures,
    captures_with_behavior: study.behaviorCaptures,
    ...comparison(plan.primary, a.primary, b.primary),
    ...(fivePersonRosters ? { full_roster: comparison(plan.full, a.full, b.full) } : {}),
  };
}

try {
  const { values } = parseArgs({
    options: {
      input: { type: 'string', multiple: true },
      'main-ref': { type: 'string', default: liveMain },
      'candidate-ref': { type: 'string', default: reviewedCandidate },
      'five-person-rosters': { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help || !values.input?.length) {
    console.error('Usage: node web/scripts/eval_korean_ab.mjs --input STUDY.json [--main-ref COMMIT] [--candidate-ref COMMIT] [--five-person-rosters]');
    if (!values.help) process.exitCode = 1;
  } else {
    await run(values);
  }
} catch (error) {
  console.error(error.code === 'ENOENT' ? 'Check the input path and run npm ci --prefix web.' : error.message);
  process.exitCode = 1;
}
