// Runs after the functions from the pinned Arena evaluation harness.
// Studies and candidate-level scores travel through private process pipes.
import { readFileSync } from 'node:fs';

function pairedPlan(n) {
  let state = 71518;
  function next() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 0x100000000;
  }
  const all = Array.from({ length: n }, (_, i) => i);
  const numeric = (a, b) => a - b;
  const primary = all.map((writer) => {
    const others = all.filter((i) => i !== writer);
    for (let k = 0; k < 5; k++) {
      const pick = k + Math.floor(next() * (others.length - k));
      [others[pick], others[k]] = [others[k], others[pick]];
    }
    return { enrolled: [writer, ...others.slice(0, 4)].sort(numeric), unknown: others.slice(0, 5).sort(numeric) };
  });
  return { primary, full: all.map(() => ({ enrolled: all, unknown: [] })) };
}

async function captureScores(release, study, { raw = false, fixedLeads = false }, evaluateStudy) {
  const { engine } = release;
  const identify = engine.identifyTypist;
  const score = engine.scoreTemplate;
  const index = new Map(study.writers.map((writer, i) => [writer.id, i]));
  const threshold = engine.arenaThreshold(study.mode);
  const margin = engine.arenaMargin(study.mode);
  const matrix = [];
  const rawMatrix = [];
  const leads = [];

  engine.identifyTypist = (embedding, roster, options) => {
    const wholeRoster = roster.length === index.size;
    const row = [];
    if (wholeRoster && raw) {
      engine.scoreTemplate = (...args) => {
        const verdict = score.call(engine, ...args);
        row.push(.65 * verdict.centroidSim + .35 * verdict.exemplarSim);
        return verdict;
      };
    }
    let verdict;
    try {
      verdict = identify.call(engine, embedding, roster, options);
    } finally {
      engine.scoreTemplate = score;
    }
    if (wholeRoster) {
      const candidateScores = new Array(index.size);
      for (const item of verdict.rankings) candidateScores[index.get(item.id)] = item.similarity;
      matrix.push(candidateScores);
      if (raw) {
        requireCondition(row.length === index.size && row.every(Number.isFinite), 'Missing raw candidate scores.');
        rawMatrix.push(row);
      }
    }
    if (fixedLeads && roster.length === 5) {
      const similarity = verdict.winner.similarity;
      requireCondition(verdict.identified === (similarity >= threshold && verdict.lead >= margin), 'Gate mismatch.');
      leads.push({ similarity, lead: verdict.lead, winner: index.get(verdict.winner.id), accepted: verdict.identified });
    }
    return verdict;
  };
  let evaluation;
  try {
    evaluation = await evaluateStudy();
  } finally {
    engine.identifyTypist = identify;
  }
  const queryCount = study.queryCaptures / engine.arenaProtocol().query;
  requireCondition(matrix.length === queryCount, 'Expected one full roster per probe.');
  if (fixedLeads) requireCondition(leads.length === 2 * queryCount, 'Expected a known and an outsider roster per probe.');
  return {
    evaluation, matrix, ...(raw ? { rawMatrix } : {}), ...(fixedLeads ? { leads } : {}), threshold, margin,
    revision: release.revision, config_sha256: release.configSha256, model_hash: engine.modelHash, runtime: release.runtime,
  };
}

async function scoreAIHub(study, main, pr) {
  const plan = pairedPlan(study.writers.length);
  const runs = {};
  for (const [label, release] of Object.entries({ main, pr })) {
    const { evaluation, ...scores } = await captureScores(release, study, { raw: label === 'pr' },
      () => evaluate(release, study, plan, true));
    runs[label] = { ...scores, fixed: evaluation.primary };
  }
  return { plan: plan.primary, ...runs };
}

const [mainRef, prRef, historicalRef] = process.argv.slice(2);
const input = JSON.parse(readFileSync(0, 'utf8'));
const scratch = await mkdtemp(resolve(tmpdir(), 'keyprint-ko-frozen-'));
try {
  const [mainSource, prSource, historicalSource] = snapshotSources(scratch, [mainRef, prRef, historicalRef]);
  const main = await loadRelease(scratch, 'main', mainSource);
  const pr = await loadRelease(scratch, 'pr', prSource);
  const protocol = main.engine.arenaProtocol();
  const next = pr.engine.arenaProtocol();
  requireCondition(protocol.enroll === 5 && protocol.query === 2 &&
    next.enroll === protocol.enroll && next.query === protocol.query, 'Expected a five-capture enrollment and a two-capture probe.');
  requireCondition(pr.engine.arenaThreshold('touch') === .78 && pr.engine.arenaMargin('touch') === .12,
    'Unexpected frozen PR touch settings.');

  const aihub = [];
  for (const study of input.aihub) {
    aihub.push(await scoreAIHub(loadStudy(study, protocol), main, pr));
  }
  const fact = {};
  for (const [language, data] of Object.entries(input.fact)) {
    const study = loadStudy(data, protocol);
    const { evaluation: report, ...scores } = await captureScores(pr, study, { raw: true, fixedLeads: true },
      () => reportStudy(study, protocol, main, pr, true));
    fact[language] = { report, ...scores };
  }
  const historical = await loadRelease(scratch, 'historical', historicalSource);
  const study = loadStudy(input.historical, historical.engine.arenaProtocol());
  const gallery = study.writers.map((_, index) => index);
  const plan = { primary: gallery.map(() => ({ enrolled: gallery, unknown: [] })) };
  const { matrix } = await captureScores(historical, study, {}, () => evaluate(historical, study, plan, false));
  process.stdout.write(JSON.stringify({ aihub, fact, historical: matrix }));
} finally {
  await rm(scratch, { recursive: true, force: true });
}
