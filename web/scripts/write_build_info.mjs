import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const revision = process.argv[2] || process.env.GITHUB_SHA;
if (!/^[a-f0-9]{40}$/.test(revision || '')) throw new Error('Expected a full Git commit SHA.');
const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const config = JSON.parse(fs.readFileSync(path.join(dist, 'models', 'model_config.json'), 'utf8'));
const models = {};
const encoders = { sentence_17ch: config.sentence_model, password_9ch: config.password_model,
  password_17ch: config.password_aux_model };
for (const [role, expected] of Object.entries(config.onnx_sha256)) {
  const file = encoders[role].file;
  const bytes = fs.readFileSync(path.join(dist, 'models', file));
  models[file] = createHash('sha256').update(bytes).digest('hex');
  if (models[file] !== expected) {
    throw new Error('Published encoder differs from its model configuration: ' + file);
  }
}
models['model_config.json'] = createHash('sha256').update(fs.readFileSync(path.join(dist, 'models', 'model_config.json'))).digest('hex');
fs.writeFileSync(path.join(dist, 'build-info.json'), JSON.stringify({ revision, models }, null, 2) + '\n');
