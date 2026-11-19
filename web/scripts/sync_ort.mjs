import { copyFile, mkdir, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const source = fileURLToPath(new URL('../node_modules/onnxruntime-web/dist/', import.meta.url));
const destination = fileURLToPath(new URL('../public/ort/', import.meta.url));
const runtime = new Set(['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']);
await mkdir(destination, { recursive: true });
for (const file of await readdir(source)) {
  if (runtime.has(file)) continue;
  // Previous builds copied all ORT entry points and execution providers here.
  if (file.startsWith('ort') && /\.(wasm|mjs)$/.test(file)) await rm(path.join(destination, file), { force: true });
}
for (const file of runtime) await copyFile(path.join(source, file), path.join(destination, file));
