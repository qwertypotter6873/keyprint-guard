import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    target: 'es2022',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
