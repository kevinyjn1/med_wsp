import { defineConfig } from "vite";

// three.js is most of the bundle; keep it in its own chunk so the app code stays small and caches separately.
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three")) return "three";
        },
      },
    },
  },
});
