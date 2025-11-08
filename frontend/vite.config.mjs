import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { getTaggerPlugin } from "./server/taggerPlugin.js";
import { goldListingsApiPlugin } from "./server/goldListingsApi.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// https://vitejs.dev/config/
export default defineConfig({
  // This changes the out put dir from dist to build
  // comment this out if that isn't relevant for your project
  build: {
    outDir: "build",
    chunkSizeWarningLimit: 2000,
  },
  plugins: [react(), getTaggerPlugin(), goldListingsApiPlugin()],
  resolve: {
    alias: {
      components: path.resolve(__dirname, "src/components"),
      pages: path.resolve(__dirname, "src/pages"),
      "react-router-dom": path.resolve(__dirname, "shims/react-router-dom.jsx"),
      "lucide-react": path.resolve(__dirname, "shims/lucide-react.jsx"),
      recharts: path.resolve(__dirname, "shims/recharts.jsx"),
    },
  },
  server: {
    port: 3000,
    host: "127.0.0.1",
    strictPort: true,
    allowedHosts: [".amazonaws.com", ".builtwithrocket.new"],
  },
});
