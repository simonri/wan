import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Served by FastAPI at /client/ in production.
  base: "/client/",
  server: {
    // Dev mode: proxy API + WS to the wan server.
    proxy: {
      "/v1": {
        target: "http://127.0.0.1:8001",
        ws: true,
      },
    },
  },
});
