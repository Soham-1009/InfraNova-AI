import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/colorize': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/thermal-preview': 'http://localhost:8000',
      '/postprocess': 'http://localhost:8000',
    },
  },
})
