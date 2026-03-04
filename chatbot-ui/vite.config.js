import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  build: {
    outDir: '../chatbot-ui-dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ask': 'http://localhost:8000',
      '/ask_text': 'http://localhost:8000',
      '/ask_image': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/tts': 'http://localhost:8000',
      '/admin': 'http://localhost:8000',
      '/rag': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
