import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// Serves /api/spotify-token from the Vite server so the Spotify client secret
// never reaches the browser. Works under `npm run dev` and `npm run preview`.
function spotifyToken(env) {
  const handler = async (_req, res) => {
    res.setHeader('Content-Type', 'application/json')
    if (!env.SPOTIFY_CLIENT_ID || !env.SPOTIFY_CLIENT_SECRET) {
      res.statusCode = 500
      res.end(JSON.stringify({ error: 'Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.local' }))
      return
    }
    try {
      const auth = Buffer.from(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`).toString('base64')
      const response = await fetch('https://accounts.spotify.com/api/token', {
        method: 'POST',
        headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'grant_type=client_credentials',
      })
      res.statusCode = response.status
      res.end(await response.text())
    } catch (error) {
      res.statusCode = 502
      res.end(JSON.stringify({ error: String(error) }))
    }
  }
  return {
    name: 'spotify-token',
    configureServer(server) {
      server.middlewares.use('/api/spotify-token', handler)
    },
    configurePreviewServer(server) {
      server.middlewares.use('/api/spotify-token', handler)
    },
  }
}

export default defineConfig(({ mode }) => {
  // '' prefix loads non-VITE_ vars too; they stay on the server.
  const env = loadEnv(mode, process.cwd(), '')
  return { plugins: [react(), spotifyToken(env)] }
})
