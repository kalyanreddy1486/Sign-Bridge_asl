# Deployment notes

## The webcam caveat (read first)

Phase 1 uses `cv2.VideoCapture()` inside `web_server.py` to read a physical webcam attached to the machine running the server. That means:

- When you run it **locally**, the server reads **your** laptop's camera — which is exactly what you want.
- When you run it on **Heroku / Render / Railway / any cloud VM**, the server reads the cloud machine's camera — which doesn't exist. Phase 1 will appear stuck on "warming up…" forever.

For a public demo of Phase 1, pick one of:

1. **Run locally + tunnel.** Start the server on your laptop, then expose it with a tunnel:
   - [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
   - [Tailscale Funnel](https://tailscale.com/kb/1223/funnel)
   - [ngrok](https://ngrok.com/) (`ngrok http 5000`)
   The camera stays local, visitors hit a public URL, the signal round-trips through the tunnel.
2. **Rewrite Phase 1 as browser-side inference** (out of scope for this repo today). Move MediaPipe + TFLite to the frontend with `@mediapipe/tasks-vision`, drop the Python camera loop entirely, and then you can deploy anywhere.
3. **Deploy Phase 2 only** if all you need is the text → sign playback demo.

## Supported deploy targets

| Target               | Works? | How                                                 |
| -------------------- | ------ | --------------------------------------------------- |
| Local + tunnel       | Yes    | `python web_server.py` + ngrok/Tailscale/Cloudflare |
| Docker on own VM     | Yes    | See below — but VM needs a real webcam passthrough  |
| Heroku / Render      | Phase 2 only | Phase 1 has no camera to read                 |
| Railway / Fly.io     | Phase 2 only | Same caveat                                   |
| GitHub Pages         | No     | Python server required                              |

## Environment variables

All set in `.env` (copy from `.env.example`) or injected by the platform.

| Var     | Default   | Notes                                         |
| ------- | --------- | --------------------------------------------- |
| `HOST`  | `0.0.0.0` | Bind address                                  |
| `PORT`  | `5000`    | Cloud platforms override this automatically   |
| `DEBUG` | `false`   | Set to `true` for Flask debug during dev only |

## Docker

```bash
# Build
docker build -t asl-bridge .

# Run (note: container has no webcam unless you pass one through)
docker run --rm -p 5000:5000 asl-bridge
# → http://localhost:5000/phase2/   works
# → http://localhost:5000/          loads but camera stays dark

# Linux host with a /dev/video0 passthrough (Phase 1 works):
docker run --rm --device=/dev/video0:/dev/video0 -p 5000:5000 asl-bridge
```

## Heroku / Render / Railway (Phase 2 demo only)

The `Procfile` runs `python web_server.py`. The platform injects `PORT` automatically.

```bash
# Heroku
heroku create asl-bridge-demo
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs
git push heroku main

# Render / Railway
# Connect the repo in the dashboard, keep defaults, the Procfile does the rest.
```

After deploy, tell visitors to use `/phase2/` only. The root `/` (Phase 1) will load but never produce a camera feed.

## Pre-deploy checklist

- [ ] `python web_server.py` runs cleanly locally (server prints `[OK] Server ready`).
- [ ] `web_ui/dist/` exists and is committed, or the build step is wired into your platform.
- [ ] `models/asl_landmark_dl_model.h5` is committed (Phase 1 needs it at runtime).
- [ ] `phase2/Asl video data/` contains all 26 MP4s (Phase 2 needs them).
- [ ] `.env` is **not** committed (it's in `.gitignore`); `.env.example` is committed.
- [ ] `CORS` origins in `web_server.py` include the domain you're deploying to, if different from localhost.
