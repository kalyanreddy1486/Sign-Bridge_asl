# Deployment notes

## Architecture: where the camera lives

The browser owns the webcam (via `getUserMedia`) and streams JPEG frames to the server over Socket.IO. The server runs MediaPipe + the MLP on each received frame and emits predictions back. The server never calls `cv2.VideoCapture` — so it doesn't need a physical camera, and the app deploys to any cloud host.

Two practical implications:

1. **HTTPS (or `localhost`) is required** — browsers block `getUserMedia` on insecure origins. All the deploy targets below give you HTTPS automatically.
2. **Visitors grant camera permission in their browser**, not you as the host. First-time users will see a "sign-bridge.hf.space wants to use your camera" prompt.

## Supported deploy targets

| Target                         | Works? | How                                                             |
| ------------------------------ | ------ | --------------------------------------------------------------- |
| Local laptop (dev)             | Yes    | `python web_server.py` → http://localhost:5000                  |
| **Hugging Face Spaces**        | Yes    | Docker SDK, `app_port: 7860`. Free tier works. See below.       |
| Render / Railway / Fly.io      | Yes    | Docker-based or Python buildpack with the included `Procfile`.  |
| Google Cloud Run / AWS Fargate | Yes    | Deploy the Docker image.                                        |
| Any VPS                        | Yes    | Install Python, run `python web_server.py`, put behind nginx.   |
| GitHub Pages / Netlify / Vercel static | No     | They don't run long-lived Python processes.             |

## Hugging Face Spaces (recommended free path)

1. Create a new Space at https://huggingface.co/new-space — **Docker** SDK.
2. Clone the Space's git repo locally, copy this project in, push.
3. Or push this repo directly as a Space by renaming the remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/sign-bridge-asl
   git push hf main
   ```
4. HF reads the README frontmatter (`sdk: docker`, `app_port: 7860`) and the included `Dockerfile` to build the image and route traffic.
5. Wait ~5–10 min for the first build (TensorFlow is a chunky install). Subsequent pushes rebuild in ~1 min.

The free CPU tier (16 GB RAM, 2 vCPU) handles the MLP + MediaPipe workload. Expect ~150–300 ms round-trip per frame on free CPU, which is fine because the pump runs at ~8 fps.

## Environment variables

Set in `.env` (copy from `.env.example`) for local dev, or inject via the platform.

| Var     | Default   | Notes                                                  |
| ------- | --------- | ------------------------------------------------------ |
| `HOST`  | `0.0.0.0` | Bind address                                           |
| `PORT`  | `5000`    | HF Spaces expects `7860` — the Dockerfile sets this    |
| `DEBUG` | `false`   | Set to `true` for Flask debug during dev only          |

## Docker (any host)

```bash
# Build
docker build -t sign-bridge .

# Run — works standalone, visitors' browsers supply the webcam
docker run --rm -p 7860:7860 sign-bridge
# → http://localhost:7860/

# Override port if needed
docker run --rm -p 8080:8080 -e PORT=8080 sign-bridge
```

## Render / Railway / Heroku via Procfile

The included `Procfile` runs `python web_server.py`. The platform auto-injects `PORT`.

```bash
# Render / Railway: connect the repo in the dashboard, keep defaults.
# Heroku:
heroku create sign-bridge-demo
heroku buildpacks:add heroku/nodejs      # builds web_ui/dist
heroku buildpacks:add heroku/python      # runs web_server.py
git push heroku main
```

## Pre-deploy checklist

- [ ] `python web_server.py` runs cleanly locally → `[OK] Server ready`.
- [ ] http://localhost:5000/ loads and `Start camera` opens the webcam in the browser.
- [ ] `models/asl_landmark_dl_model.h5` is committed.
- [ ] `data/ASL_Data/` is committed (startup builds the Phase 2 pose library from it).
- [ ] `phase2/Asl video data/` contains all 26 MP4s.
- [ ] `web_ui/dist/` is committed, or a CI step rebuilds it on push.
- [ ] `.env` is **not** committed (gitignored); `.env.example` is.

## Troubleshooting

**"Camera access denied" or blank placeholder.** The user hit "block" on the permission prompt, or the site is not served over HTTPS. On any public deploy, HTTPS is automatic. For local testing, `localhost:5000` counts as secure.

**Slow predictions (>1s per frame).** You're on a tiny instance without CPU. Upgrade the tier, or reduce `FRAME_INTERVAL_MS` in `web_ui/src/App.jsx` from `120` to `200` (5 fps instead of ~8 fps) to cut server load.

**"Too many frames" / socket disconnects.** The client is pushing faster than the server can consume. Raise `FRAME_INTERVAL_MS`, or upgrade the instance.
