---
title: Sign Bridge ASL
emoji: 🤟
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Two-way ASL — sign → text and text → sign in one app
---

# ASL Two-Way Bridge

A real-time American Sign Language recognition app with **two directions in one interface**:

- **Phase 1 — Sign → Text.** The browser captures your webcam with `getUserMedia` and streams JPEG frames over Socket.IO. The server runs MediaPipe hand landmarks + a lightweight MLP (A–Z) and sends predictions back. A temporal stability window builds sentences while you sign.
- **Phase 2 — Text → Sign.** Type a sentence and the app plays the corresponding ASL video clips back to you, letter by letter.

Both phases share a single Flask + Socket.IO backend and a warm-minimal UI.

---

## Why this project

Most open-source ASL repos go one way only: either live detection or a static reference dictionary. This repo glues both surfaces together so the same app can be used by someone learning to sign (Phase 2) and by someone learning to be understood (Phase 1). The letter-level scope is intentional — it keeps the model small, the camera loop fast, and the UX understandable.

## Tech stack

| Layer         | What it uses                                                    |
| ------------- | --------------------------------------------------------------- |
| Detection     | MediaPipe Hands (21 3D landmarks)                               |
| Classifier    | TensorFlow / Keras MLP — Dense(256) → Dense(128) → 26 softmax  |
| Server        | Flask + Flask-SocketIO (threading mode) on port 5000            |
| Frontend      | React 19 + Vite, Socket.IO client, pre-built into `web_ui/dist` |
| Phase 2 asset | 26 per-letter `.mp4` clips served statically from `phase2/`     |

## Repository layout

```
Sign-language-detection/
├── web_server.py            # Flask + Socket.IO entry point
├── config/config.py         # Central config (labels, thresholds, paths)
├── src/
│   ├── collect.py           # Capture landmark samples to data/ASL_Data/
│   ├── train.py             # Train the MLP, write models/asl_landmark_dl_model.h5
│   └── utils.py             # normalize_landmarks, PredictionSmoother, helpers
├── models/
│   └── asl_landmark_dl_model.h5   # Live inference model (committed)
├── data/ASL_Data/           # Training samples (.npy), one folder per letter
├── phase2/                  # Text → Sign static app + 26 MP4 clips
├── web_ui/
│   ├── src/                 # React source
│   └── dist/                # Built bundle served by Flask (committed)
├── requirements.txt
├── Dockerfile / .dockerignore
├── Procfile                 # For Heroku / Render / Railway
├── .env.example
├── DEPLOY.md                # Deployment notes + architectural caveat
└── LICENSE                  # MIT
```

## Quickstart (local)

Prerequisites: **Python 3.11**, **Node 20+**, a working webcam.

```bash
# 1. Python env
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. Frontend bundle (already committed, but rebuild if you change web_ui/src)
npm install --prefix web_ui
npm run build --prefix web_ui

# 3. Run
python web_server.py
# → http://localhost:5000
```

Phase 2 lives at `http://localhost:5000/phase2/`.

Optional: copy `.env.example` to `.env` to override `HOST` / `PORT` / `DEBUG`.

## Using the app

**Phase 1 — signing to text:**

- Press the play button to start the camera.
- Sign a letter and hold it steady for ~0.8 s — the green stability bar fills and the letter is appended to the sentence.
- Leave no hand in frame for ~1.2 s to insert a space.
- Keyboard shortcuts: `Space` adds a space, `Backspace` deletes the last letter, `Esc` clears.

**Phase 2 — text to signing:**

- Type a word or sentence.
- Press play to watch each letter signed back to you as a short video clip.

## Retraining (optional)

You can collect your own samples and retrain:

```bash
python src/collect.py        # saves landmark vectors to data/ASL_Data/<LETTER>/*.npy
python src/train.py          # writes models/asl_landmark_dl_model.h5
```

Hyperparameters live in `config/config.py`.

## Deployment

See [DEPLOY.md](DEPLOY.md). **Important caveat up front:** Phase 1 reads the webcam from wherever `web_server.py` runs. Deploying it to a normal cloud host means the server has no camera and Phase 1 goes dark. For a public demo, run it locally or use a self-hosted tunnel (Tailscale / Cloudflare Tunnel / ngrok) so the camera stays on the user's machine. Phase 2 is just static video playback and deploys anywhere.

## License

MIT — see [LICENSE](LICENSE).
