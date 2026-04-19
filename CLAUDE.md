# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

- Install Python deps: `pip install -r requirements.txt`
- Install web server deps used by `web_server.py` (not currently in `requirements.txt`): `pip install flask flask-socketio flask-cors`
- Install frontend deps: `npm install --prefix web_ui`
- Build frontend bundle served by Flask: `npm run build --prefix web_ui`
- Run Flask + Socket.IO app (serves `web_ui/dist`): `python web_server.py`
- Run React dev server: `npm run dev --prefix web_ui` (Vite on port 3000)
- Run data collection (camera): `python src/collect.py`
- Train model from collected `.npy` samples: `python src/train.py`
- Run desktop Tkinter GUI pipeline: `python src/gui.py`
- Run CLI menu: `python main.py`

### Validation / checks

- Python syntax smoke check: `python -m py_compile main.py web_server.py src/*.py`
- Frontend production build check: `npm run build --prefix web_ui`

### Tests and lint

- No automated test suite is currently present (`tests/` and pytest config are absent).
- No lint/format tool config is currently present (no ESLint, Ruff, Flake8, or Black config in repo).
- If pytest tests are added, run a single test with: `python -m pytest path/to/test_file.py::test_name`

## High-level architecture

This project has two inference surfaces sharing the same landmark pipeline: a web app (`web_server.py` + `web_ui`) and a desktop GUI (`src/gui.py`). Both rely on MediaPipe hand landmarks -> normalization -> TensorFlow MLP prediction -> temporal smoothing.

### Core pipeline

1. **Data collection (`src/collect.py`)**
   - Captures MediaPipe hand landmarks from webcam.
   - Saves raw landmark vectors to `data/ASL_Data/<LETTER>/*.npy`.
   - Supports manual capture and auto-capture based on landmark stability.

2. **Training (`src/train.py`)**
   - Loads `.npy` samples for A-Z classes.
   - Normalizes landmarks via `utils.normalize_landmarks()` (wrist-relative coordinates).
   - Trains configurable MLP from `config/config.py`.
   - Writes model artifacts and evaluation outputs to `models/`.
   - Saves both `.keras` and `.h5` model files.

3. **Inference (web + desktop)**
   - Web: `web_server.py` uses Flask-SocketIO; streams frames and predictions via WebSocket events.
   - Desktop: `src/gui.py` runs capture/inference in a background thread and updates Tkinter UI.
   - Shared utility logic lives in `src/utils.py` (`PredictionSmoother`, landmark helpers, data balance checks).

### Configuration model

`config/config.py` is the central source for:
- class labels (`A-Z`),
- camera and MediaPipe thresholds,
- training hyperparameters,
- inference confidence/smoothing settings,
- model/data paths.

Most modules import this config object directly; behavioral changes are usually made here first.

### Web app integration details

- Flask serves the built frontend from `web_ui/dist` (`/` and static path fallback routes).
- Frontend (`web_ui/src/App.jsx`) connects to Socket.IO at `http://localhost:5000`.
- Real-time updates are pushed via `frame_update` events containing:
  - base64 JPEG frame,
  - prediction payload (`letter`, `confidence`, `stability`, hand presence),
  - text-builder state.
- Text editing controls (`add_space`, `delete_last`, `clear_text`) are event-driven over Socket.IO.

### Known repository gotchas

- `main.py` menu options for LSTM (`train_lstm`, `gui_lstm`) reference modules not present in this repository.
- `web_server.py` loads `config.MODEL_PATH` (`.h5`) directly; keep that file available for web inference.
- README content appears ahead of current code in some places (for example architecture/training details), so verify behavior against source files when implementing changes.
