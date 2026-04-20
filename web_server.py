"""
Flask-SocketIO Web Server for ASL Recognition.

Architecture: the browser owns the webcam (via getUserMedia) and pushes
JPEG frames over Socket.IO. The server runs MediaPipe + the trained MLP
on each received frame and emits back a prediction + text-builder state.
No camera is read on the server, so this deploys to any cloud host.
"""

import os
import sys
import base64
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as _mp_hands_force_import  # noqa: F401
import tensorflow as tf
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import config
from utils import normalize_landmarks, extract_landmarks_from_hand, PredictionSmoother

# When the server and the frontend share an origin (i.e. Flask serves the
# built React bundle, which is our deploy story), the Socket.IO handshake
# Origin is just that same origin. Allow any origin — nothing sensitive
# to protect, and it keeps HF Spaces / Render / Railway / etc. all working
# without per-platform config.
_ALLOWED_ORIGINS = "*"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": _ALLOWED_ORIGINS}})
socketio = SocketIO(app, cors_allowed_origins=_ALLOWED_ORIGINS, async_mode='threading',
                    max_http_buffer_size=2_000_000)

# Globals loaded once at startup, shared (read-only) across sessions.
model = None
hands = None
phase2_pose_library = {}

# MediaPipe Hands is NOT thread-safe and keeps internal timestamp state.
# Socket.IO runs each 'video_frame' event on a worker thread, so concurrent
# visitors would race hands.process() and permanently corrupt its graph
# ("Packet timestamp mismatch ..."). Serialize every call behind this lock.
hands_lock = threading.Lock()

# Per-session state, keyed by Socket.IO session id. Multiple HF Spaces
# visitors each get their own smoother + text builder.
sessions = {}

STABILITY_TIME = 0.8
COOLDOWN_TIME = 1.5
CONFIDENCE_THRESHOLD = 0.8
SPACE_TIMEOUT = 1.2


def _new_session_state():
    """Fresh state for a new Socket.IO connection."""
    return {
        "smoother": PredictionSmoother(),
        "text_builder": {
            "sentence": "",
            "current_letter": None,
            "letter_start_time": None,
            "last_add_time": 0.0,
            "no_hand_start_time": None,
        },
    }


def load_model():
    global model
    try:
        model_path = config.MODEL_PATH
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[OK] Model loaded from: {model_path}")
            try:
                dummy = np.zeros((1, 63), dtype=np.float32)
                model.predict(dummy, verbose=0)
                print("[OK] Model warmed up")
            except Exception as warm_err:
                print(f"[WARN] Model warm-up failed: {warm_err}")
            return True
        print(f"[ERR] Model not found at: {model_path}")
        return False
    except Exception as e:
        print(f"[ERR] Error loading model: {e}")
        return False


def initialize_mediapipe():
    global hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=config.MEDIAPIPE["max_num_hands"],
        min_detection_confidence=config.MEDIAPIPE["min_detection_confidence"],
        min_tracking_confidence=config.MEDIAPIPE["min_tracking_confidence"],
    )
    try:
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hands.process(dummy_rgb)
        print("[OK] MediaPipe initialized (pre-warmed)")
    except Exception as warm_err:
        print(f"[OK] MediaPipe initialized (warm-up skipped: {warm_err})")


def load_phase2_pose_library():
    global phase2_pose_library

    loaded = 0
    phase2_pose_library = {}

    for letter in config.CLASS_NAMES:
        letter_dir = os.path.join(config.DATA_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue

        samples = []
        for file_name in os.listdir(letter_dir):
            if not file_name.endswith('.npy'):
                continue
            sample_path = os.path.join(letter_dir, file_name)
            try:
                sample = np.load(sample_path)
                sample = np.array(sample, dtype=np.float32).reshape(-1)
                if sample.size < 63:
                    continue
                sample = sample[:63]
                samples.append(normalize_landmarks(sample))
            except Exception:
                continue

        if not samples:
            continue

        stacked = np.stack(samples, axis=0)
        canonical = np.median(stacked, axis=0)
        precision = config.PHASE2.get("pose_precision", 5)
        canonical = np.round(canonical, precision)
        phase2_pose_library[letter] = canonical.astype(np.float32).tolist()
        loaded += 1

    print(f"[OK] Phase 2 pose library loaded: {loaded}/{len(config.CLASS_NAMES)} letters")
    return loaded > 0


def build_phase2_pose_sequence(text, speed_multiplier=1.0):
    speed_multiplier = max(0.25, min(3.0, float(speed_multiplier)))
    sanitized = (text or "").upper()
    max_length = config.PHASE2.get("max_text_length", 120)
    sanitized = sanitized[:max_length]

    base_letter_ms = config.PHASE2.get("letter_hold_ms", 700)
    base_space_ms = config.PHASE2.get("space_hold_ms", 400)
    base_rest_ms = 200

    letter_ms = max(100, int(base_letter_ms / speed_multiplier))
    space_ms = max(80, int(base_space_ms / speed_multiplier))
    rest_ms = max(60, int(base_rest_ms / speed_multiplier))

    REST_ENTRY = {"type": "rest", "duration_ms": rest_ms}

    sequence = []
    unsupported = []
    last_was_letter = False

    for char in sanitized:
        if char == " ":
            sequence.append({"type": "space", "letter": " ", "duration_ms": space_ms})
            last_was_letter = False
            continue

        if char in phase2_pose_library:
            if last_was_letter:
                sequence.append(REST_ENTRY.copy())
            sequence.append({
                "type": "letter",
                "letter": char,
                "pose": phase2_pose_library[char],
                "duration_ms": letter_ms,
            })
            last_was_letter = True
        else:
            if last_was_letter:
                sequence.append(REST_ENTRY.copy())
            sequence.append({"type": "unsupported", "letter": char, "duration_ms": letter_ms})
            unsupported.append(char)
            last_was_letter = True

    return {
        "input_text": text or "",
        "sanitized_text": sanitized,
        "sequence": sequence,
        "unsupported": unsupported,
        "letter_hold_ms": letter_ms,
        "space_hold_ms": space_ms,
        "rest_ms": rest_ms,
        "speed_multiplier": speed_multiplier,
        "library_size": len(phase2_pose_library),
    }


def update_text_builder(state, letter, confidence, hand_detected):
    """Advance the text-builder state machine by one frame. Mutates `state`."""
    current_time = time.time()
    result = {
        "letter_added": None,
        "stability_progress": 0.0,
        "in_cooldown": False,
        "cooldown_remaining": 0.0,
    }

    time_since_add = current_time - state["last_add_time"]
    if time_since_add < COOLDOWN_TIME:
        result["in_cooldown"] = True
        result["cooldown_remaining"] = COOLDOWN_TIME - time_since_add
        return result

    if not hand_detected:
        if state["no_hand_start_time"] is None:
            state["no_hand_start_time"] = current_time
        elif current_time - state["no_hand_start_time"] >= SPACE_TIMEOUT:
            if state["sentence"] and not state["sentence"].endswith(" "):
                state["sentence"] += " "
                result["letter_added"] = "[SPACE]"
                state["last_add_time"] = current_time
                result["in_cooldown"] = True
                result["cooldown_remaining"] = COOLDOWN_TIME
            state["no_hand_start_time"] = None

        state["current_letter"] = None
        state["letter_start_time"] = None
        return result

    state["no_hand_start_time"] = None

    if confidence < CONFIDENCE_THRESHOLD:
        state["current_letter"] = None
        state["letter_start_time"] = None
        return result

    if letter == state["current_letter"]:
        if state["letter_start_time"]:
            elapsed = current_time - state["letter_start_time"]
            result["stability_progress"] = min(1.0, elapsed / STABILITY_TIME)
            if elapsed >= STABILITY_TIME:
                state["sentence"] += letter
                result["letter_added"] = letter
                state["last_add_time"] = current_time
                state["current_letter"] = None
                state["letter_start_time"] = None
                result["in_cooldown"] = True
                result["cooldown_remaining"] = COOLDOWN_TIME
    else:
        state["current_letter"] = letter
        state["letter_start_time"] = current_time
        result["stability_progress"] = 0.0

    return result


def _decode_frame(data_url_or_b64):
    """Accept a 'data:image/jpeg;base64,...' URL or a raw b64 string, return BGR ndarray."""
    if not isinstance(data_url_or_b64, str):
        return None
    s = data_url_or_b64.split(",", 1)[1] if data_url_or_b64.startswith("data:") else data_url_or_b64
    try:
        img_bytes = base64.b64decode(s)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[WARN] frame decode failed: {e}")
        return None


def process_client_frame(state, bgr_frame):
    """Run MediaPipe + model + text-builder on a client-supplied frame."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    with hands_lock:
        result = hands.process(rgb)

    prediction = {
        "hand_detected": False,
        "letter": None,
        "confidence": 0.0,
        "stability": 0.0,
    }

    smoother = state["smoother"]

    if result.multi_hand_landmarks:
        prediction["hand_detected"] = True
        hand_landmarks = result.multi_hand_landmarks[0]
        raw_landmarks = extract_landmarks_from_hand(hand_landmarks)
        keypoints = normalize_landmarks(raw_landmarks).reshape(1, -1)
        try:
            pred = model.predict(keypoints, verbose=0)
            class_id = int(np.argmax(pred))
            confidence = float(pred[0][class_id])
            smoother.add_prediction(class_id, confidence)
            smoothed_id, smoothed_conf, stability = smoother.get_smoothed_prediction()
            if smoothed_id is not None and smoothed_conf >= config.INFERENCE["confidence_threshold"]:
                prediction["letter"] = config.CLASS_NAMES[smoothed_id]
                prediction["confidence"] = smoothed_conf
                prediction["stability"] = stability
        except Exception as e:
            print(f"[WARN] prediction error: {e}")
    else:
        smoother.clear()

    builder_result = update_text_builder(
        state["text_builder"],
        prediction["letter"],
        prediction["confidence"],
        prediction["hand_detected"],
    )

    return {
        "prediction": prediction,
        "text_builder": {
            "sentence": state["text_builder"]["sentence"],
            **builder_result,
        },
    }


# -----------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('web_ui/dist', 'index.html')


@app.route('/phase2/')
@app.route('/phase2')
def phase2_index():
    return send_from_directory('phase2', 'index.html')


@app.route('/phase2/<path:path>')
def phase2_files(path):
    return send_from_directory('phase2', path)


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('web_ui/dist', path)


# -----------------------------------------------------------------------
# Socket.IO events
# -----------------------------------------------------------------------

def _sid():
    from flask import request
    return request.sid


@socketio.on('connect')
def handle_connect():
    sessions[_sid()] = _new_session_state()
    print(f"[OK] Client connected ({_sid()}, total sessions: {len(sessions)})")
    emit('connected', {'status': 'connected', 'model_loaded': model is not None})
    emit('phase2_library_status', {
        'available': list(phase2_pose_library.keys()),
        'letters_available': len(phase2_pose_library),
        'total': 26,
    })


@socketio.on('disconnect')
def handle_disconnect():
    sessions.pop(_sid(), None)
    print(f"[INFO] Client disconnected ({_sid()}, remaining sessions: {len(sessions)})")


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Client pushed a JPEG frame from its webcam. Decode, run ML, emit prediction.
    `data` is either a data-URL string or {'frame': '...'}.
    """
    state = sessions.get(_sid())
    if state is None or model is None or hands is None:
        return

    payload = data.get('frame') if isinstance(data, dict) else data
    frame = _decode_frame(payload)
    if frame is None:
        return

    result = process_client_frame(state, frame)
    emit('frame_update', result)


@socketio.on('reset_stream')
def handle_reset_stream():
    """Client stopped its camera — clear session smoothing + in-progress letter."""
    state = sessions.get(_sid())
    if state is None:
        return
    state["smoother"].clear()
    state["text_builder"]["current_letter"] = None
    state["text_builder"]["letter_start_time"] = None
    state["text_builder"]["no_hand_start_time"] = None


@socketio.on('clear_text')
def handle_clear_text():
    state = sessions.get(_sid())
    if state is None:
        return
    state["text_builder"]["sentence"] = ""
    emit('text_cleared', {'sentence': ''})


@socketio.on('delete_last')
def handle_delete_last():
    state = sessions.get(_sid())
    if state is None:
        return
    tb = state["text_builder"]
    if tb["sentence"]:
        tb["sentence"] = tb["sentence"][:-1]
    emit('text_updated', {'sentence': tb["sentence"]})


@socketio.on('add_space')
def handle_add_space():
    state = sessions.get(_sid())
    if state is None:
        return
    tb = state["text_builder"]
    if tb["sentence"] and not tb["sentence"].endswith(" "):
        tb["sentence"] += " "
    emit('text_updated', {'sentence': tb["sentence"]})


@socketio.on('phase2_request_pose_sequence')
def handle_phase2_request_pose_sequence(data):
    if not config.PHASE2.get("enabled", True):
        emit('phase2_pose_sequence', {"ok": False, "message": "Phase 2 is disabled", "sequence": []})
        return

    payload = data if isinstance(data, dict) else {}
    text = payload.get("text", "")
    speed = payload.get("speed", 1.0)
    result = build_phase2_pose_sequence(text, speed_multiplier=speed)

    playable = [item for item in result["sequence"] if item["type"] != "rest"]
    if not playable:
        emit('phase2_pose_sequence', {"ok": False, "message": "No valid characters found in input", **result})
        return

    emit('phase2_pose_sequence', {
        "ok": True,
        "message": f"Sequence ready: {len(playable)} symbols at {speed}x speed",
        **result,
    })


if __name__ == '__main__':
    print("=" * 60)
    print("ASL Recognition Web Server")
    print("=" * 60)

    if load_model():
        initialize_mediapipe()
        load_phase2_pose_library()
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', '5000'))
        debug = os.environ.get('DEBUG', 'false').lower() == 'true'
        print("\n[OK] Server ready")
        print(f"-> Open http://localhost:{port} in your browser")
        print("=" * 60)
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        print("\n[ERR] Failed to start server")
