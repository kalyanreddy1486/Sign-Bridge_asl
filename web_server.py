"""
Flask-SocketIO Web Server for ASL Recognition
Provides real-time video streaming and predictions via WebSocket
"""

import os
import sys
import base64
import json
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import config
from utils import normalize_landmarks, extract_landmarks_from_hand, PredictionSmoother

_ALLOWED_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:3000",  # Vite dev server
    "http://127.0.0.1:3000",
]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": _ALLOWED_ORIGINS}})
socketio = SocketIO(app, cors_allowed_origins=_ALLOWED_ORIGINS, async_mode='threading')

# Global variables
model = None
hands = None
mp_drawing = None
camera = None
is_running = False
thread = None
smoother = PredictionSmoother()
phase2_pose_library = {}

# Text Builder State
text_builder_state = {
    "sentence": "",
    "current_letter": None,
    "letter_start_time": None,
    "last_add_time": 0,
    "no_hand_start_time": None,
    "stability_progress": 0.0,
    "in_cooldown": False,
    "cooldown_remaining": 0.0
}

# Configuration
STABILITY_TIME = 0.8  # seconds to hold sign
COOLDOWN_TIME = 1.5   # seconds after adding letter
CONFIDENCE_THRESHOLD = 0.8
SPACE_TIMEOUT = 1.2   # seconds no hand = space


def load_model():
    """Load the trained MLP model and warm it up with a dummy prediction."""
    global model
    try:
        model_path = config.MODEL_PATH
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[OK] Model loaded from: {model_path}")
            # Pre-warm: first TF predict is very slow due to graph tracing.
            # Running a dummy inference now pays that cost at startup instead
            # of on the user's first real frame after clicking Start Camera.
            try:
                dummy = np.zeros((1, 63), dtype=np.float32)
                model.predict(dummy, verbose=0)
                print("[OK] Model warmed up")
            except Exception as warm_err:
                print(f"[WARN] Model warm-up failed: {warm_err}")
            return True
        else:
            print(f"[ERR] Model not found at: {model_path}")
            return False
    except Exception as e:
        print(f"[ERR] Error loading model: {e}")
        return False


def initialize_mediapipe():
    """Initialize MediaPipe hands."""
    global hands, mp_drawing
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=config.MEDIAPIPE["max_num_hands"],
        min_detection_confidence=config.MEDIAPIPE["min_detection_confidence"],
        min_tracking_confidence=config.MEDIAPIPE["min_tracking_confidence"]
    )
    try:
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hands.process(dummy_rgb)
        print("[OK] MediaPipe initialized (pre-warmed)")
    except Exception as warm_err:
        print(f"[OK] MediaPipe initialized (warm-up skipped: {warm_err})")


def load_phase2_pose_library():
    """Load canonical per-letter hand poses from existing ASL landmark samples."""
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
    """
    Convert typed text into a sequence of canonical hand poses.
    Injects a brief 'rest' (open-palm) entry between every letter so the
    3D hand returns to a neutral position before forming the next sign.
    speed_multiplier: float in [0.25, 3.0] — divides all duration_ms values.
    """
    # Clamp speed multiplier to safe range
    speed_multiplier = max(0.25, min(3.0, float(speed_multiplier)))

    sanitized = (text or "").upper()
    max_length = config.PHASE2.get("max_text_length", 120)
    sanitized = sanitized[:max_length]

    base_letter_ms  = config.PHASE2.get("letter_hold_ms", 700)
    base_space_ms   = config.PHASE2.get("space_hold_ms", 400)
    base_rest_ms    = 200  # neutral pause between letters

    # Apply speed — faster speed = shorter durations
    letter_ms = max(100, int(base_letter_ms / speed_multiplier))
    space_ms  = max(80,  int(base_space_ms  / speed_multiplier))
    rest_ms   = max(60,  int(base_rest_ms   / speed_multiplier))

    REST_ENTRY = {"type": "rest", "duration_ms": rest_ms}

    sequence   = []
    unsupported = []
    last_was_letter = False  # track whether to inject rest before next letter

    for char in sanitized:
        if char == " ":
            # Space — no rest needed around it, it acts as its own pause
            sequence.append({
                "type": "space",
                "letter": " ",
                "duration_ms": space_ms
            })
            last_was_letter = False
            continue

        if char in phase2_pose_library:
            # Inject rest pose BEFORE this letter (if previous was also a letter)
            if last_was_letter:
                sequence.append(REST_ENTRY.copy())

            sequence.append({
                "type": "letter",
                "letter": char,
                "pose": phase2_pose_library[char],
                "duration_ms": letter_ms
            })
            last_was_letter = True
        else:
            # Letter has no pose data — emit an unsupported entry (don't skip silently)
            if last_was_letter:
                sequence.append(REST_ENTRY.copy())
            sequence.append({
                "type": "unsupported",
                "letter": char,
                "duration_ms": letter_ms  # still takes up time in sequence
            })
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
        "library_size": len(phase2_pose_library)
    }


def update_text_builder(letter, confidence, hand_detected):
    """Update text builder state."""
    global text_builder_state
    
    current_time = time.time()
    result = {
        "letter_added": None,
        "stability_progress": 0.0,
        "in_cooldown": False,
        "cooldown_remaining": 0.0
    }
    
    # Check cooldown
    time_since_add = current_time - text_builder_state["last_add_time"]
    if time_since_add < COOLDOWN_TIME:
        result["in_cooldown"] = True
        result["cooldown_remaining"] = COOLDOWN_TIME - time_since_add
        text_builder_state.update(result)
        return result
    
    # Handle no hand - potential space
    if not hand_detected:
        if text_builder_state["no_hand_start_time"] is None:
            text_builder_state["no_hand_start_time"] = current_time
        elif current_time - text_builder_state["no_hand_start_time"] >= SPACE_TIMEOUT:
            if text_builder_state["sentence"] and not text_builder_state["sentence"].endswith(" "):
                text_builder_state["sentence"] += " "
                result["letter_added"] = "[SPACE]"
                text_builder_state["last_add_time"] = current_time
                result["in_cooldown"] = True
                result["cooldown_remaining"] = COOLDOWN_TIME
            text_builder_state["no_hand_start_time"] = None
        
        text_builder_state["current_letter"] = None
        text_builder_state["letter_start_time"] = None
        text_builder_state.update(result)
        return result
    
    # Hand detected
    text_builder_state["no_hand_start_time"] = None
    
    # Check confidence
    if confidence < CONFIDENCE_THRESHOLD:
        text_builder_state["current_letter"] = None
        text_builder_state["letter_start_time"] = None
        text_builder_state.update(result)
        return result
    
    # Same letter?
    if letter == text_builder_state["current_letter"]:
        if text_builder_state["letter_start_time"]:
            elapsed = current_time - text_builder_state["letter_start_time"]
            result["stability_progress"] = min(1.0, elapsed / STABILITY_TIME)
            
            if elapsed >= STABILITY_TIME:
                text_builder_state["sentence"] += letter
                result["letter_added"] = letter
                text_builder_state["last_add_time"] = current_time
                text_builder_state["current_letter"] = None
                text_builder_state["letter_start_time"] = None
                result["in_cooldown"] = True
                result["cooldown_remaining"] = COOLDOWN_TIME
    else:
        text_builder_state["current_letter"] = letter
        text_builder_state["letter_start_time"] = current_time
        result["stability_progress"] = 0.0
    
    text_builder_state.update(result)
    return result


def process_frame():
    """Process a single frame and return results."""
    global camera, is_running
    
    if camera is None or not camera.isOpened():
        return None
    
    ret, frame = camera.read()
    if not ret:
        return None
    
    # Flip and convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    result = hands.process(rgb)
    
    prediction_data = {
        "hand_detected": False,
        "letter": None,
        "confidence": 0.0,
        "stability": 0.0,
        "landmarks": None
    }
    
    if result.multi_hand_landmarks:
        prediction_data["hand_detected"] = True
        
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp.solutions.hands.HAND_CONNECTIONS
            )
            
            # Extract and predict
            raw_landmarks = extract_landmarks_from_hand(hand_landmarks)
            keypoints = normalize_landmarks(raw_landmarks)
            keypoints = keypoints.reshape(1, -1)
            
            try:
                pred = model.predict(keypoints, verbose=0)
                class_id = np.argmax(pred)
                confidence = float(pred[0][class_id])
                
                smoother.add_prediction(class_id, confidence)
                smoothed_id, smoothed_conf, stability = smoother.get_smoothed_prediction()
                
                if smoothed_conf >= config.INFERENCE["confidence_threshold"]:
                    prediction_data["letter"] = config.CLASS_NAMES[smoothed_id]
                    prediction_data["confidence"] = smoothed_conf
                    prediction_data["stability"] = stability
                    
            except Exception as e:
                print(f"Prediction error: {e}")
    else:
        smoother.clear()
    
    # Update text builder
    builder_result = update_text_builder(
        prediction_data["letter"],
        prediction_data["confidence"],
        prediction_data["hand_detected"]
    )
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "frame": f"data:image/jpeg;base64,{frame_base64}",
        "prediction": prediction_data,
        "text_builder": {
            "sentence": text_builder_state["sentence"],
            **builder_result
        }
    }


def video_stream_thread():
    """Background thread for video streaming."""
    global is_running, camera

    cam_index = config.COLLECTION["camera_index"]
    # On Windows, DirectShow opens the webcam ~2s faster than the default MSMF backend
    if sys.platform == 'win32':
        camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(cam_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("[ERR] Failed to open camera")
        socketio.emit('error', {'message': 'Failed to open camera'})
        socketio.emit('camera_status', {'status': 'stopped'})
        return

    print("[OK] Camera opened")
    socketio.emit('camera_status', {'status': 'started'})
    is_running = True

    try:
        while is_running:
            try:
                data = process_frame()
                if data:
                    socketio.emit('frame_update', data)
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(0.1)
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception as rel_err:
                print(f"[WARN] Camera release error: {rel_err}")
        camera = None
        is_running = False
        print("[OK] Video stream stopped")


# Flask Routes
@app.route('/')
def index():
    """Serve the main page (Phase 1 — sign detection)."""
    return send_from_directory('web_ui/dist', 'index.html')


@app.route('/phase2/')
@app.route('/phase2')
def phase2_index():
    """Serve the Phase 2 word-to-sign video player."""
    return send_from_directory('phase2', 'index.html')


@app.route('/phase2/<path:path>')
def phase2_files(path):
    """Serve Phase 2 static files (CSS, JS, MP4 videos)."""
    return send_from_directory('phase2', path)


@app.route('/<path:path>')
def static_files(path):
    """Serve Phase 1 static files."""
    return send_from_directory('web_ui/dist', path)


# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('[OK] Client connected')
    emit('connected', {'status': 'connected', 'model_loaded': model is not None})
    emit('phase2_library_status', {
        'available': list(phase2_pose_library.keys()),
        'letters_available': len(phase2_pose_library),
        'total': 26
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('[ERR] Client disconnected')


@socketio.on('start_camera')
def handle_start_camera():
    """Start the camera stream."""
    global thread, is_running
    
    if thread is None or not thread.is_alive():
        is_running = False
        time.sleep(0.1)
        thread = threading.Thread(target=video_stream_thread)
        thread.daemon = True
        thread.start()
        emit('camera_status', {'status': 'opening'})
    else:
        emit('camera_status', {'status': 'already_running'})


@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop the camera stream."""
    global is_running
    is_running = False
    emit('camera_status', {'status': 'stopped'})


@socketio.on('clear_text')
def handle_clear_text():
    """Clear the text builder sentence."""
    global text_builder_state
    text_builder_state["sentence"] = ""
    emit('text_cleared', {'sentence': ''})


@socketio.on('delete_last')
def handle_delete_last():
    """Delete last character."""
    global text_builder_state
    if text_builder_state["sentence"]:
        text_builder_state["sentence"] = text_builder_state["sentence"][:-1]
    emit('text_updated', {'sentence': text_builder_state["sentence"]})


@socketio.on('add_space')
def handle_add_space():
    """Add space to sentence."""
    global text_builder_state
    if text_builder_state["sentence"] and not text_builder_state["sentence"].endswith(" "):
        text_builder_state["sentence"] += " "
    emit('text_updated', {'sentence': text_builder_state["sentence"]})


@socketio.on('phase2_request_pose_sequence')
def handle_phase2_request_pose_sequence(data):
    """Return typed text playback sequence for Phase 2 3D visualization."""
    if not config.PHASE2.get("enabled", True):
        emit('phase2_pose_sequence', {
            "ok": False,
            "message": "Phase 2 is disabled",
            "sequence": []
        })
        return

    payload = data if isinstance(data, dict) else {}
    text  = payload.get("text", "")
    speed = payload.get("speed", 1.0)  # read speed multiplier from frontend

    result = build_phase2_pose_sequence(text, speed_multiplier=speed)

    # Check whether there are any playable items (letter, unsupported, space)
    playable = [item for item in result["sequence"] if item["type"] != "rest"]
    if not playable:
        emit('phase2_pose_sequence', {
            "ok": False,
            "message": "No valid characters found in input",
            **result
        })
        return

    emit('phase2_pose_sequence', {
        "ok": True,
        "message": f"Sequence ready: {len(playable)} symbols at {speed}x speed",
        **result
    })


if __name__ == '__main__':
    print("="*60)
    print("ASL Recognition Web Server")
    print("="*60)
    
    # Initialize
    if load_model():
        initialize_mediapipe()
        load_phase2_pose_library()
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', '5000'))
        debug = os.environ.get('DEBUG', 'false').lower() == 'true'
        print("\n[OK] Server ready")
        print(f"-> Open http://localhost:{port} in your browser")
        print("="*60)
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        print("\n[ERR] Failed to start server")
