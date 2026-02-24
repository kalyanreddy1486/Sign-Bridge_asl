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

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
model = None
hands = None
mp_drawing = None
camera = None
is_running = False
thread = None
smoother = PredictionSmoother()

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
    """Load the trained MLP model."""
    global model
    try:
        model_path = config.MODEL_PATH
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded from: {model_path}")
            return True
        else:
            print(f"✗ Model not found at: {model_path}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
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
    print("✓ MediaPipe initialized")


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
    
    camera = cv2.VideoCapture(config.COLLECTION["camera_index"])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print("✗ Failed to open camera")
        socketio.emit('error', {'message': 'Failed to open camera'})
        return
    
    print("✓ Camera opened")
    is_running = True
    
    while is_running:
        try:
            data = process_frame()
            if data:
                socketio.emit('frame_update', data)
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)
    
    if camera:
        camera.release()
    print("✓ Video stream stopped")


# Flask Routes
@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('web_ui/dist', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Serve static files."""
    return send_from_directory('web_ui/dist', path)


# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('✓ Client connected')
    emit('connected', {'status': 'connected', 'model_loaded': model is not None})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('✗ Client disconnected')


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
        emit('camera_status', {'status': 'started'})
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


if __name__ == '__main__':
    print("="*60)
    print("ASL Recognition Web Server")
    print("="*60)
    
    # Initialize
    if load_model():
        initialize_mediapipe()
        print("\n✓ Server ready")
        print("→ Open http://localhost:5000 in your browser")
        print("="*60)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        print("\n✗ Failed to start server")
