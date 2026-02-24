"""
Configuration module for Sign Language Recognition System.
Centralizes all paths and parameters for easy modification.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ASL_Data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, "asl_landmark_dl_model.h5")
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "asl_landmark_dl_model.keras")

# Class names
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26

# Data collection settings
COLLECTION = {
    "auto_capture_interval_ms": 300,
    "stability_threshold": 0.05,  # 5% movement threshold
    "min_samples_per_class": 80,
    "target_samples_per_class": 150,
    "camera_index": 0,
    "detection_confidence": 0.7,
    "tracking_confidence": 0.7,
}

# Model training settings
TRAINING = {
    "mlp": {
        "hidden_layers": [256, 128],
        "dropout": [0.4, 0.3],
        "epochs": 30,
        "batch_size": 32,
        "validation_split": 0.2,
        "random_state": 42,
    },
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "restore_best_weights": True,
    }
}

# Inference settings
INFERENCE = {
    "confidence_threshold": 0.7,
    "smoothing_buffer_size": 10,  # Number of frames for majority voting
    "use_smoothing": True,
}

# GUI settings
GUI = {
    "window_title": "ASL Real-Time Recognition",
    "window_size": "900x600",
    "video_width": 700,
    "video_height": 400,
    "update_interval_ms": 10,
    "show_confidence": True,
}

# MediaPipe settings
MEDIAPIPE = {
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
}
