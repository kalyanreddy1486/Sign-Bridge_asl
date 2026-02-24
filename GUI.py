import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os

# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(os.path.dirname(__file__), "asl_landmark_dl_model.h5")
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    return landmarks.flatten()

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# GUI WINDOW
# =========================
window = tk.Tk()
window.title("ASL Real-Time Recognition")
window.geometry("900x600")
window.resizable(False, False)

# =========================
# LABELS
# =========================
video_label = Label(window)
video_label.pack(pady=10)

prediction_label = Label(
    window,
    text="Prediction: None",
    font=("Arial", 24),
    fg="green"
)
prediction_label.pack(pady=10)

# =========================
# CAMERA CONTROL
# =========================
cap = None
running = False

def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()

def stop_camera():
    global cap, running
    running = False
    prediction_label.config(text="Prediction: None")
    if cap:
        cap.release()
        cap = None

def update_frame():
    global cap, running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # =========================
            # EXTRACT + NORMALIZE
            # =========================
            raw_landmarks = []
            for lm in hand_landmarks.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])

            keypoints = normalize_landmarks(raw_landmarks)
            keypoints = keypoints.reshape(1, -1)

            # =========================
            # PREDICTION
            # =========================
            prediction = model.predict(keypoints, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            letter = CLASS_NAMES[class_id]

            # =========================
            # DISPLAY
            # =========================
            if confidence > 0.7:
                prediction_label.config(
                    text=f"Prediction: {letter} ({confidence:.2f})"
                )
            else:
                prediction_label.config(text="Prediction: Detecting...")

    # Convert frame for Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((700, 400))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(10, update_frame)

# =========================
# BUTTONS
# =========================
button_frame = tk.Frame(window)
button_frame.pack(pady=20)

start_btn = Button(
    button_frame,
    text="Start Camera",
    font=("Arial", 14),
    bg="green",
    fg="white",
    command=start_camera
)
start_btn.grid(row=0, column=0, padx=20)

stop_btn = Button(
    button_frame,
    text="Stop Camera",
    font=("Arial", 14),
    bg="red",
    fg="white",
    command=stop_camera
)
stop_btn.grid(row=0, column=1, padx=20)

# =========================
# RUN GUI
# =========================
window.mainloop()
