"""
Improved GUI with prediction smoothing, error handling, and better UX.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import threading
import queue

from config import config
from utils import (
    PredictionSmoother, 
    normalize_landmarks, 
    extract_landmarks_from_hand,
    check_data_balance
)


class VideoThread(threading.Thread):
    """
    Thread for video capture and inference.
    Runs separately from GUI to prevent freezing.
    """
    
    def __init__(self, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = False
        self.cap = None
        self.model = None
        self.smoother = PredictionSmoother()
        self.hands = None
        
    def load_model(self):
        """Load the trained model."""
        try:
            # Try modern format first
            if os.path.exists(config.MODEL_PATH_KERAS):
                self.model = tf.keras.models.load_model(config.MODEL_PATH_KERAS)
                print(f"Loaded model from: {config.MODEL_PATH_KERAS}")
            # Fall back to .h5 format
            elif os.path.exists(config.MODEL_PATH):
                self.model = tf.keras.models.load_model(config.MODEL_PATH)
                print(f"Loaded model from: {config.MODEL_PATH}")
            else:
                raise FileNotFoundError("No model file found!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.result_queue.put({"error": f"Model loading failed: {e}"})
            return False
    
    def initialize_camera(self):
        """Initialize camera capture."""
        try:
            print(f"Video thread: Opening camera at index {config.COLLECTION['camera_index']}")
            self.cap = cv2.VideoCapture(config.COLLECTION["camera_index"])
            
            if not self.cap.isOpened():
                error_msg = f"Could not open camera at index {config.COLLECTION['camera_index']}"
                print(f"Video thread: {error_msg}")
                self.result_queue.put({"error": error_msg})
                return False
            
            print("Video thread: Camera opened successfully")
            
            # Set resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test read
            ret, test_frame = self.cap.read()
            if not ret:
                error_msg = "Camera opened but cannot read frames"
                print(f"Video thread: {error_msg}")
                self.result_queue.put({"error": error_msg})
                return False
            
            print(f"Video thread: Camera test read successful, frame shape: {test_frame.shape}")
            return True
            
        except Exception as e:
            error_msg = f"Camera initialization failed: {e}"
            print(f"Video thread: {error_msg}")
            self.result_queue.put({"error": error_msg})
            return False
    
    def initialize_mediapipe(self):
        """Initialize MediaPipe hands."""
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=config.MEDIAPIPE["max_num_hands"],
            min_detection_confidence=config.MEDIAPIPE["min_detection_confidence"],
            min_tracking_confidence=config.MEDIAPIPE["min_tracking_confidence"]
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def run(self):
        """Main thread loop."""
        # Initialize
        print("Video thread: Loading model...")
        if not self.load_model():
            print("Video thread: Model loading failed, exiting")
            return
        
        print("Video thread: Initializing camera...")
        if not self.initialize_camera():
            print("Video thread: Camera initialization failed, exiting")
            return
        
        print("Video thread: Initializing MediaPipe...")
        self.initialize_mediapipe()
        
        self.running = True
        print("Video thread: Started successfully")
        
        frame_count = 0
        frames_sent = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                frame_count += 1
                if frame_count > 30:  # Too many failed frames
                    print("Video thread: Too many frame read failures, stopping")
                    break
                continue
            frame_count = 0
            
            # Debug: print every 30 frames
            frames_sent += 1
            if frames_sent % 30 == 0:
                print(f"Video thread: Sent {frames_sent} frames")
            
            # Process frame
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            
            prediction_data = {
                "hand_detected": False,
                "letter": None,
                "confidence": 0.0,
                "stability": 0.0
            }
            
            if result.multi_hand_landmarks:
                prediction_data["hand_detected"] = True
                
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
                    
                    # Extract and normalize landmarks
                    raw_landmarks = extract_landmarks_from_hand(hand_landmarks)
                    keypoints = normalize_landmarks(raw_landmarks)
                    keypoints = keypoints.reshape(1, -1)
                    
                    # Predict
                    try:
                        prediction = self.model.predict(keypoints, verbose=0)
                        class_id = np.argmax(prediction)
                        confidence = prediction[0][class_id]
                        
                        # Add to smoother
                        self.smoother.add_prediction(class_id, confidence)
                        
                        # Get smoothed prediction
                        if config.INFERENCE["use_smoothing"]:
                            smoothed_id, smoothed_conf, stability = self.smoother.get_smoothed_prediction()
                        else:
                            smoothed_id, smoothed_conf, stability = class_id, confidence, 1.0
                        
                        if smoothed_conf >= config.INFERENCE["confidence_threshold"]:
                            prediction_data["letter"] = config.CLASS_NAMES[smoothed_id]
                            prediction_data["confidence"] = smoothed_conf
                            prediction_data["stability"] = stability
                            
                    except Exception as e:
                        print(f"Prediction error: {e}")
            else:
                # No hand detected - clear smoother
                self.smoother.clear()
            
            # Send frame and prediction to GUI
            try:
                self.frame_queue.put_nowait((frame, prediction_data))
            except queue.Full:
                # Drop frame if GUI can't keep up
                pass
        
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        print("Video thread stopped")
    
    def stop(self):
        """Stop the thread."""
        self.running = False


class ASLGUI:
    """Main GUI application for ASL recognition."""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title(config.GUI["window_title"])
        self.window.geometry(config.GUI["window_size"])
        self.window.resizable(False, False)
        
        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        
        # Video thread
        self.video_thread = None
        
        self.setup_ui()
        self.check_errors()
        
        # Auto-start camera if model exists
        if os.path.exists(config.MODEL_PATH) or os.path.exists(config.MODEL_PATH_KERAS):
            self.window.after(500, self.start_camera)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Video frame with fixed size
        self.video_frame = Frame(
            self.window,
            width=config.GUI["video_width"],
            height=config.GUI["video_height"],
            bg="black"
        )
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Video display inside frame
        self.video_label = Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Status frame
        status_frame = Frame(self.window)
        status_frame.pack(pady=5)
        
        # Hand detection indicator
        self.hand_indicator = Label(
            status_frame, 
            text="● No Hand", 
            font=("Arial", 12),
            fg="red"
        )
        self.hand_indicator.pack(side=tk.LEFT, padx=10)
        
        # Stability indicator
        self.stability_label = Label(
            status_frame,
            text="Stability: -",
            font=("Arial", 12),
            fg="gray"
        )
        self.stability_label.pack(side=tk.LEFT, padx=10)
        
        # Prediction display
        self.prediction_label = Label(
            self.window,
            text="Prediction: -",
            font=("Arial", 28, "bold"),
            fg="green"
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence bar
        self.confidence_frame = Frame(self.window)
        self.confidence_frame.pack(pady=5)
        
        Label(
            self.confidence_frame,
            text="Confidence:",
            font=("Arial", 12)
        ).pack(side=tk.LEFT)
        
        self.confidence_bar = Frame(
            self.confidence_frame,
            width=200,
            height=20,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=1
        )
        self.confidence_bar.pack(side=tk.LEFT, padx=5)
        
        self.confidence_fill = Frame(
            self.confidence_bar,
            width=0,
            height=18,
            bg="green"
        )
        self.confidence_fill.place(x=1, y=1)
        
        self.confidence_text = Label(
            self.confidence_frame,
            text="0%",
            font=("Arial", 12),
            width=5
        )
        self.confidence_text.pack(side=tk.LEFT)
        
        # Info label
        self.info_label = Label(
            self.window,
            text="Initializing camera...",
            font=("Arial", 10),
            fg="gray"
        )
        self.info_label.pack(pady=5)
    
    def start_camera(self):
        """Start the camera and video thread."""
        try:
            self.video_thread = VideoThread(self.frame_queue, self.result_queue)
            self.video_thread.start()
            
            # Wait a bit for thread to initialize
            import time
            time.sleep(0.5)
            
            # Check if thread is still running
            if not self.video_thread.is_alive():
                # Check for errors in queue
                try:
                    error_msg = self.result_queue.get_nowait()
                    if "error" in error_msg:
                        self.info_label.config(text=f"Error: {error_msg['error']}", fg="red")
                        messagebox.showerror("Error", error_msg["error"])
                        return
                except queue.Empty:
                    pass
                self.info_label.config(text="Camera failed to start. Check camera connection.", fg="red")
                return
            
            self.info_label.config(text="Camera running. Show hand signs to predict.", fg="green")
            
            # Start GUI update loop
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            self.info_label.config(text=f"Error: {e}", fg="red")
    
    def stop_camera(self):
        """Stop the camera and video thread."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
        
        self.info_label.config(text="Camera stopped.")
        
        # Reset display
        self.prediction_label.config(text="Prediction: -")
        self.hand_indicator.config(text="● No Hand", fg="red")
        self.stability_label.config(text="Stability: -")
        self.update_confidence_bar(0)
        
        # Clear video - show black background
        self.video_label.config(image="")
        self.video_label.config(bg="black")
    
    def update_gui(self):
        """Update GUI with new frame and prediction."""
        if self.video_thread:
            try:
                # Get frame and prediction from queue
                frame, prediction_data = self.frame_queue.get_nowait()
                
                # Update video display
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((config.GUI["video_width"], config.GUI["video_height"]))
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Keep reference to prevent garbage collection
                self.current_image = imgtk
                self.video_label.configure(image=imgtk)
                
                # Update hand detection indicator
                if prediction_data["hand_detected"]:
                    self.hand_indicator.config(text="● Hand Detected", fg="green")
                else:
                    self.hand_indicator.config(text="● No Hand", fg="red")
                
                # Update prediction
                if prediction_data["letter"]:
                    letter = prediction_data["letter"]
                    conf = prediction_data["confidence"]
                    stability = prediction_data["stability"]
                    
                    self.prediction_label.config(
                        text=f"Prediction: {letter}",
                        fg="green" if conf > 0.8 else "orange" if conf > 0.7 else "red"
                    )
                    self.stability_label.config(
                        text=f"Stability: {stability:.0%}",
                        fg="green" if stability > 0.8 else "orange"
                    )
                    self.update_confidence_bar(conf)
                else:
                    self.prediction_label.config(text="Prediction: -", fg="gray")
                    self.stability_label.config(text="Stability: -", fg="gray")
                    self.update_confidence_bar(0)
                
            except queue.Empty:
                # No frame available yet, continue
                pass
            except Exception as e:
                print(f"GUI update error: {e}")
            
            # Check for errors from thread
            try:
                error_msg = self.result_queue.get_nowait()
                if "error" in error_msg:
                    messagebox.showerror("Error", error_msg["error"])
                    self.stop_camera()
                    return
            except queue.Empty:
                pass
            
            # Schedule next update (always continue if thread exists)
            if self.video_thread and self.video_thread.is_alive():
                self.window.after(config.GUI["update_interval_ms"], self.update_gui)
            else:
                self.info_label.config(text="Camera disconnected.", fg="red")
    
    def update_confidence_bar(self, confidence):
        """Update the confidence bar visualization."""
        max_width = 198  # Account for border
        fill_width = int(max_width * confidence)
        self.confidence_fill.config(width=fill_width)
        self.confidence_text.config(text=f"{confidence:.0%}")
        
        # Color based on confidence
        if confidence > 0.8:
            color = "green"
        elif confidence > 0.7:
            color = "orange"
        else:
            color = "red"
        self.confidence_fill.config(bg=color)
    
    def check_errors(self):
        """Check for common issues before starting."""
        # Check if model exists
        if not os.path.exists(config.MODEL_PATH) and not os.path.exists(config.MODEL_PATH_KERAS):
            self.info_label.config(
                text="⚠ Warning: No model found. Please train first.",
                fg="red"
            )
    
    def on_close(self):
        """Handle window close event."""
        self.stop_camera()
        self.window.destroy()
    
    def run(self):
        """Start the GUI."""
        self.window.mainloop()


def main():
    """Main entry point."""
    print("="*60)
    print("ASL Real-Time Recognition")
    print("="*60)
    
    # Check data balance
    print("\nChecking data balance...")
    balance = check_data_balance()
    print(f"Total samples: {balance['total']}")
    
    if balance['warnings']:
        print("\n⚠ Data imbalance detected:")
        for w in balance['warnings'][:3]:
            print(f"  - {w}")
        print("  (Consider collecting more data for underrepresented classes)")
    
    # Start GUI
    print("\nStarting GUI...")
    app = ASLGUI()
    app.run()


if __name__ == "__main__":
    main()
