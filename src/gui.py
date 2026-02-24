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
import time

from config import config
from utils import (
    PredictionSmoother, 
    normalize_landmarks, 
    extract_landmarks_from_hand,
    check_data_balance
)


class VideoThread(threading.Thread):
    """Video capture and inference thread for ASL recognition."""
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


class TextBuilder:
    """Text builder that converts stable predictions into typed text."""
    
    def __init__(self, stability_time=0.8, cooldown_time=1.5, confidence_threshold=0.8):
        self.stability_time = stability_time  # Time to hold sign before adding
        self.cooldown_time = cooldown_time    # Cooldown after adding letter
        self.confidence_threshold = confidence_threshold
        
        self.sentence = ""                    # Built sentence
        self.current_letter = None            # Currently detected letter
        self.letter_start_time = None         # When current letter was first seen
        self.last_add_time = 0                # When last letter was added
        self.no_hand_start_time = None        # When hand was lost (for space)
        self.space_timeout = 1.2              # No hand time to add space
        
    def update(self, letter, confidence, hand_detected):
        """
        Update with new prediction. Returns dict with status info.
        """
        current_time = time.time()
        result = {
            "letter_added": None,
            "stability_progress": 0.0,
            "in_cooldown": False,
            "cooldown_remaining": 0.0
        }
        
        # Check cooldown
        time_since_add = current_time - self.last_add_time
        if time_since_add < self.cooldown_time:
            result["in_cooldown"] = True
            result["cooldown_remaining"] = self.cooldown_time - time_since_add
            return result
        
        # Handle no hand - potential space
        if not hand_detected:
            if self.no_hand_start_time is None:
                self.no_hand_start_time = current_time
            elif current_time - self.no_hand_start_time >= self.space_timeout:
                # Add space if sentence exists and doesn't end with space
                if self.sentence and not self.sentence.endswith(" "):
                    self.sentence += " "
                    result["letter_added"] = "[SPACE]"
                    self.last_add_time = current_time
                self.no_hand_start_time = None
            
            # Reset letter tracking
            self.current_letter = None
            self.letter_start_time = None
            return result
        
        # Hand detected - reset no-hand timer
        self.no_hand_start_time = None
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.current_letter = None
            self.letter_start_time = None
            return result
        
        # Same letter as before?
        if letter == self.current_letter:
            # Calculate stability progress
            if self.letter_start_time:
                elapsed = current_time - self.letter_start_time
                result["stability_progress"] = min(1.0, elapsed / self.stability_time)
                
                # Add letter if stable long enough
                if elapsed >= self.stability_time:
                    self.sentence += letter
                    result["letter_added"] = letter
                    self.last_add_time = current_time
                    self.current_letter = None
                    self.letter_start_time = None
        else:
            # New letter - start tracking
            self.current_letter = letter
            self.letter_start_time = current_time
            result["stability_progress"] = 0.0
        
        return result
    
    def delete_last(self):
        """Delete the last character."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
    
    def add_space(self):
        """Manually add a space."""
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
    
    def clear(self):
        """Clear the entire sentence."""
        self.sentence = ""
        self.current_letter = None
        self.letter_start_time = None
    
    def get_sentence(self):
        """Get the current sentence."""
        return self.sentence


class ASLGUI:
    """Main GUI application for ASL recognition with Text Builder."""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title(config.GUI["window_title"] + " - Text Builder")
        self.window.geometry("900x750")  # Larger window for text builder
        self.window.resizable(False, False)
        
        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        
        # Video thread
        self.video_thread = None
        
        # Text Builder
        self.text_builder = TextBuilder(
            stability_time=0.8,
            cooldown_time=1.5,
            confidence_threshold=0.8
        )
        
        self.setup_ui()
        self.setup_keyboard_bindings()
        self.check_errors()
        
        # Auto-start camera if model exists
        if os.path.exists(config.MODEL_PATH) or os.path.exists(config.MODEL_PATH_KERAS):
            self.window.after(500, self.start_camera)
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        """Setup the user interface with Text Builder."""
        # Video frame with fixed size
        self.video_frame = Frame(
            self.window,
            width=config.GUI["video_width"],
            height=config.GUI["video_height"],
            bg="black"
        )
        self.video_frame.pack(pady=5)
        self.video_frame.pack_propagate(False)
        
        # Video display inside frame
        self.video_label = Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Status frame
        status_frame = Frame(self.window)
        status_frame.pack(pady=3)
        
        # Hand detection indicator
        self.hand_indicator = Label(
            status_frame, 
            text="● No Hand", 
            font=("Arial", 11),
            fg="red"
        )
        self.hand_indicator.pack(side=tk.LEFT, padx=10)
        
        # Stability indicator
        self.stability_label = Label(
            status_frame,
            text="Stability: -",
            font=("Arial", 11),
            fg="gray"
        )
        self.stability_label.pack(side=tk.LEFT, padx=10)
        
        # Prediction display
        self.prediction_label = Label(
            self.window,
            text="Prediction: -",
            font=("Arial", 24, "bold"),
            fg="green"
        )
        self.prediction_label.pack(pady=5)
        
        # Confidence bar
        self.confidence_frame = Frame(self.window)
        self.confidence_frame.pack(pady=3)
        
        Label(
            self.confidence_frame,
            text="Confidence:",
            font=("Arial", 11)
        ).pack(side=tk.LEFT)
        
        self.confidence_bar = Frame(
            self.confidence_frame,
            width=200,
            height=18,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=1
        )
        self.confidence_bar.pack(side=tk.LEFT, padx=5)
        
        self.confidence_fill = Frame(
            self.confidence_bar,
            width=0,
            height=16,
            bg="green"
        )
        self.confidence_fill.place(x=1, y=1)
        
        self.confidence_text = Label(
            self.confidence_frame,
            text="0%",
            font=("Arial", 11),
            width=5
        )
        self.confidence_text.pack(side=tk.LEFT)
        
        # ========== TEXT BUILDER SECTION ==========
        separator = Frame(self.window, height=2, bg="gray")
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Letter confirmation progress bar
        progress_frame = Frame(self.window)
        progress_frame.pack(pady=3)
        
        Label(
            progress_frame,
            text="Hold to confirm:",
            font=("Arial", 11)
        ).pack(side=tk.LEFT)
        
        self.progress_bar = Frame(
            progress_frame,
            width=300,
            height=20,
            bg="lightgray",
            relief=tk.SUNKEN,
            bd=1
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        
        self.progress_fill = Frame(
            self.progress_bar,
            width=0,
            height=18,
            bg="blue"
        )
        self.progress_fill.place(x=1, y=1)
        
        self.progress_text = Label(
            progress_frame,
            text="0%",
            font=("Arial", 11),
            width=5
        )
        self.progress_text.pack(side=tk.LEFT)
        
        # Status label for text builder
        self.builder_status = Label(
            self.window,
            text="Hold a sign steady for 0.8s to add it",
            font=("Arial", 10),
            fg="gray"
        )
        self.builder_status.pack(pady=3)
        
        # Text display area
        text_label = Label(
            self.window,
            text="Constructed Text:",
            font=("Arial", 12, "bold")
        )
        text_label.pack(pady=(5, 0))
        
        self.text_display = Label(
            self.window,
            text="_",
            font=("Courier", 20),
            fg="#333",
            bg="#f0f0f0",
            width=40,
            height=2,
            relief=tk.SUNKEN,
            anchor="w",
            padx=10
        )
        self.text_display.pack(pady=5, padx=20)
        
        # Control buttons
        button_frame = Frame(self.window)
        button_frame.pack(pady=10)
        
        self.space_btn = Button(
            button_frame,
            text="Space",
            font=("Arial", 11),
            width=8,
            command=self.on_space_click
        )
        self.space_btn.pack(side=tk.LEFT, padx=5)
        
        self.delete_btn = Button(
            button_frame,
            text="Delete",
            font=("Arial", 11),
            width=8,
            command=self.on_delete_click
        )
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = Button(
            button_frame,
            text="Clear All",
            font=("Arial", 11),
            width=8,
            command=self.on_clear_click
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label
        self.info_label = Label(
            self.window,
            text="Initializing camera...",
            font=("Arial", 9),
            fg="gray"
        )
        self.info_label.pack(pady=3)
        
        # Instructions
        instructions = Label(
            self.window,
            text="Hold sign steady to add | No hand for 1.2s = Space | Backspace = Delete",
            font=("Arial", 9),
            fg="#666"
        )
        instructions.pack(pady=2)
    
    def setup_keyboard_bindings(self):
        """Setup keyboard shortcuts."""
        self.window.bind("<BackSpace>", lambda e: self.on_delete_click())
        self.window.bind("<space>", lambda e: self.on_space_click())
        self.window.bind("<Escape>", lambda e: self.on_clear_click())
    
    def on_space_click(self):
        """Handle space button click."""
        self.text_builder.add_space()
        self.update_text_display()
    
    def on_delete_click(self):
        """Handle delete button click."""
        self.text_builder.delete_last()
        self.update_text_display()
    
    def on_clear_click(self):
        """Handle clear button click."""
        self.text_builder.clear()
        self.update_text_display()
    
    def update_text_display(self):
        """Update the text display with current sentence."""
        sentence = self.text_builder.get_sentence()
        display_text = sentence + "_" if sentence else "_"
        self.text_display.config(text=display_text)
    
    def update_progress_bar(self, progress, in_cooldown=False, cooldown_remaining=0):
        """Update the letter confirmation progress bar."""
        max_width = 298
        
        if in_cooldown:
            # Show cooldown in orange
            fill_width = int(max_width * (cooldown_remaining / 1.5))
            self.progress_fill.config(width=fill_width, bg="orange")
            self.progress_text.config(text=f"{cooldown_remaining:.1f}s")
            self.builder_status.config(text="Cooldown - wait before next letter", fg="orange")
        else:
            fill_width = int(max_width * progress)
            self.progress_fill.config(width=fill_width, bg="blue")
            self.progress_text.config(text=f"{progress:.0%}")
            
            if progress > 0:
                self.builder_status.config(text="Keep holding to confirm...", fg="blue")
            else:
                self.builder_status.config(text="Hold a sign steady for 0.8s to add it", fg="gray")
    
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
                    
                    # Update text builder
                    builder_result = self.text_builder.update(
                        letter, conf, prediction_data["hand_detected"]
                    )
                    
                    # Check if letter was added
                    if builder_result["letter_added"]:
                        self.update_text_display()
                        self.builder_status.config(
                            text=f"Added: {builder_result['letter_added']}",
                            fg="green"
                        )
                    
                    # Update progress bar
                    self.update_progress_bar(
                        builder_result["stability_progress"],
                        builder_result["in_cooldown"],
                        builder_result["cooldown_remaining"]
                    )
                else:
                    self.prediction_label.config(text="Prediction: -", fg="gray")
                    self.stability_label.config(text="Stability: -", fg="gray")
                    self.update_confidence_bar(0)
                    
                    # Update text builder for no hand (potential space)
                    builder_result = self.text_builder.update(
                        None, 0, prediction_data["hand_detected"]
                    )
                    if builder_result["letter_added"]:
                        self.update_text_display()
                    self.update_progress_bar(0)
                
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
