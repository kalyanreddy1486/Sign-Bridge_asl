"""
Improved data collection with automatic capture mode.
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

from config import config
from utils import extract_landmarks_from_hand, compute_landmark_stability, check_data_balance


# Data collector class for ASL dataset
class DataCollector:
    """Automatic and manual data collection for ASL."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MEDIAPIPE["max_num_hands"],
            min_detection_confidence=config.MEDIAPIPE["min_detection_confidence"],
            min_tracking_confidence=config.MEDIAPIPE["min_tracking_confidence"]
        )
        
        self.cap = None
        self.previous_landmarks = None
        self.stability_start_time = None
        self.samples_collected = 0
        
        # Ensure data directories exist
        self.setup_directories()
    
    def setup_directories(self):
        """Create data directories if they don't exist."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        for label in config.CLASS_NAMES:
            os.makedirs(os.path.join(config.DATA_DIR, label), exist_ok=True)
    
    def initialize_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(config.COLLECTION["camera_index"])
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def get_sample_count(self, label):
        """Get current sample count for a class."""
        folder = os.path.join(config.DATA_DIR, label)
        if os.path.exists(folder):
            return len([f for f in os.listdir(folder) if f.endswith('.npy')])
        return 0
    
    def save_sample(self, label, landmarks):
        """Save a sample to disk."""
        folder = os.path.join(config.DATA_DIR, label)
        count = self.get_sample_count(label)
        file_path = os.path.join(folder, f"{count}.npy")
        np.save(file_path, landmarks)
        return file_path
    
    def draw_info_panel(self, frame, label, mode, auto_ready=False, stability=0):
        """Draw information panel on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text info
        y_offset = 30
        
        # Current letter
        cv2.putText(frame, f"Letter: {label}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Sample count
        count = self.get_sample_count(label)
        target = config.COLLECTION["target_samples_per_class"]
        color = (0, 255, 0) if count >= target else (0, 165, 255) if count >= target * 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Samples: {count}/{target}", (200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Mode
        y_offset += 35
        mode_text = "AUTO (Hold steady)" if mode == "auto" else "MANUAL (Press 's' to save)"
        mode_color = (0, 255, 0) if mode == "auto" and auto_ready else (255, 255, 255)
        cv2.putText(frame, f"Mode: {mode_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Stability indicator
        if mode == "auto":
            y_offset += 35
            bar_width = 200
            filled = int(bar_width * stability)
            cv2.rectangle(frame, (10, y_offset - 20), (10 + bar_width, y_offset), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, y_offset - 20), (10 + filled, y_offset), (0, 255, 0), -1)
            cv2.putText(frame, f"Stability: {stability:.0%}", (220, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        y_offset = h - 20
        cv2.putText(frame, "'a'=auto | 'm'=manual | 'n'=next | 'p'=prev | 'q'=quit",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def collect_manual(self, label):
        """Manual collection mode - press 's' to save."""
        print(f"\nManual mode for '{label}'. Press 's' to save, 'n' for next letter, 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            
            hand_detected = False
            landmarks = None
            
            if result.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = extract_landmarks_from_hand(hand_landmarks)
            
            # Draw info
            frame = self.draw_info_panel(frame, label, "manual", hand_detected)
            
            # Status indicator
            status = "✓ HAND DETECTED" if hand_detected else "✗ NO HAND"
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and landmarks is not None:
                file_path = self.save_sample(label, landmarks)
                self.samples_collected += 1
                print(f"  Saved: {file_path}")
                
                # Flash green
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(100)
                
            elif key == ord('n'):
                return "next"
            elif key == ord('p'):
                return "prev"
            elif key == ord('q'):
                return "quit"
            elif key == ord('a'):
                return "auto"
    
    def collect_auto(self, label):
        """Automatic collection mode - saves when hand is stable."""
        print(f"\nAuto mode for '{label}'. Hold hand steady to capture. 'n'=next, 'p'=prev, 'q'=quit")
        
        last_capture_time = 0
        capture_interval = config.COLLECTION["auto_capture_interval_ms"] / 1000.0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            
            hand_detected = False
            landmarks = None
            is_stable = False
            stability = 0
            
            if result.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = extract_landmarks_from_hand(hand_landmarks)
                    
                    # Check stability
                    is_stable = compute_landmark_stability(landmarks, self.previous_landmarks)
                    
                    if is_stable:
                        if self.stability_start_time is None:
                            self.stability_start_time = time.time()
                        stability = min(1.0, (time.time() - self.stability_start_time) / 0.5)
                    else:
                        self.stability_start_time = None
                        stability = 0
                    
                    self.previous_landmarks = landmarks
            else:
                self.previous_landmarks = None
                self.stability_start_time = None
            
            # Auto capture
            auto_ready = False
            current_time = time.time()
            if (hand_detected and is_stable and stability >= 1.0 and 
                current_time - last_capture_time >= capture_interval):
                
                file_path = self.save_sample(label, landmarks)
                self.samples_collected += 1
                last_capture_time = current_time
                print(f"  Auto-saved: {file_path}")
                
                # Flash green
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Reset stability
                self.stability_start_time = None
                auto_ready = True
            
            # Draw info
            frame = self.draw_info_panel(frame, label, "auto", auto_ready, stability)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                return "next"
            elif key == ord('p'):
                return "prev"
            elif key == ord('q'):
                return "quit"
            elif key == ord('m'):
                return "manual"
    
    def run(self):
        """Main collection loop."""
        print("="*60)
        print("ASL Data Collection")
        print("="*60)
        
        # Check current data status
        balance = check_data_balance()
        print(f"\nCurrent data: {balance['total']} total samples")
        print(f"Per-class range: {balance['min']} - {balance['max']}")
        
        if balance['warnings']:
            print("\nPriority classes (need more data):")
            for w in balance['warnings'][:5]:
                print(f"  - {w}")
        
        # Initialize
        self.initialize_camera()
        
        # Collection settings
        current_idx = 0
        mode = "manual"  # Start in manual mode
        
        print("\n" + "="*60)
        print("Controls:")
        print("  's' = Save sample (manual mode)")
        print("  'a' = Switch to auto mode")
        print("  'm' = Switch to manual mode")
        print("  'n' = Next letter")
        print("  'p' = Previous letter")
        print("  'q' = Quit")
        print("="*60)
        
        try:
            while 0 <= current_idx < len(config.CLASS_NAMES):
                label = config.CLASS_NAMES[current_idx]
                
                if mode == "manual":
                    result = self.collect_manual(label)
                else:
                    result = self.collect_auto(label)
                
                if result == "next":
                    current_idx += 1
                elif result == "prev":
                    current_idx = max(0, current_idx - 1)
                elif result == "quit":
                    break
                elif result == "auto":
                    mode = "auto"
                elif result == "manual":
                    mode = "manual"
                    
        finally:
            self.cleanup()
        
        # Summary
        print("\n" + "="*60)
        print("Collection Summary")
        print("="*60)
        print(f"New samples collected: {self.samples_collected}")
        
        balance = check_data_balance()
        print(f"Total dataset size: {balance['total']}")
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


def main():
    """Main entry point."""
    collector = DataCollector()
    collector.run()


if __name__ == "__main__":
    main()
