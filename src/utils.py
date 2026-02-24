"""
Utility functions for Sign Language Recognition System.
Includes prediction smoothing, evaluation metrics, and landmark processing.
"""

import numpy as np
from collections import deque
from collections import Counter
import sys
import os

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class PredictionSmoother:
    """
    Smooths predictions using majority voting over a buffer of recent frames.
    Reduces flickering between similar classes.
    """
    
    def __init__(self, buffer_size=None):
        self.buffer_size = buffer_size or config.INFERENCE["smoothing_buffer_size"]
        self.buffer = deque(maxlen=self.buffer_size)
        self.confidence_buffer = deque(maxlen=self.buffer_size)
    
    def add_prediction(self, class_id, confidence):
        """Add a new prediction to the buffer."""
        self.buffer.append(class_id)
        self.confidence_buffer.append(confidence)
    
    def get_smoothed_prediction(self):
        """
        Get the most frequent class in the buffer.
        Returns: (class_id, confidence, stability_score)
        """
        if len(self.buffer) == 0:
            return None, 0.0, 0.0
        
        # Majority voting
        class_counts = Counter(self.buffer)
        most_common = class_counts.most_common(1)[0]
        class_id = most_common[0]
        frequency = most_common[1] / len(self.buffer)
        
        # Average confidence for this class
        confidences = [conf for cid, conf in zip(self.buffer, self.confidence_buffer) 
                      if cid == class_id]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return class_id, avg_confidence, frequency
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.confidence_buffer.clear()


def normalize_landmarks(landmarks):
    """
    Normalize landmarks by subtracting wrist position.
    Makes the model invariant to hand position in frame.
    """
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]  # landmark 0 = wrist
    landmarks = landmarks - wrist
    return landmarks.flatten()


def compute_landmark_stability(current_landmarks, previous_landmarks, threshold=None):
    """
    Compute if hand is stable (not moving too much).
    Returns True if stable, False otherwise.
    """
    if previous_landmarks is None:
        return False
    
    threshold = threshold or config.COLLECTION["stability_threshold"]
    
    current = np.array(current_landmarks).reshape(-1, 3)
    previous = np.array(previous_landmarks).reshape(-1, 3)
    
    # Compute movement as max displacement of any landmark
    movement = np.max(np.abs(current - previous))
    
    return movement < threshold


def extract_landmarks_from_hand(hand_landmarks):
    """Extract raw landmarks from MediaPipe hand landmarks."""
    raw_landmarks = []
    for lm in hand_landmarks.landmark:
        raw_landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(raw_landmarks)


class EvaluationMetrics:
    """
    Computes and stores evaluation metrics for model performance.
    """
    
    def __init__(self, class_names=None):
        self.class_names = class_names or config.CLASS_NAMES
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
    
    def add(self, predicted_class, true_class, confidence=1.0):
        """Add a prediction for evaluation."""
        self.predictions.append(predicted_class)
        self.ground_truth.append(true_class)
        self.confidences.append(confidence)
    
    def compute_confusion_matrix(self):
        """Compute confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.ground_truth, self.predictions, 
                               labels=range(len(self.class_names)))
    
    def compute_classification_report(self):
        """Compute detailed classification report."""
        from sklearn.metrics import classification_report
        return classification_report(
            self.ground_truth, 
            self.predictions,
            target_names=self.class_names,
            output_dict=True
        )
    
    def compute_per_class_accuracy(self):
        """Compute accuracy for each class."""
        cm = self.compute_confusion_matrix()
        row_sums = cm.sum(axis=1)
        per_class_acc = []
        for i in range(len(self.class_names)):
            if row_sums[i] == 0:
                per_class_acc.append(0.0)  # No samples for this class
            else:
                per_class_acc.append(cm[i, i] / row_sums[i])
        return {name: float(acc) for name, acc in zip(self.class_names, per_class_acc)}
    
    def get_summary(self):
        """Get a summary of all metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if len(self.predictions) == 0:
            return {"error": "No predictions made"}
        
        return {
            "accuracy": accuracy_score(self.ground_truth, self.predictions),
            "precision_macro": precision_score(self.ground_truth, self.predictions, 
                                               average='macro', zero_division=0),
            "recall_macro": recall_score(self.ground_truth, self.predictions, 
                                         average='macro', zero_division=0),
            "f1_macro": f1_score(self.ground_truth, self.predictions, 
                                average='macro', zero_division=0),
            "per_class_accuracy": self.compute_per_class_accuracy(),
            "total_samples": len(self.predictions)
        }
    
    def print_report(self):
        """Print a formatted evaluation report."""
        from sklearn.metrics import classification_report
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        if len(self.predictions) == 0:
            print("No predictions to evaluate.")
            return
        
        print(classification_report(
            self.ground_truth, 
            self.predictions,
            target_names=self.class_names,
            zero_division=0
        ))
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        print("-" * 30)
        per_class = self.compute_per_class_accuracy()
        for name, acc in sorted(per_class.items(), key=lambda x: x[1]):
            status = "✓" if acc >= 0.8 else "⚠" if acc >= 0.6 else "✗"
            print(f"{status} {name}: {acc:.2%}")
        
        print("="*60)


def check_data_balance(data_dir=None):
    """
    Check the distribution of samples across classes.
    Returns a dictionary with counts and warnings for imbalanced classes.
    """
    import os
    
    data_dir = data_dir or config.DATA_DIR
    class_names = config.CLASS_NAMES
    
    counts = {}
    warnings = []
    
    for label in class_names:
        folder = os.path.join(data_dir, label)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.npy')])
            counts[label] = count
        else:
            counts[label] = 0
            warnings.append(f"Missing folder for class '{label}'")
    
    # Check for imbalance
    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0
    
    for label, count in counts.items():
        if count < config.COLLECTION["min_samples_per_class"]:
            warnings.append(f"Class '{label}' has only {count} samples (min: {config.COLLECTION['min_samples_per_class']})")
        if max_count > 0 and count < max_count * 0.5:
            warnings.append(f"Class '{label}' is severely underrepresented ({count} vs {max_count})")
    
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "min": min_count,
        "max": max_count,
        "warnings": warnings,
        "is_balanced": len(warnings) == 0
    }


if __name__ == "__main__":
    # Test the utilities
    print("Testing PredictionSmoother...")
    smoother = PredictionSmoother(buffer_size=5)
    
    # Simulate predictions
    for i in range(10):
        smoother.add_prediction(i % 3, 0.8 + (i % 3) * 0.05)
    
    pred, conf, stability = smoother.get_smoothed_prediction()
    print(f"Smoothed prediction: class={pred}, confidence={conf:.2f}, stability={stability:.2f}")
    
    print("\nTesting data balance check...")
    balance = check_data_balance()
    print(f"Total samples: {balance['total']}")
    print(f"Min: {balance['min']}, Max: {balance['max']}")
    if balance['warnings']:
        print("Warnings:")
        for w in balance['warnings'][:5]:
            print(f"  - {w}")
