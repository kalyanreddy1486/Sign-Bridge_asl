"""
Improved training script with evaluation metrics, early stopping, and confusion matrix.
"""

import os
import sys
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from utils import normalize_landmarks, check_data_balance, EvaluationMetrics


def load_data():
    """Load and preprocess training data."""
    data = []
    labels = []
    
    print("Loading data...")
    
    for label in config.CLASS_NAMES:
        folder = os.path.join(config.DATA_DIR, label)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    try:
                        raw = np.load(os.path.join(folder, file))
                        normalized = normalize_landmarks(raw)
                        data.append(normalized)
                        labels.append(label)
                    except Exception as e:
                        print(f"  Warning: Could not load {file}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Check if data is empty before accessing shape
    if data.size == 0:
        print("\n✗ No data found! Please collect data first.")
        return np.array([]), np.array([])
    
    print(f"\nTotal samples: {data.shape[0]}")
    print(f"Feature size: {data.shape[1]}")
    
    return data, labels


def create_model(input_shape, num_classes):
    """Create MLP model with configurable architecture."""
    mlp_config = config.TRAINING["mlp"]
    
    model = Sequential([
        Dense(mlp_config["hidden_layers"][0], activation='relu', input_shape=(input_shape,)),
        Dropout(mlp_config["dropout"][0]),
        Dense(mlp_config["hidden_layers"][1], activation='relu'),
        Dropout(mlp_config["dropout"][1]),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to: {save_path}")


# Main training function for ASL model
def train():
    """Main training function."""
    
    # Create model directory if needed (before configuring callbacks)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Check data balance first
    print("="*60)
    print("DATA BALANCE CHECK")
    print("="*60)
    balance = check_data_balance()
    print(f"Total samples: {balance['total']}")
    print(f"Samples per class: min={balance['min']}, max={balance['max']}")
    
    if balance['warnings']:
        print("\n⚠ Warnings:")
        for w in balance['warnings']:
            print(f"  - {w}")
    else:
        print("\n✓ Data is well balanced")
    
    # Load data
    data, labels = load_data()
    
    if len(data) == 0:
        print("\n✗ No data found! Please collect data first.")
        return
    
    # Label encoding
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded, config.NUM_CLASSES)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_onehot,
        test_size=config.TRAINING["mlp"]["validation_split"],
        random_state=config.TRAINING["mlp"]["random_state"],
        stratify=labels_encoded  # Maintain class distribution
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    model = create_model(data.shape[1], config.NUM_CLASSES)
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    callbacks = []
    
    if config.TRAINING["early_stopping"]["enabled"]:
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=config.TRAINING["early_stopping"]["patience"],
            restore_best_weights=config.TRAINING["early_stopping"]["restore_best_weights"],
            verbose=1
        )
        callbacks.append(early_stop)
    
    # Model checkpoint
    checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.TRAINING["mlp"]["epochs"],
        batch_size=config.TRAINING["mlp"]["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Compute metrics
    evaluator = EvaluationMetrics(config.CLASS_NAMES)
    for pred, true, conf in zip(y_pred, y_true, np.max(y_pred_probs, axis=1)):
        evaluator.add(pred, true, conf)
    
    evaluator.print_report()
    
    # Save metrics
    metrics = evaluator.get_summary()
    metrics_path = os.path.join(config.MODEL_DIR, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Plot confusion matrix
    cm = evaluator.compute_confusion_matrix()
    cm_path = os.path.join(config.MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)
    
    # Plot training history
    history_path = os.path.join(config.MODEL_DIR, "training_history.png")
    plot_training_history(history, history_path)
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Save in modern .keras format
    keras_path = config.MODEL_PATH_KERAS
    model.save(keras_path)
    print(f"Model saved to: {keras_path}")
    
    # Also save in .h5 for backward compatibility
    h5_path = config.MODEL_PATH
    model.save(h5_path)
    print(f"Model also saved to: {h5_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final validation accuracy: {metrics['accuracy']:.2%}")


if __name__ == "__main__":
    # Create model directory if needed
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Train
    train()
