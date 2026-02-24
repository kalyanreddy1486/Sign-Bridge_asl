import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "ASL_Data")
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26
EPOCHS = 30
BATCH_SIZE = 32

# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]              # landmark 0 = wrist
    landmarks = landmarks - wrist
    return landmarks.flatten()

# =========================
# LOAD DATA
# =========================
data = []
labels = []

for label in CLASS_NAMES:
    folder = os.path.join(DATA_DIR, label)
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                raw = np.load(os.path.join(folder, file))
                normalized = normalize_landmarks(raw)
                data.append(normalized)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Total samples:", data.shape[0])
print("Feature size:", data.shape[1])

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded, NUM_CLASSES)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_onehot, test_size=0.2, random_state=42
)

# =========================
# MODEL (MLP)
# =========================
model = Sequential([
    Dense(256, activation='relu', input_shape=(data.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# SAVE MODEL
# =========================
model.save("asl_landmark_dl_model.h5")

# =========================
# PLOT ACCURACY
# =========================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
