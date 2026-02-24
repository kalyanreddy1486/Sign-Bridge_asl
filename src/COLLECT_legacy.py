import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DATA_DIR = "ASL_Data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    for label in CLASS_NAMES:
        os.makedirs(os.path.join(DATA_DIR, label))

cap = cv2.VideoCapture(0)

for label in CLASS_NAMES:
    print(f"Collecting data for {label}. Press 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append(landmark.x)
                    keypoints.append(landmark.y)
                    keypoints.append(landmark.z)

                keypoints = np.array(keypoints)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    file_path = os.path.join(DATA_DIR, label, f"{len(os.listdir(os.path.join(DATA_DIR, label)))}.npy")
                    np.save(file_path, keypoints)
                    print(f"Saved {file_path}")

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()