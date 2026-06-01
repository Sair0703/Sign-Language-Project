"""
Real-time ASL detection from webcam using trained LSTM model.
Runs at 20-25 FPS on CPU. Press 'q' to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

ACTIONS = np.array([
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'hello','thanks','yes','no','please'
])
SEQUENCE_LEN = 30
THRESHOLD = 0.85        # minimum confidence to display prediction
COOLDOWN_FRAMES = 15    # frames between repeated same-class predictions

model = load_model('model/asl_lstm.h5')
print("Model loaded.")

sequence = deque(maxlen=SEQUENCE_LEN)
sentence = []
last_prediction = None
cooldown = 0
fps_times = deque(maxlen=30)


def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


def prob_viz(res, actions, image, colors):
    output_frame = image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 20), (int(prob * 100), 75 + num * 20), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 72 + num * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return output_frame


COLORS = [(245,117,16) if i % 2 == 0 else (117,245,16) for i in range(len(ACTIONS))]

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LEN:
            res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
            predicted = ACTIONS[np.argmax(res)]
            confidence = res[np.argmax(res)]

            if confidence > THRESHOLD:
                if cooldown == 0 or predicted != last_prediction:
                    if len(sentence) == 0 or predicted != sentence[-1]:
                        sentence.append(predicted)
                    last_prediction = predicted
                    cooldown = COOLDOWN_FRAMES

            if cooldown > 0:
                cooldown -= 1

            # Keep sentence to last 5 words
            sentence = sentence[-5:]
            image = prob_viz(res, ACTIONS, image, COLORS)

        # FPS counter
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        # Overlay
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"FPS: {fps:.1f}", (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('ASL Real-Time Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
