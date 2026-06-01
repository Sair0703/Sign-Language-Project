"""
Collect ASL keypoint sequences for training using webcam + MediaPipe.
Press a class key (A-Z or custom) to start recording that class.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

ACTIONS = np.array([
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'hello','thanks','yes','no','please'
])
SEQUENCES = 30      # sequences per action
SEQUENCE_LEN = 30   # frames per sequence
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
for action in ACTIONS:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_DIR, action, str(seq)), exist_ok=True)


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


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        for seq in range(SEQUENCES):
            for frame_num in range(SEQUENCE_LEN):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, f'STARTING: {action} seq {seq}', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    cv2.imshow('Feed', image)
                    cv2.waitKey(1500)
                else:
                    cv2.putText(image, f'Collecting: {action} ({seq}/{SEQUENCES})', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow('Feed', image)

                keypoints = extract_keypoints(results)
                path = os.path.join(DATA_DIR, action, str(seq), str(frame_num))
                np.save(path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
