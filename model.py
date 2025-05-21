import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("gesture_model.h5")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Prediction buffer
buffer_size = 5
prediction_buffer = deque(maxlen=buffer_size)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    majority_label = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)[0][0]
            predicted_label = 1 if prediction > 0.5 else 0
            prediction_buffer.append(predicted_label)

            # Calculate majority vote
            if len(prediction_buffer) == buffer_size:
                majority_label = int(round(np.mean(prediction_buffer)))
                cv2.putText(frame, f'Stable Prediction: {majority_label}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Buffered Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
