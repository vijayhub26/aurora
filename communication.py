import cv2
import mediapipe as mp
import numpy as np
import serial
import time
from collections import deque
from tensorflow.keras.models import load_model

# Load trained model 
model = load_model("gesture_model.h5")

# Set up serial connection to ESP32
esp = serial.Serial('COM11', 9600)  # 🔧 Replace 'COM3' with your actual port
time.sleep(2)  # wait for ESP32 to initialize

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Buffer for stable output
buffer_size = 5
prediction_buffer = deque(maxlen=buffer_size)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Model prediction
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)[0][0]
            predicted_label = 1 if prediction > 0.5 else 0
            prediction_buffer.append(predicted_label)

            # Use majority vote to smooth noise
            if len(prediction_buffer) == buffer_size:
                majority_label = int(round(np.mean(prediction_buffer)))
                cv2.putText(frame, f'Stable Prediction: {majority_label}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 🔥 Send to ESP32
                esp.write(str(majority_label).encode())
                print(f"Sent to ESP32: {majority_label}")

    cv2.imshow("Gesture Prediction + ESP32 Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
esp.close()
cv2.destroyAllWindows()
