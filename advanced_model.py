"""
advanced_model.py — Aurora Real-Time Gesture Controller (v2 — HaGRID)
=====================================================================
Uses the 8-class gesture model trained on HaGRID to control appliances.
No finger-tracking or screen zones — just gesture in front of the camera.

Controls:
  ✊  fist     → All devices OFF
  🖐  palm     → All devices ON
  👍  like     → Light ON
  👎  dislike  → Light OFF
  ☝  one      → Fan ON
  ✋  stop     → Fan OFF
  👌  ok       → Volume UP  (+10)
  ✌  peace    → Volume DOWN (-10)
"""

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from collections import deque
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ─── LABEL MAP ───────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: ('✊ ALL OFF',      (80,  80,  80)),
    1: ('🖐 ALL ON',       (0,  220, 100)),
    2: ('👍 LIGHT ON',     (0,  220, 255)),
    3: ('👎 LIGHT OFF',    (0,   80, 200)),
    4: ('☝ FAN ON',       (0,  160, 255)),
    5: ('✋ FAN OFF',      (0,   80, 160)),
    6: ('👌 VOL UP',       (50, 200,  50)),
    7: ('✌ VOL DOWN',     (50, 120,  50)),
}

# ─── APPLIANCE STATE ─────────────────────────────────────────────────────────
state = {
    'light_on': False,
    'fan_on':   False,
    'volume':   50,        # 0–100
}

def apply_command(label_id):
    if label_id == 0:
        state['light_on'] = False
        state['fan_on']   = False
    elif label_id == 1:
        state['light_on'] = True
        state['fan_on']   = True
    elif label_id == 2:
        state['light_on'] = True
    elif label_id == 3:
        state['light_on'] = False
    elif label_id == 4:
        state['fan_on'] = True
    elif label_id == 5:
        state['fan_on'] = False
    elif label_id == 6:
        state['volume'] = min(100, state['volume'] + 10)
    elif label_id == 7:
        state['volume'] = max(0,   state['volume'] - 10)

# ─── DRAW HUD ────────────────────────────────────────────────────────────────
def draw_hud(frame, gesture_label, gesture_color, confidence):
    h, w = frame.shape[:2]
    # Dark bottom bar
    cv2.rectangle(frame, (0, h - 90), (w, h), (15, 15, 15), -1)

    # Gesture name
    cv2.putText(frame, gesture_label, (12, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, gesture_color, 2)

    # Confidence bar
    bar_w = int(confidence * 300)
    cv2.rectangle(frame, (12, h - 38), (312, h - 18), (50, 50, 50), -1)
    cv2.rectangle(frame, (12, h - 38), (12 + bar_w, h - 18), gesture_color, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%", (318, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Appliance panel (top-right)
    panel_x = w - 200
    cv2.rectangle(frame, (panel_x - 10, 0), (w, 120), (20, 20, 20), -1)

    # Light
    lc = (0, 220, 100) if state['light_on'] else (0, 0, 150)
    cv2.circle(frame, (panel_x + 10, 30), 12, lc, -1)
    cv2.putText(frame, f"LIGHT {'ON' if state['light_on'] else 'OFF'}",
                (panel_x + 28, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1)

    # Fan
    fc = (0, 180, 255) if state['fan_on'] else (0, 0, 150)
    cv2.circle(frame, (panel_x + 10, 65), 12, fc, -1)
    cv2.putText(frame, f"FAN   {'ON' if state['fan_on'] else 'OFF'}",
                (panel_x + 28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fc, 1)

    # Volume bar
    vol_w = int(state['volume'] / 100 * 170)
    cv2.rectangle(frame, (panel_x, 88), (panel_x + 170, 108), (50, 50, 50), -1)
    cv2.rectangle(frame, (panel_x, 88), (panel_x + vol_w, 108), (50, 200, 50), -1)
    cv2.putText(frame, f"VOL {state['volume']}%",
                (panel_x, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

    return frame

# ─── LOAD ────────────────────────────────────────────────────────────────────
import os
MODEL_FILE = 'gesture_model_v2.h5'
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"'{MODEL_FILE}' not found. Run training.py first.")

print(f"🔃  Loading model '{MODEL_FILE}' …")
model = load_model(MODEL_FILE)
print("✅  Model loaded.")

# ─── MEDIAPIPE ───────────────────────────────────────────────────────────────
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ─── PREDICTION SMOOTHING ────────────────────────────────────────────────────
BUFFER_SIZE      = 12        # frames to average
CONFIDENCE_GATE  = 0.80      # minimum confidence to trigger a command
COOLDOWN_SEC     = 1.2       # seconds between repeated commands

pred_buffer      = deque(maxlen=BUFFER_SIZE)
last_label       = -1
last_command_t   = 0.0

cap = cv2.VideoCapture(0)
print("🚀  Aurora v2 — HaGRID Gesture Controller (ESC to quit)")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_label = "No hand"
    gesture_color = (100, 100, 100)
    confidence    = 0.0

    if results.multi_hand_landmarks:
        lm_obj = results.multi_hand_landmarks[0]

        # Draw thin skeleton
        mp_drawing.draw_landmarks(
            frame, lm_obj, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(60, 60, 60),  thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1)
        )

        # Extract 63 landmark values
        landmarks = []
        for pt in lm_obj.landmark:
            landmarks.extend([pt.x, pt.y, pt.z])
        x_input = np.array(landmarks, dtype='float32').reshape(1, -1)

        # Predict
        probs      = model.predict(x_input, verbose=0)[0]
        pred_label = int(np.argmax(probs))
        pred_conf  = float(probs[pred_label])

        pred_buffer.append(pred_label)

        # Majority vote over buffer
        from collections import Counter
        vote_label = Counter(pred_buffer).most_common(1)[0][0]
        confidence = float(probs[vote_label])

        gesture_label, gesture_color = LABEL_NAMES[vote_label]

        # Fire command if confident + cooldown passed
        now = time.time()
        if (confidence >= CONFIDENCE_GATE
                and (vote_label != last_label or now - last_command_t > COOLDOWN_SEC)):
            apply_command(vote_label)
            last_label     = vote_label
            last_command_t = now
            print(f"  → {gesture_label}  ({confidence*100:.0f}%)")

    frame = draw_hud(frame, gesture_label, gesture_color, confidence)
    cv2.imshow("Aurora v2 ✦ HaGRID Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
