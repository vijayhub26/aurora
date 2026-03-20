"""
process_hagrid.py — HaGRID Dataset Landmark Extractor
======================================================
Processes HaGRID gesture images through MediaPipe to produce
a landmark CSV ready for training.py.

Output CSV columns: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20, label]
  = 63 landmark values + 1 integer label = 64 columns total.

Run: python process_hagrid.py
"""

import cv2
import csv
import os
import sys
from tqdm import tqdm

# ─── MediaPipe Setup ─────────────────────────────────────────────────────────
from mediapipe.python.solutions import hands as mp_hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Path to the hagrid_30k folder that contains the gesture sub-folders
DATASET_ROOT = os.path.join('dataset', 'hagrid-sample-30k-384p', 'hagrid_30k')
OUTPUT_CSV   = 'hagrid_hand_data.csv'

# Gestures we want to use and their integer labels
# Only these 8 folders will be processed — others are ignored
GESTURE_MAP = {
    'train_val_fist':     0,   # ✊  → All OFF
    'train_val_palm':     1,   # 🖐  → All ON
    'train_val_like':     2,   # 👍  → Light ON
    'train_val_dislike':  3,   # 👎  → Light OFF
    'train_val_one':      4,   # ☝  → Fan ON
    'train_val_stop':     5,   # ✋  → Fan OFF
    'train_val_ok':       6,   # 👌  → Volume UP
    'train_val_peace':    7,   # ✌  → Volume DOWN
}

# ─── PROCESS ─────────────────────────────────────────────────────────────────
data     = []
skipped  = 0
detected = 0

print("=" * 60)
print("  HaGRID Landmark Extractor for Aurora")
print("=" * 60)
print(f"  Dataset : {DATASET_ROOT}")
print(f"  Gestures: {len(GESTURE_MAP)} classes")
print(f"  Output  : {OUTPUT_CSV}")
print("=" * 60)

for folder_name, label in GESTURE_MAP.items():
    folder_path = os.path.join(DATASET_ROOT, folder_name)

    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}  (skipping)")
        continue

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\n📂  [{label}] {folder_name}  ({len(files)} images)")

    for filename in tqdm(files, ncols=70, unit='img'):
        img_path = os.path.join(folder_path, filename)
        img      = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        # MediaPipe expects RGB
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            row = []
            for point in lm.landmark:
                row.extend([point.x, point.y, point.z])
            row.append(label)
            data.append(row)
            detected += 1
        else:
            skipped += 1

# ─── SAVE ────────────────────────────────────────────────────────────────────
print(f"\n💾  Writing {len(data)} rows to '{OUTPUT_CSV}' …")
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header (optional but helpful)
    header = [f"{ax}{i}" for i in range(21) for ax in ('x','y','z')] + ['label']
    writer.writerow(header)
    writer.writerows(data)

print("\n" + "=" * 60)
print(f"  ✅  Done!")
print(f"  Landmarks saved : {detected}")
print(f"  Images skipped  : {skipped}  (no hand detected)")
print(f"  Output file     : {OUTPUT_CSV}")
print("=" * 60)
print("\n  ▶  Next step: run  python training.py")
