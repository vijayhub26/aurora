"""
training.py — Aurora Multi-Class Gesture Classifier
====================================================
Trains an 8-class MLP on MediaPipe landmarks extracted from HaGRID.
Run AFTER process_hagrid.py has generated hagrid_hand_data.csv.

Output: gesture_model_v2.h5
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ─── LABEL MAP (must match process_hagrid.py) ────────────────────────────────
LABEL_NAMES = {
    0: 'fist (All OFF)',
    1: 'palm (All ON)',
    2: 'like (Light ON)',
    3: 'dislike (Light OFF)',
    4: 'one (Fan ON)',
    5: 'stop (Fan OFF)',
    6: 'ok (Volume UP)',
    7: 'peace (Volume DOWN)',
}
N_CLASSES = len(LABEL_NAMES)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
CSV_FILE = 'hagrid_hand_data.csv'
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"'{CSV_FILE}' not found. Run process_hagrid.py first.")

print(f"📂  Loading '{CSV_FILE}' …")
data = pd.read_csv(CSV_FILE)
X    = data.iloc[:, :-1].values.astype('float32')   # 63 landmarks
y    = data.iloc[:,  -1].values.astype('int32')      # integer label 0-7

print(f"    Total samples : {len(X)}")
for lbl, name in LABEL_NAMES.items():
    count = (y == lbl).sum()
    print(f"    [{lbl}] {name:<25}  {count} samples")

# ─── SPLIT ───────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─── MODEL ───────────────────────────────────────────────────────────────────
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),

    Dense(N_CLASSES, activation='softmax'),   # 8-class output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',   # labels are integers
    metrics=['accuracy']
)
model.summary()

# ─── CALLBACKS ───────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',  factor=0.5, patience=4, min_lr=1e-5),
]

# ─── TRAIN ───────────────────────────────────────────────────────────────────
print("\n🚀  Training …")
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊  Test Loss    : {loss:.4f}")
print(f"📊  Test Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

y_pred = model.predict(X_test).argmax(axis=1)
print("\n📋  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[LABEL_NAMES[i] for i in range(N_CLASSES)]))

# ─── SAVE ────────────────────────────────────────────────────────────────────
MODEL_OUT = 'gesture_model_v2.h5'
model.save(MODEL_OUT)
print(f"✅  Model saved as '{MODEL_OUT}'")
print("    ▶  Next step: python advanced_model.py")
