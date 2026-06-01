"""
Train LSTM model on collected ASL keypoint sequences.
Saves trained model to model/asl_lstm.h5
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

ACTIONS = np.array([
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'hello','thanks','yes','no','please'
])
SEQUENCES = 30
SEQUENCE_LEN = 30
DATA_DIR = "data"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

label_map = {action: i for i, action in enumerate(ACTIONS)}

sequences, labels = [], []
for action in ACTIONS:
    for seq in range(SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LEN):
            path = os.path.join(DATA_DIR, action, str(seq), f"{frame_num}.npy")
            if not os.path.exists(path):
                break
            window.append(np.load(path))
        if len(window) == SEQUENCE_LEN:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(ACTIONS), activation='softmax'),
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

callbacks = [
    TensorBoard(log_dir=os.path.join('logs')),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

model.save(os.path.join(MODEL_DIR, 'asl_lstm.h5'))
print(f"Model saved to {MODEL_DIR}/asl_lstm.h5")

y_hat = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_hat, axis=1)
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['categorical_accuracy'], label='train')
axes[0].plot(history.history['val_categorical_accuracy'], label='val')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
print("Training plot saved.")
