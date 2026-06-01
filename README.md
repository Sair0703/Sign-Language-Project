# Real-Time Sign Language Detection System

Real-time American Sign Language (ASL) detection using LSTM neural networks and MediaPipe keypoint extraction. Achieves 92% accuracy across 20+ ASL classes at 20–25 FPS on CPU.

## Demo

![Demo](demo.gif)

## Features

- **Real-time inference** at 20–25 FPS on CPU (no GPU required)
- **20+ ASL classes** including letters A–Z and common words
- **LSTM architecture** with MediaPipe hand/pose keypoints as features
- **Training pipeline** with data collection, preprocessing, and model training scripts

## Tech Stack

- Python, TensorFlow/Keras, OpenCV, MediaPipe
- LSTM neural network (sequence classification)
- NumPy, Scikit-Learn, Matplotlib

## Project Structure

```
sign-language-detector/
├── collect_data.py       # Webcam-based keypoint data collection
├── train_model.py        # LSTM model training
├── realtime_detect.py    # Live inference from webcam
├── model/
│   └── asl_lstm.h5       # Trained model weights
├── data/                 # Collected keypoint sequences
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data (optional — pretrained model included)
```bash
python collect_data.py
```
- Opens webcam, press the class key (A–Z) to record sequences
- Each class collects 30 sequences × 30 frames of MediaPipe keypoints

### 2. Train the Model
```bash
python train_model.py
```
- Trains LSTM on collected sequences
- Saves model to `model/asl_lstm.h5`

### 3. Run Real-Time Detection
```bash
python realtime_detect.py
```
- Opens webcam and starts live ASL detection
- Press `q` to quit

## Model Architecture

```
Input: (30 frames, 1662 keypoints)  ← MediaPipe hand + pose landmarks
LSTM(64) → LSTM(128) → LSTM(64)
Dense(64, relu) → Dense(32, relu)
Dense(num_classes, softmax)
```

- **Input**: 30-frame sequences of MediaPipe keypoints (hand + pose = 1662 features)
- **Training**: 30 sequences per class, Adam optimizer, categorical cross-entropy
- **Accuracy**: 92% on held-out test set across 20+ ASL classes

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 92% |
| Inference Speed | 20–25 FPS (CPU) |
| Classes | 20+ ASL signs |
| Input Features | 1662 keypoints/frame |
