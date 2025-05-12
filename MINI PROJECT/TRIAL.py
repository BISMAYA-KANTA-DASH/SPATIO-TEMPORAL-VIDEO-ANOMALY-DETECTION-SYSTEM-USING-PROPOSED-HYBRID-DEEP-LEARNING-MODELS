import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, ConvLSTM2D, Conv3D, MaxPooling3D, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Ensure TensorFlow uses GPU efficiently
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Video folder path
video_folder = "VIDEO_1"
if not os.path.exists(video_folder):
    raise FileNotFoundError(f"Folder '{video_folder}' does not exist!")

# Load videos and extract frames
def load_videos(video_folder, frame_skip=10, max_frames=10):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".MP4")]
    data, labels = [], []

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frame_resized = cv2.resize(frame, (224, 224)) / 255.0  # Normalize
                frames.append(frame_resized)
            frame_count += 1
        cap.release()
        
        # Select only a fixed number of frames (padding if necessary)
        if len(frames) >= max_frames:
            frames = frames[:max_frames]
        else:
            # Pad with last frame to maintain sequence length
            frames += [frames[-1]] * (max_frames - len(frames))

        data.append(frames)
        labels.append(1 if "anomaly" in video_file.lower() else 0)  # Binary labels

    return np.array(data), np.array(labels)

# Feature Extraction using ResNet50
def extract_features(data):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

    features = []
    for video in data:
        video_features = np.array([feature_extractor.predict(np.expand_dims(frame, axis=0), verbose=0) for frame in video])
        features.append(video_features.squeeze())

    features = np.array(features)  # Shape: (num_samples, 10, 2048)
    print("Feature shape after extraction:", features.shape)
    return features

# CNN+RNN Model
def build_cnn_rnn_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3D CNN Model
def build_3d_cnn_model():
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(10, 224, 224, 3)),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Autoencoder Model (ConvLSTM)
def build_autoencoder():
    model = Sequential([
        ConvLSTM2D(32, kernel_size=(3, 3), activation='relu', input_shape=(10, 224, 224, 3), return_sequences=True),
        Dropout(0.3),
        ConvLSTM2D(64, kernel_size=(3, 3), activation='relu', return_sequences=False),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Transformer-Based Model
def build_transformer_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load Data
data, labels = load_videos(video_folder)

if data.shape[0] == 0:
    raise ValueError("No video data was loaded. Check the video folder path and file format.")

# Compute class weights (for imbalanced datasets)
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
print("Computed Class Weights:", class_weights)

# Extract Features for CNN+RNN and Transformer Model
features = extract_features(data)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Split Data for 3D CNN and Autoencoder
X_train_raw, X_test_raw, _, _ = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Train & Evaluate Models
models = {
    "CNN+RNN (LSTM)": (build_cnn_rnn_model((10, 2048)), X_train, X_test),
    "3D CNN": (build_3d_cnn_model(), X_train_raw, X_test_raw),
    "Autoencoder (ConvLSTM)": (build_autoencoder(), X_train_raw, X_test_raw)
}

results = {}

for name, (model, X_train_input, X_test_input) in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_input, y_train, epochs=5, batch_size=8, validation_split=0.1, class_weight=class_weights, verbose=1)
    y_pred = (model.predict(X_test_input) > 0.5).astype(int)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

print("\n✅ Training & Evaluation Complete!")
