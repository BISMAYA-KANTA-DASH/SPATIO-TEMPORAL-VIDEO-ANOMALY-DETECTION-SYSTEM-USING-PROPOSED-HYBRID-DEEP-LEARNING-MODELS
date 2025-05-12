import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract frames from videos
def extract_frames(video_folder, output_folder, frame_interval=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_data = []
    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame{frame_count}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_data.append(frame_path)
            frame_count += 1
            success, frame = cap.read()
        cap.release()
    return frame_data

# Custom Dataset for Frame Sequences
class FrameDataset(Dataset):
    def __init__(self, frame_paths, seq_length=5):
        self.frame_paths = frame_paths
        self.seq_length = seq_length
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.frame_paths) - self.seq_length + 1  
    
    def __getitem__(self, idx):
        frames = []
        for i in range(self.seq_length):
            frame = cv2.imread(self.frame_paths[idx + i])
            frame = self.transform(frame)
            frames.append(frame)
        
        frames = torch.stack(frames)  # Shape: (seq_length, C, H, W)
        return frames

# CNN-RNN Model to Convert Frames to Numeric Representation
class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 128)  # Feature vector output
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Convert frames to numeric values and store in a file
def extract_features_and_save(model, dataloader, output_file):
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            feature_vector = model(images)
            features.append(feature_vector.cpu().numpy())
    
    features = np.vstack(features)
    with open(output_file, 'wb') as f:
        pickle.dump(features, f)

# Load data and split
video_folder = "VIDEO_1"
output_folder = "frames"
frame_paths = extract_frames(video_folder, output_folder)
train_paths, test_paths = train_test_split(frame_paths, test_size=0.2, random_state=42)

train_dataset = FrameDataset(train_paths, seq_length=5)
test_dataset = FrameDataset(test_paths, seq_length=5)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Extract and save numeric representations
cnn_rnn_model = CNNRNN().to(device)
extract_features_and_save(cnn_rnn_model, train_loader, "train_features.pkl")
extract_features_and_save(cnn_rnn_model, test_loader, "test_features.pkl")

# Placeholder for Training and Evaluation of Autoencoders, 3D-CNN, Transformers, and ANN
# Define models and train them on extracted numeric features...

# Calculate accuracy, precision, recall, and F1 score
# (This part will be implemented once models are trained)

print("Feature extraction complete. Models can now be trained on the extracted features.")
