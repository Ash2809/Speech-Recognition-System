import numpy as np
import librosa
import os
from tqdm import tqdm  
import pickle

def feature_Extraction(DATASET_PATH):
    IMG_SIZE = (40, 100)  
    SAMPLE_RATE = 16000

    def extract_features(file_path):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] < IMG_SIZE[1]:  
            pad_width = IMG_SIZE[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
        else:  
            mel_spec_db = mel_spec_db[:, :IMG_SIZE[1]]
        
        return mel_spec_db

    X, y = [], []
    label_map = {label: i for i, label in enumerate(os.listdir(DATASET_PATH))}

    for label in tqdm(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_path, file)
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)

    with open("speech_commands.pkl", "wb") as f:
        pickle.dump((X, y, label_map), f)

    print("Feature extraction complete!")
    print(f"Dataset Shape: {X.shape}, Labels Shape: {y.shape}")
