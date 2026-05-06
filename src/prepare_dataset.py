import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Dosya yolları
LABELS_CSV = '../data/squat_labels.csv'
SKELETON_DIR = '../data/skeleton_data'
OUTPUT_DIR = '../data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Etiketleri oku
labels_df = pd.read_csv(LABELS_CSV)

# Etiketleri sayısal forma çevir
label_to_int = {
    'doğru': 0,
    'dizler_içeri': 1,
    'öne_eğilme': 2,
    'ayaklar_bitişik': 3,
    'parmak_ucuna_basma': 4
}

X = []
y = []

print("Veriler okunuyor...")

for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    video = row['video_name']
    frame = row['frame']
    label = row['label']

    json_filename = f"{video}_{frame}.json"
    json_path = os.path.join(SKELETON_DIR, json_filename)

    if not os.path.exists(json_path):
        print(f"⚠️ {json_path} bulunamadı, atlanıyor.")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    keypoints = data.get("landmarks", [])
    if len(keypoints) != 33:
        print(f"⚠️ {json_filename} beklenen 33 nokta yerine {len(keypoints)} içeriyor, atlanıyor.")
        continue

    keypoint_vector = []
    for kp in keypoints:
        keypoint_vector.extend([kp['x'], kp['y'], kp['z'], kp['visibility']])

    X.append(keypoint_vector)
    y.append(label_to_int[label])

# Numpy dizilerine çevir ve kaydet
X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)

print("✅ Dataset oluşturuldu!")
print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
