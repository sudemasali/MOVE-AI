import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Dosya yolları
SKELETON_DIR = '../data/skeleton_data'
OUTPUT_DIR = '../data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Her dizi için frame sayısı
SEQUENCE_LENGTH = 15  # Her squat hareketi için 15 frame'lik dizi

def get_video_label(video_name):
    """Video adından etiketi çıkar"""
    if 'correct' in video_name:
        return 'doğru'
    elif 'knees_in' in video_name:
        return 'dizler_içeri'
    elif 'leaning_forward' in video_name:
        return 'öne_eğilme'
    elif 'feet_too_close' in video_name:
        return 'ayaklar_bitişik'
    elif 'weight_forward' in video_name:
        return 'parmak_ucuna_basma'
    return None

def load_skeleton_data(json_path):
    """JSON dosyasından iskelet verilerini yükle"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_sequences(skeleton_files, sequence_length):
    """İskelet verilerinden sıralı diziler oluştur"""
    sequences = []
    labels = []
    
    # Video dosyalarını grupla
    video_groups = {}
    for file in skeleton_files:
        video_name = file.split('_')[0] + '_' + file.split('_')[1]  # squat_correct1 gibi
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(file)
    
    # Her video için sıralı diziler oluştur
    for video_name, files in tqdm(video_groups.items(), desc="Videolar işleniyor"):
        # Dosyaları frame numarasına göre sırala
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Video etiketini al
        label = get_video_label(video_name)
        if not label:
            continue
            
        # Sıralı diziler oluştur
        for i in range(0, len(files) - sequence_length + 1, sequence_length // 2):  # %50 örtüşme
            sequence = []
            for j in range(sequence_length):
                file_path = os.path.join(SKELETON_DIR, files[i + j])
                data = load_skeleton_data(file_path)
                
                # İskelet noktalarını düzleştir
                landmarks = []
                for lm in data['landmarks']:
                    landmarks.extend([lm['x'], lm['y'], lm['z'], lm['visibility']])
                sequence.append(landmarks)
            
            sequences.append(sequence)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

def main():
    # Tüm JSON dosyalarını bul
    skeleton_files = [f for f in os.listdir(SKELETON_DIR) if f.endswith('.json')]
    print(f"Toplam {len(skeleton_files)} iskelet dosyası bulundu")
    
    # Sıralı diziler oluştur
    X, y = create_sequences(skeleton_files, SEQUENCE_LENGTH)
    
    print(f"Oluşturulan dizi sayısı: {len(X)}")
    print(f"Dizi şekli: {X.shape}")
    print(f"Etiket sayısı: {len(y)}")
    
    # Verileri kaydet
    np.save(os.path.join(OUTPUT_DIR, 'X_sequential.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_sequential.npy'), y)
    print("Veriler kaydedildi")

if __name__ == '__main__':
    main() 