import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Dizin yolları
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'features')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_label(video_name):
    """Video adından etiketleri çıkarır"""
    if 'correct' in video_name:
        return {
            'is_correct': 1,
            'error_type': 'none'
        }
    else:
        error_type = video_name.split('wrong_')[1].split('.')[0]
        return {
            'is_correct': 0,
            'error_type': error_type
        }

def load_and_prepare_data():
    """Özellik dosyalarını okur ve eğitim verisi hazırlar"""
    print("Özellik dosyaları okunuyor...")
    
    # Tüm verileri topla
    all_features = []
    all_labels = []
    all_video_names = []
    
    for filename in os.listdir(FEATURES_DIR):
        if not filename.endswith('_features.json'):
            continue
            
        video_name = filename.replace('_features.json', '')
        
        # JSON dosyasını oku
        with open(os.path.join(FEATURES_DIR, filename), 'r') as f:
            data = json.load(f)
        
        # Her frame için özellikleri ve etiketleri al
        for frame in data['frames']:
            features = frame['features']
            feature_vector = [
                features['left_knee_angle'],
                features['right_knee_angle'],
                features['feet_distance'],
                features['body_lean_angle'],
                features['weight_center']
            ]
            
            label = extract_label(video_name)
            
            all_features.append(feature_vector)
            all_labels.append(label)
            all_video_names.append(video_name)
    
    # NumPy dizilerine dönüştür
    X = np.array(all_features)
    video_names = np.array(all_video_names)
    
    # Etiketleri ayır
    y_correct = np.array([label['is_correct'] for label in all_labels])
    y_error_type = np.array([label['error_type'] for label in all_labels])
    
    print(f"Toplam {len(X)} frame yüklendi.")
    print(f"Özellikler şekli: {X.shape}")
    
    # Özellikleri normalize et
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim ve test setlerini ayır
    X_train, X_test, y_correct_train, y_correct_test, y_error_train, y_error_test, video_train, video_test = train_test_split(
        X_scaled, y_correct, y_error_type, video_names, test_size=0.2, random_state=42, stratify=y_correct
    )
    
    # Verileri kaydet
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_correct_train.npy'), y_correct_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_correct_test.npy'), y_correct_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_error_train.npy'), y_error_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_error_test.npy'), y_error_test)
    np.save(os.path.join(OUTPUT_DIR, 'video_train.npy'), video_train)
    np.save(os.path.join(OUTPUT_DIR, 'video_test.npy'), video_test)
    
    # Scaler'ı kaydet
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    
    print("\nVeri seti istatistikleri:")
    print(f"Eğitim seti boyutu: {len(X_train)}")
    print(f"Test seti boyutu: {len(X_test)}")
    
    # Etiket dağılımlarını göster
    print("\nEğitim seti etiket dağılımı:")
    print("Doğru squatlar:", sum(y_correct_train == 1))
    print("Yanlış squatlar:", sum(y_correct_train == 0))
    
    print("\nTest seti etiket dağılımı:")
    print("Doğru squatlar:", sum(y_correct_test == 1))
    print("Yanlış squatlar:", sum(y_correct_test == 0))
    
    # Hata türlerinin dağılımını göster
    print("\nHata türleri dağılımı:")
    error_types, counts = np.unique(y_error_type, return_counts=True)
    for error_type, count in zip(error_types, counts):
        print(f"{error_type}: {count}")
    
    print(f"\n✅ Veriler {OUTPUT_DIR} dizinine kaydedildi.")

if __name__ == "__main__":
    load_and_prepare_data() 