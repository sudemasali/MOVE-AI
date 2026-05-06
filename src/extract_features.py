import os
import json
import numpy as np
from tqdm import tqdm

# Dizin yolları
SKELETON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'skeleton_data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'features')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Pose indeksleri
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def calculate_angle(a, b, c):
    """Üç nokta arasındaki açıyı hesaplar"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def calculate_distance(a, b):
    """İki nokta arasındaki mesafeyi hesaplar"""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def extract_features(landmarks):
    """İskelet noktalarından özellikleri çıkarır"""
    features = {}
    
    try:
        # Diz açıları
        left_knee = [landmarks[LEFT_KNEE]['x'], landmarks[LEFT_KNEE]['y']]  # Sol diz
        left_hip = [landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']]   # Sol kalça
        left_ankle = [landmarks[LEFT_ANKLE]['x'], landmarks[LEFT_ANKLE]['y']] # Sol ayak bileği
        
        right_knee = [landmarks[RIGHT_KNEE]['x'], landmarks[RIGHT_KNEE]['y']]  # Sağ diz
        right_hip = [landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']]   # Sağ kalça
        right_ankle = [landmarks[RIGHT_ANKLE]['x'], landmarks[RIGHT_ANKLE]['y']] # Sağ ayak bileği
        
        features['left_knee_angle'] = calculate_angle(left_hip, left_knee, left_ankle)
        features['right_knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Ayaklar arası mesafe
        features['feet_distance'] = calculate_distance(left_ankle, right_ankle)
        
        # Vücudun öne eğilme açısı (kalça ve omuz noktaları kullanılarak)
        left_shoulder = [landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']]
        right_shoulder = [landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']]
        shoulder_mid = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
        hip_mid = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
        
        # Dikey çizgi ile omuz-kalça çizgisi arasındaki açı
        vertical = [hip_mid[0], hip_mid[1] - 1]  # Dikey çizgi için bir nokta
        features['body_lean_angle'] = calculate_angle(vertical, hip_mid, shoulder_mid)
        
        # Ağırlık merkezi (kalça noktasının x koordinatı)
        features['weight_center'] = hip_mid[0]
        
    except Exception as e:
        print(f"❌ Özellik çıkarma hatası: {str(e)}")
        return None
    
    return features

def process_skeleton_files():
    """Tüm iskelet dosyalarını işler ve özellikleri çıkarır"""
    print("İskelet dosyaları işleniyor...")
    
    # Video bazında özellikleri topla
    video_features = {}
    
    for filename in tqdm(os.listdir(SKELETON_DIR)):
        if not filename.endswith('.json'):
            continue
            
        # Video adını ve frame numarasını al
        video_name = '_'.join(filename.split('_')[:-1])  # Son kısmı (frame numarası) hariç al
        frame_num = int(filename.split('_')[-1].split('.')[0])
        
        # JSON dosyasını oku
        with open(os.path.join(SKELETON_DIR, filename), 'r') as f:
            data = json.load(f)
            
        # Özellikleri çıkar
        features = extract_features(data['landmarks'])
        if features is None:
            continue
        
        # Video bazında özellikleri topla
        if video_name not in video_features:
            video_features[video_name] = []
        video_features[video_name].append({
            'frame': frame_num,
            'features': features
        })
    
    # Her video için özellikleri kaydet
    for video_name, frames in video_features.items():
        if not frames:  # Eğer video için hiç frame işlenmediyse atla
            continue
            
        # Frame'leri sırala
        frames.sort(key=lambda x: x['frame'])
        
        # Özellikleri kaydet
        output_file = os.path.join(OUTPUT_DIR, f"{video_name}_features.json")
        with open(output_file, 'w') as f:
            json.dump({
                'video': video_name,
                'frames': frames
            }, f, indent=2)
    
    print(f"✅ Özellikler {OUTPUT_DIR} dizinine kaydedildi.")

if __name__ == "__main__":
    process_skeleton_files() 