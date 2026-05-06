import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from extract_features import extract_features

# Dizin yolları
MODEL_DIR = "data/models"
TEMP_DIR = "data/temp"

# Hata tipleri sözlüğü
ERROR_TYPES = {
    'leaning_forward': 'Öne Fazla Eğilme',
    'leaning_forward5': 'Öne Fazla Eğilme',
    'knees_in': 'Dizler İçe Dönük',
    'knees_in1': 'Dizler İçe Dönük',
    'weight_forward': 'Ağırlık Öne Kaymış',
    'weight_forward6': 'Ağırlık Öne Kaymış',
    'wrong_depth': 'Yetersiz Derinlik',
    'wrong_depth3': 'Yetersiz Derinlik'
}

# Maksimum frame sayısı
MAX_FRAMES = 200

def process_video(video_path):
    """Video dosyasını işleyip iskelet verilerini çıkarır"""
    print(f"\nVideo işleniyor: {video_path}")
    
    # MediaPipe Pose'u başlat
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Video açılamadı!")
        return None
    
    # Geçici dizini oluştur
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Toplam frame sayısını al
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Toplam frame sayısı: {total_frames}")
    
    # Frame'leri işle
    frame_count = 0
    processed_frames = 0
    all_features = []
    
    print("\nFrame'ler işleniyor...")
    while cap.isOpened() and processed_frames < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Her 5. frame'i işle
        if frame_count % 5 != 0:
            continue
        
        # Frame'i işle
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Landmark'ları numpy dizisine dönüştür
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                })
            
            # Özellikleri çıkar
            features = extract_features(landmarks)
            if features:
                feature_vector = [
                    float(features['left_knee_angle']),
                    float(features['right_knee_angle']),
                    float(features['feet_distance']),
                    float(features['body_lean_angle']),
                    float(features['weight_center'])
                ]
                all_features.append(feature_vector)
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    print(f"✅ {processed_frames}/{min(total_frames//5, MAX_FRAMES)} frame işlendi")
    
    cap.release()
    pose.close()
    
    print(f"\nToplam {frame_count} frame'den {processed_frames} frame başarıyla işlendi")
    return np.array(all_features, dtype=np.float64)

def analyze_squat(features):
    """Çıkarılan özellikleri kullanarak squat analizi yapar"""
    print("\nSquat analizi yapılıyor...")
    
    try:
        # Modelleri yükle
        print("Modeller yükleniyor...")
        correct_model = joblib.load(os.path.join(MODEL_DIR, "correct_model.joblib"))
        error_model = joblib.load(os.path.join(MODEL_DIR, "error_model.joblib"))
        
        # Her frame için tahmin yap
        correct_predictions = correct_model.predict(features)
        error_predictions = error_model.predict(features)
        
        # Sonuçları analiz et
        correct_count = int(sum(correct_predictions))
        total_frames = int(len(correct_predictions))
        correct_percentage = float((correct_count / total_frames) * 100)
        
        # Hata tiplerini say
        error_types = {}
        for error in error_predictions:
            error_type = ERROR_TYPES.get(error, error)  # Bilinmeyen hata tipleri için orijinal etiketi kullan
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # En sık görülen hataları sırala
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        # Sonuçları hazırla
        results = {
            'total_frames': total_frames,
            'correct_frames': correct_count,
            'incorrect_frames': total_frames - correct_count,
            'correct_percentage': correct_percentage,
            'is_correct': correct_percentage >= 60,
            'errors': [],
            'suggestions': []
        }
        
        # Hataları ekle
        for error_type, count in sorted_errors:
            percentage = float((count / total_frames) * 100)
            results['errors'].append({
                'type': error_type,
                'count': int(count),
                'percentage': percentage
            })
        
        # Önerileri ekle
        if "Öne Fazla Eğilme" in error_types:
            results['suggestions'].extend([
                "Sırtınızı daha dik tutmaya çalışın",
                "Göğsünüzü yukarı kaldırın"
            ])
        if "Dizler İçe Dönük" in error_types:
            results['suggestions'].extend([
                "Dizlerinizi ayak parmak uçlarınızla aynı hizada tutun",
                "Kalçanızı daha aktif kullanın"
            ])
        if "Ağırlık Öne Kaymış" in error_types:
            results['suggestions'].extend([
                "Ağırlığınızı topuklarınıza verin",
                "Ayaklarınızı yere sıkıca basın"
            ])
        if "Yetersiz Derinlik" in error_types:
            results['suggestions'].extend([
                "Kalçanızı daha aşağı indirin",
                "Dizlerinizi 90 derece açıya getirin"
            ])
        
        return results
            
    except Exception as e:
        print(f"❌ Analiz sırasında hata oluştu: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test edilecek video
    test_video = "data/squat_videos/squat_wrong_knees_in1.mp4"
    
    # Video işleme
    features = process_video(test_video)
    
    if features is not None:
        # Squat analizi
        results = analyze_squat(features)
        print("\n=== Analiz Sonuçları ===")
        print(f"Toplam Frame Sayısı: {results['total_frames']}")
        print(f"Doğru Frame Sayısı: {results['correct_frames']}")
        print(f"Yanlış Frame Sayısı: {results['incorrect_frames']}")
        print(f"Doğruluk Yüzdesi: {results['correct_percentage']:.1f}%")
        
        if not results['is_correct']:
            print("\n❌ Squat Hatalı!")
            print("\nTespit Edilen Hatalar:")
            for error in results['errors']:
                print(f"• {error['type']}: {error['count']} frame ({error['percentage']:.1f}%)")
            
            print("\nÖneriler:")
            for suggestion in results['suggestions']:
                print(f"- {suggestion}")
        else:
            print("\n✅ Squat Doğru!")
            print("\nTebrikler! Squat hareketiniz genel olarak doğru formda.")
    else:
        print("❌ Video işlenemedi!") 