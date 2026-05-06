import os
import cv2
import mediapipe as mp
import json
from tqdm import tqdm

# Girdi ve çıktı dizinleri
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'squat_videos')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'skeleton_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Video directory: {VIDEO_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Video dizininin var olup olmadığını kontrol et
if not os.path.exists(VIDEO_DIR):
    print(f"❌ Video dizini bulunamadı: {VIDEO_DIR}")
    exit(1)

# Video dosyalarını sırala
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"Found {len(video_files)} video files")
print("Video files:", video_files)

if len(video_files) == 0:
    print("❌ Video dizininde MP4 dosyası bulunamadı!")
    exit(1)

try:
    # Mediapipe başlat
    print("MediaPipe başlatılıyor...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe başlatıldı ✅")

    print("\nVideo işlemeye başlanıyor...")
    total_videos = len(video_files)
    for video_idx, video_file in enumerate(video_files, 1):
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"\n[{video_idx}/{total_videos}] İşleniyor: {video_file}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open {video_file}")
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames: {total_frames}")

            frame_idx = 0
            processed_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % 5 != 0:  # Her 5 framede bir al
                    frame_idx += 1
                    continue

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    data = {
                        "video": video_file,
                        "frame": frame_idx,
                        "label": "",  # Etiket daha sonra eklenecek
                        "landmarks": []
                    }

                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            data["landmarks"].append({
                                "x": lm.x,
                                "y": lm.y,
                                "z": lm.z,
                                "visibility": lm.visibility
                            })

                        # Dosya adı: squat_correct3.mp4_0.json gibi
                        json_filename = f"{video_file}_{frame_idx}.json"
                        json_path = os.path.join(OUTPUT_DIR, json_filename)
                        with open(json_path, 'w') as f:
                            json.dump(data, f)
                        processed_frames += 1

                        if processed_frames % 10 == 0:  # Her 10 frame'de bir ilerleme göster
                            print(f"  Frame: {frame_idx}/{total_frames}, İşlenen: {processed_frames}")

                except Exception as e:
                    print(f"❌ Frame işleme hatası (frame {frame_idx}): {str(e)}")
                    continue

                frame_idx += 1

            cap.release()
            print(f"✅ {video_file}: {processed_frames} frame işlendi")

        except Exception as e:
            print(f"❌ Video işleme hatası ({video_file}): {str(e)}")
            continue

except Exception as e:
    print(f"❌ Ana döngüde hata: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n✅ Tüm videolardan iskelet çıkarımı tamamlandı.")
