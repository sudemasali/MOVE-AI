import os
import cv2
import pandas as pd

# Klasör ayarları
VIDEO_DIR = '../data/squat_videos'
LABELS_CSV = '../data/squat_labels.csv'

# Etiketleri tutacak liste
labels = []

# Etiket tuşları (Türkçe klavye için ASCII değerleri)
label_map = {
    100: 'doğru',    # 'd' tuşu (Türkçe klavye)
    119: 'dizler_içeri',    # 'w' tuşu (Türkçe klavye)
    101: 'öne_eğilme',      # 'e' tuşu (Türkçe klavye)
    102: 'ayaklar_bitişik',  # 'f' tuşu (Türkçe klavye)
    113: 'çık',           # 'q' tuşu (Çıkış tuşu)
    112: 'parmak_ucuna_basma',  # 'p' tuşu (Yeni etiket: Ayak parmak ucuna basma)
}

frame_skip = 5  # Kaç frame'de bir etiketleme yapılacak (hız için)

print("""
Etiketleme başlıyor...
[D] Doğru
[W] Dizler İçeri
[E] Öne Eğilme
[F] Ayaklar Bitişik
[P] Ayak Parmak Ucuna Basma (Yeni Etiket)
[SPACE] Frame atla (etiketleme yapmadan geç)
[BACKSPACE] Son etiketi geri al
[Q] Çık ve kaydet
""")

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Video açılamadı: {video_file}")
        continue

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_display = frame.copy()

            # Üstüne bilgi yazalım
            info_text = f'Video: {video_file} | Frame: {frame_idx}'
            cv2.putText(frame_display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Squat Etiketleme', frame_display)

            key = cv2.waitKey(0)

            # Tuş basıldığında hangi değerin alındığını terminalde yazdıralım
            print(f"Basılan tuş: {key}")

            if key == ord('Q') or key == ord('q'):
                print("Çıkış yapılıyor...")
                cap.release()
                cv2.destroyAllWindows()
                # Verileri kaydedelim
                df = pd.DataFrame(labels, columns=['video_name', 'frame', 'label'])
                df.to_csv(LABELS_CSV, index=False)
                print(f"Etiketler {LABELS_CSV} dosyasına kaydedildi.")
                exit()

            elif key in label_map:
                label = label_map[key]
                labels.append((video_file, frame_idx, label))
                print(f"Etiketlendi: {video_file} Frame {frame_idx} -> {label}")

            elif key == 32:  # SPACE tuşu
                print(f"Atlandı: {video_file} Frame {frame_idx}")

            elif key == 8:  # BACKSPACE tuşu
                if labels:
                    removed = labels.pop()
                    print(f"Son etiket silindi: {removed}")
                else:
                    print("Silinecek bir etiket yok.")

            else:
                print("Geçersiz tuş! (D, W, E, F, P, SPACE, BACKSPACE, Q kullan)")

        frame_idx += 1

    # Her video bittiğinde veriyi kaydedelim
    print(f"{video_file} etiketleme tamamlandı, veriler kaydediliyor...")
    df = pd.DataFrame(labels, columns=['video_name', 'frame', 'label'])
    df.to_csv(LABELS_CSV, index=False)

    cap.release()

cv2.destroyAllWindows()

# Son kaydetme işlemi
df = pd.DataFrame(labels, columns=['video_name', 'frame', 'label'])
df.to_csv(LABELS_CSV, index=False)
print(f"Etiketler {LABELS_CSV} dosyasına kaydedildi.")
