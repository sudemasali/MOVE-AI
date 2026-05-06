import numpy as np
import pickle
import cv2
import mediapipe as mp

# Modeli yükleyelim
with open('../data/processed/squat_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mediapipe başlat (isteğe bağlı, gerçek zamanlı video için)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Gerçek zamanlı video (kamera görüntüsü) açalım
cap = cv2.VideoCapture(0)  # 0, bilgisayarın varsayılan kamerasıdır

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # İskelet tespiti yapalım
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Video üzerine iskelet çizelim
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # İskelet noktalarını alalım
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        keypoints = np.array(keypoints).reshape(1, -1)

        # Modeli kullanarak tahmin yapalım
        prediction = model.predict(keypoints)
        label = prediction[0]

        # Etiket metnini gösterelim
        label_map = {0: 'doğru', 1: 'dizler_içeri', 2: 'öne_eğilme', 3: 'ayaklar_bitişik', 4: 'parmak_ucuna_basma'}
        cv2.putText(frame, f'Tahmin: {label_map[label]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonuçları ekranda gösterelim
    cv2.imshow('Squat Testi', frame)

    # Çıkmak için 'Q' tuşuna basılabilir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
