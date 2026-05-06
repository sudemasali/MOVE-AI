# src/model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def main():
    # Veriyi yükle
    X = np.load('../data/processed/X.npy')
    y = np.load('../data/processed/y.npy')

    print(f"X şekli: {X.shape}, y şekli: {y.shape}")

    # Train-test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modeli oluştur
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Başarıyı ölç
    acc = accuracy_score(y_test, y_pred)
    print(f"Doğruluk: {acc:.2f}")
    print("\nSınıflandırma Raporu:\n")
    print(classification_report(y_test, y_pred))

    # 🔥 Modeli pickle olarak kaydet
    with open("../data/processed/squat_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model pickle olarak kaydedildi: squat_model.pkl")

if __name__ == '__main__':
    main()
