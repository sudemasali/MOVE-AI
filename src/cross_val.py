import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

def main():
    # Veriyi yükle
    X = np.load('../data/processed/X.npy')
    y = np.load('../data/processed/y.npy')

    print(f"X şekli: {X.shape}, y şekli: {y.shape}")

    # Modeli oluştur
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation (k-fold)
    cv_scores = cross_val_score(model, X, y, cv=5)  # 5 katmanlı cross-validation

    print(f"Cross-validation doğruluk skorları: {cv_scores}")
    print(f"Ortalama doğruluk: {np.mean(cv_scores):.2f}")
    print(f"Doğruluk skoru standart sapması: {np.std(cv_scores):.2f}")

    # Modeli eğit
    model.fit(X, y)

    # 🔥 Modeli pickle olarak kaydet
    with open("../data/processed/squat_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model pickle olarak kaydedildi: squat_model.pkl")

if __name__ == '__main__':
    main()
