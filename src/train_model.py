import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Dizin yolları
TRAINING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Karışıklık matrisini çizer ve kaydeder"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, title, output_path):
    """Özellik önemini çizer ve kaydeder"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_models():
    """Random Forest modellerini eğitir"""
    print("Veri yükleniyor...")
    
    # Verileri yükle
    X_train = np.load(os.path.join(TRAINING_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(TRAINING_DIR, 'X_test.npy'))
    y_correct_train = np.load(os.path.join(TRAINING_DIR, 'y_correct_train.npy'))
    y_correct_test = np.load(os.path.join(TRAINING_DIR, 'y_correct_test.npy'))
    y_error_train = np.load(os.path.join(TRAINING_DIR, 'y_error_train.npy'))
    y_error_test = np.load(os.path.join(TRAINING_DIR, 'y_error_test.npy'))
    
    feature_names = ['Sol Diz Açısı', 'Sağ Diz Açısı', 'Ayaklar Arası Mesafe', 
                    'Vücut Eğilme Açısı', 'Ağırlık Merkezi']
    
    # Doğru/Yanlış sınıflandırma modeli
    print("\nDoğru/Yanlış sınıflandırma modeli eğitiliyor...")
    correct_model = RandomForestClassifier(n_estimators=100, random_state=42)
    correct_model.fit(X_train, y_correct_train)
    
    # Doğru/Yanlış model performansı
    y_correct_pred = correct_model.predict(X_test)
    print("\nDoğru/Yanlış Model Performansı:")
    print(classification_report(y_correct_test, y_correct_pred, 
                              target_names=['Yanlış', 'Doğru']))
    
    # Karışıklık matrisini çiz
    plot_confusion_matrix(y_correct_test, y_correct_pred, 
                         labels=['Yanlış', 'Doğru'],
                         title='Doğru/Yanlış Sınıflandırma Karışıklık Matrisi',
                         output_path=os.path.join(MODEL_DIR, 'correct_confusion_matrix.png'))
    
    # Özellik önemini çiz
    plot_feature_importance(correct_model, feature_names,
                          'Doğru/Yanlış Sınıflandırma - Özellik Önemi',
                          os.path.join(MODEL_DIR, 'correct_feature_importance.png'))
    
    # Hata türü sınıflandırma modeli
    print("\nHata türü sınıflandırma modeli eğitiliyor...")
    error_model = RandomForestClassifier(n_estimators=100, random_state=42)
    error_model.fit(X_train[y_correct_train == 0], y_error_train[y_correct_train == 0])
    
    # Hata türü model performansı
    X_test_errors = X_test[y_correct_test == 0]
    y_error_test_filtered = y_error_test[y_correct_test == 0]
    y_error_pred = error_model.predict(X_test_errors)
    
    print("\nHata Türü Model Performansı:")
    print(classification_report(y_error_test_filtered, y_error_pred))
    
    # Karışıklık matrisini çiz
    error_types = np.unique(y_error_train[y_correct_train == 0])
    plot_confusion_matrix(y_error_test_filtered, y_error_pred,
                         labels=error_types,
                         title='Hata Türü Sınıflandırma Karışıklık Matrisi',
                         output_path=os.path.join(MODEL_DIR, 'error_confusion_matrix.png'))
    
    # Özellik önemini çiz
    plot_feature_importance(error_model, feature_names,
                          'Hata Türü Sınıflandırma - Özellik Önemi',
                          os.path.join(MODEL_DIR, 'error_feature_importance.png'))
    
    # Modelleri kaydet
    print("\nModeller kaydediliyor...")
    joblib.dump(correct_model, os.path.join(MODEL_DIR, 'correct_model.joblib'))
    joblib.dump(error_model, os.path.join(MODEL_DIR, 'error_model.joblib'))
    
    print(f"\n✅ Modeller ve grafikler {MODEL_DIR} dizinine kaydedildi.")

if __name__ == "__main__":
    train_models() 