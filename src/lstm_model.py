import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def create_model(input_shape, num_classes):
    """LSTM modelini oluştur"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Verileri yükle
    X = np.load('../data/processed/X_sequential.npy')
    y = np.load('../data/processed/y_sequential.npy')
    
    print(f"Veri şekli: X={X.shape}, y={y.shape}")
    
    # Etiketleri sayısal forma çevir
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Etiket eşleştirmelerini kaydet
    with open('../data/processed/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Verileri böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )
    
    # Modeli oluştur
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)
    num_classes = y_categorical.shape[1]
    
    model = create_model(input_shape, num_classes)
    model.summary()
    
    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Modeli kaydet
    model.save('../data/processed/lstm_model.h5')
    print("Model kaydedildi")
    
    # Eğitim geçmişini kaydet
    with open('../data/processed/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    # Test performansını değerlendir
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest doğruluğu: {test_acc:.4f}")
    print(f"Test kaybı: {test_loss:.4f}")

if __name__ == '__main__':
    main() 