import numpy as np
import os

# .npy dosyasının bulunduğu dizini belirtin
npy_dir = 'C:/Users/USER/PycharmProjects/SquatAnalysisProject/data/processed'

# Tüm .npy dosyalarını kontrol et
def check_npy_files():
    # .npy dosyalarının bulunduğu dizindeki dosyalar
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    for npy_file in npy_files:
        try:
            # Dosyayı yükle
            npy_path = os.path.join(npy_dir, npy_file)
            data = np.load(npy_path)

            # Verinin doğru olup olmadığını kontrol et
            if data is None or data.size == 0:
                print(f"⚠️ {npy_file} dosyasında veri yok veya yanlış yükleme.")
            else:
                print(f"✅ {npy_file} dosyası doğru şekilde yüklendi.")
                print(f"Veri şekli: {data.shape}")

        except Exception as e:
            print(f"❌ {npy_file} dosyasında bir hata oluştu: {e}")

# Fonksiyonu çalıştır
if __name__ == "__main__":
    check_npy_files()
