import requests
import json

def test_api(video_path):
    """API'yi test et"""
    print(f"\nVideo yükleniyor: {video_path}")
    
    # Video dosyasını aç
    with open(video_path, 'rb') as video_file:
        # API'ye istek gönder
        response = requests.post(
            'http://localhost:5000/analyze',
            files={'video': video_file}
        )
    
    # Yanıtı kontrol et
    if response.status_code == 200:
        results = response.json()
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
        print(f"❌ Hata: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Doğru squat videosu ile test
    print("\n=== Doğru Squat Testi ===")
    test_api("data/squat_videos/squat_correct1.mp4")
    
    # Yanlış squat videosu ile test
    print("\n=== Yanlış Squat Testi ===")
    test_api("data/squat_videos/squat_wrong_knees_in1.mp4") 