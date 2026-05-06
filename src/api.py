from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from test_model import process_video, analyze_squat

app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing'i etkinleştir

# Geçici dosyalar için klasör
UPLOAD_FOLDER = "data/temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Video yükleme ve analiz endpoint'i"""
    try:
        # Video dosyasını kontrol et
        if 'video' not in request.files:
            return jsonify({'error': 'Video dosyası bulunamadı'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'Video dosyası seçilmedi'}), 400
        
        # Geçici dosya yolu
        temp_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        
        # Video dosyasını kaydet
        video_file.save(temp_path)
        
        # Video işleme
        features = process_video(temp_path)
        
        if features is not None:
            # Squat analizi
            results = analyze_squat(features)
            
            # Geçici dosyayı sil
            os.remove(temp_path)
            
            return jsonify(results)
        else:
            return jsonify({'error': 'Video işlenemedi'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü endpoint'i"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 