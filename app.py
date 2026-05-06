from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from src.squat_analyzer import SquatAnalyzer
from src.skeleton_extractor import SkeletonExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Upload klasörünü oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Video dosyası bulunamadı'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Video analizi yap
            extractor = SkeletonExtractor()
            analyzer = SquatAnalyzer()
            
            # İskelet verilerini çıkar
            skeleton_data = extractor.extract_skeleton(filepath)
            
            # Squat analizi yap
            results = analyzer.analyze_squat(skeleton_data)
            
            # Geçici dosyayı sil
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'İzin verilmeyen dosya formatı'}), 400

if __name__ == '__main__':
    app.run(debug=True) 