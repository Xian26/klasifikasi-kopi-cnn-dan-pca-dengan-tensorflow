import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# 1. Muat Model CNN
MODEL_PATH = 'model_kopi_cnn.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model CNN Berhasil Dimuat!")
else:
    print("❌ Model 'model_kopi_cnn.h5' tidak ditemukan!")

CLASS_NAMES = ['Dark', 'Light', 'Medium']
IMG_SIZE = 128

# Data Karakteristik Rinci untuk Edukasi di Website
ROAST_DETAILS = {
    'Light': {
        'warna': 'Cokelat muda keemasan (seperti kayu manis)',
        'suhu': '196°C - 205°C (Dihentikan tepat saat atau sebelum First Crack)',
        'karakter': 'Memiliki keasaman (acidity) yang tinggi, body yang ringan, dan tidak ada minyak pada permukaan biji. Rasa asli buah (floral/fruity) dari biji kopi sangat mendominasi.',
        'kafein': 'Paling tinggi di antara tingkat sangrai lainnya.'
    },
    'Medium': {
        'warna': 'Cokelat sedang yang pekat',
        'suhu': '210°C - 218°C (Antara First Crack selesai dan sebelum Second Crack)',
        'karakter': 'Rasa yang sangat seimbang (balanced) antara keasaman (acidity) dan kekentalan (body). Aroma kopi lebih kuat dengan sedikit jejak rasa manis karamelisasi.',
        'kafein': 'Sedang/Seimbang.'
    },
    'Dark': {
        'warna': 'Cokelat sangat tua mendekati hitam dan berminyak',
        'suhu': '225°C - 245°C (Memasuki atau melewati Second Crack)',
        'karakter': 'Keasaman hampir sepenuhnya hilang, digantikan oleh body yang sangat tebal dan rasa pahit yang dominan (smoky, roasty, atau beraroma cokelat pahit/gosong). Permukaan biji dipenuhi minyak alami kopi.',
        'kafein': 'Paling rendah karena kafein menyublim akibat suhu tinggi.'
    }
}

# 2. Desain Antarmuka Dashboard yang Rinci
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Analisis Kematangan Kopi - CNN</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f8f6f0; color: #3e2723; margin: 0; padding: 20px; }
        .dashboard { max-width: 900px; margin: 20px auto; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media(max-width: 768px) { .dashboard { grid-template-columns: 1fr; } }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .full-width { grid-column: 1 / -1; text-align: center; }
        h1, h2, h3 { color: #5d4037; margin-top: 0; }
        .upload-box { border: 2px dashed #bcaaa4; padding: 30px; border-radius: 8px; cursor: pointer; background: #efebe9; transition: 0.3s; }
        .upload-box:hover { background: #d7ccc8; }
        #preview { max-width: 100%; max-height: 250px; border-radius: 8px; margin-top: 15px; display: none; margin-left: auto; margin-right: auto; }
        button { background-color: #5d4037; color: white; border: none; padding: 12px 20px; font-size: 16px; border-radius: 6px; cursor: pointer; width: 100%; font-weight: bold; }
        button:hover { background-color: #3e2723; }
        
        /* Progress Bar Probabilitas */
        .prob-container { margin-top: 15px; }
        .prob-row { margin-bottom: 12px; }
        .prob-label { display: flex; justify-content: space-between; font-weight: 500; font-size: 14px; }
        .bar-bg { background: #eee; border-radius: 4px; height: 12px; width: 100%; margin-top: 4px; overflow: hidden; }
        .bar-fill { background: #8d6e63; height: 100%; width: 0%; transition: width 0.6s ease-out; }
        
        /* Info Edukasi Detail */
        .info-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
        .info-table td { padding: 8px 4px; border-bottom: 1px solid #efebe9; vertical-align: top; }
        .info-table td:first-child { font-weight: bold; color: #795548; width: 30%; }
        
        #result-status { font-size: 24px; font-weight: bold; color: #2e7d32; margin-bottom: 10px; text-transform: uppercase; }
        .placeholder-text { color: #9e9e9e; font-style: italic; text-align: center; padding: 40px 0; }
    </style>
</head>
<body>

<div class="dashboard">
    <div class="card full-width">
        <h1>☕ Sistem Analisis Tingkat Sangrai Kopi</h1>
        <p>Klasifikasi berbasis Deep Learning Convolutional Neural Network (CNN) - Akurasi Pengujian: 96.83%</p>
    </div>

    <div class="card">
        <h2>📷 Input Citra Biji Kopi</h2>
        <div class="upload-box" onclick="document.getElementById('imageInput').click()">
            <p id="upload-prompt">Klik di sini untuk memilih atau menyeret foto gambar...</p>
            <input type="file" id="imageInput" accept="image/*" onchange="previewImage()" style="display:none;">
            <img id="preview" alt="Pratinjau">
        </div>
        <button onclick="analyzeImage()" style="margin-top: 15px;">Jalankan Analisis CNN</button>
    </div>

    <div class="card">
        <h2>📊 Hasil Analisis Ragam Kelas</h2>
        <div id="output-placeholder" class="placeholder-text">
            Belum ada data citra yang dianalisis. Silakan unggah foto di panel kiri.
        </div>
        
        <div id="output-content" style="display: none;">
            <div id="result-status">-</div>
            
            <h3>Softmax Probability Distribution</h3>
            <div class="prob-container" id="prob-bars">
                </div>
            
            <h3 style="margin-top:25px;">Karakteristik Fisik & Kimiawi</h3>
            <table class="info-table">
                <tr><td>Estimasi Warna</td><td id="info-warna">-</td></tr>
                <tr><td>Suhu Sangrai</td><td id="info-suhu">-</td></tr>
                <tr><td>Profil Rasa</td><td id="info-karakter">-</td></tr>
                <tr><td>Kadar Kafein</td><td id="info-kafein">-</td></tr>
            </table>
        </div>
    </div>
</div>

<script>
function previewImage() {
    const file = document.getElementById('imageInput').files[0];
    const preview = document.getElementById('preview');
    const prompt = document.getElementById('upload-prompt');
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';
        prompt.style.display = 'none';
    }
}

async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    if (fileInput.files.length === 0) {
        alert('Silakan pilih file gambar biji kopi terlebih dahulu!');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    document.getElementById('output-placeholder').style.display = 'none';
    const outputContent = document.getElementById('output-content');
    outputContent.style.display = 'block';
    document.getElementById('result-status').innerHTML = 'Memproses Matriks Citra...';
    
    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // 1. Tampilkan Hasil Kelas Utama
        document.getElementById('result-status').innerHTML = `${data.prediction} Roast (${(data.confidence * 100).toFixed(2)}%)`;
        
        // 2. Tampilkan Grafik Distribusi Batang Probabilitas
        const barsContainer = document.getElementById('prob-bars');
        barsContainer.innerHTML = ''; // Reset
        
        for (const [kelas, prob] of Object.entries(data.all_probabilities)) {
            const percentage = (prob * 100).toFixed(1);
            barsContainer.innerHTML += `
                <div class="prob-row">
                    <div class="prob-label"><span>${kelas} Roast</span><span>${percentage}%</span></div>
                    <div class="bar-bg"><div class="bar-fill" style="width: ${percentage}%"></div></div>
                </div>
            `;
        }
        
        // 3. Tampilkan Informasi Edukasi Rinci
        document.getElementById('info-warna').innerText = data.details.warna;
        document.getElementById('info-suhu').innerText = data.details.suhu;
        document.getElementById('info-karakter').innerText = data.details.karakter;
        document.getElementById('info-kafein').innerText = data.details.kafein;
        
    } catch (err) {
        document.getElementById('result-status').innerHTML = 'Koneksi API Gagal.';
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Preprocessing gambar (Wajib 128x128 dan normalisasi /255.0)
        img = Image.open(file).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Eksekusi Prediksi
        raw_predictions = model.predict(img_array)[0]
        
        # Terapkan fungsi Softmax secara manual untuk mendapatkan distribusi probabilitas murni (0-1)
        exp_preds = np.exp(raw_predictions - np.max(raw_predictions))
        probabilities = exp_preds / exp_preds.sum()
        
        max_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[max_idx]
        confidence = float(probabilities[max_idx])
        
        # Susun semua nilai probabilitas kelas ke dalam kamus data
        all_prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_prob_dict,
            'details': ROAST_DETAILS[predicted_class]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)