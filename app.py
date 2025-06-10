# app.py

import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

# Menonaktifkan UserWarning dari Scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk upload file
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cache untuk menyimpan model yang sudah dilatih per file
# Ini mencegah pelatihan ulang setiap kali ada prediksi baru
MODELS_CACHE = {}

# === Fungsi Pembersihan Data dari Notebook Anda ===
def clean_price(price_str):
    if isinstance(price_str, str):
        price_str = price_str.replace(" ", "").replace(",", "").replace("M", "e6").replace("Jt", "e6")
        try: return float(price_str)
        except ValueError: return np.nan
    return price_str

def clean_area(area_str):
    if isinstance(area_str, str):
        area_str = area_str.replace(" m²", "").replace(",", "")
        try: return float(area_str)
        except ValueError: return np.nan
    return area_str

def clean_price_per_m2(price_m2_str):
    if isinstance(price_m2_str, str):
        price_m2_str = price_m2_str.replace("Rp\xa0", "").replace(".", "").replace("\xa0per\xa0m²", "").replace(",", "")
        try: return float(price_m2_str)
        except ValueError: return np.nan
    return price_m2_str

# === Route untuk Halaman Utama ===
@app.route('/')
def index():
    return render_template('index.html')

# === Route untuk Upload CSV dan Melatih Model ===
@app.route('/upload', methods=['POST'])
def upload_and_train():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'File tidak ditemukan.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih.'})

    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load dan proses dataset
        df = pd.read_csv(filepath, encoding='latin1')

        df['price_numeric'] = df['price'].apply(clean_price)
        df['listing-floorarea_numeric'] = df['listing-floorarea'].apply(clean_area)
        df['listing-floorarea_2_numeric'] = df['listing-floorarea 2'].apply(clean_price_per_m2)

        relevant_cols = ["bed", "bath", "listing-floorarea_numeric", "listing-floorarea_2_numeric", "listing-location", "price_numeric"]
        df.dropna(subset=relevant_cols, inplace=True)

        le_location = LabelEncoder()
        df["location_enc"] = le_location.fit_transform(df["listing-location"])
        
        # Persiapan data untuk model
        y_reg = df["price_numeric"]
        bins = pd.qcut(df["price_numeric"], q=3, labels=["Murah", "Sedang", "Mahal"])
        y_class = bins
        
        le_class = LabelEncoder()
        y_class_encoded = le_class.fit_transform(y_class)

        features = ["bed", "bath", "listing-floorarea_numeric", "listing-floorarea_2_numeric", "location_enc"]
        X = df[features]

        # Pembagian data
        X_train_reg, _, y_train_reg, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        X_train_cls, _, y_train_cls_encoded, _ = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)

        # Oversampling dengan SMOTE untuk data klasifikasi
        smote = SMOTE(random_state=42)
        X_train_cls_smote, y_train_cls_smote_encoded = smote.fit_resample(X_train_cls, y_train_cls_encoded)

        # === Pelatihan Model ===
        # Regresi
        rf_reg = RandomForestRegressor(random_state=42)
        rf_reg.fit(X_train_reg, y_train_reg)
        
        xgb_reg = XGBRegressor(random_state=42, verbosity=0)
        xgb_reg.fit(X_train_reg, y_train_reg)

        # Klasifikasi
        rf_cls = RandomForestClassifier(random_state=42)
        rf_cls.fit(X_train_cls_smote, y_train_cls_smote_encoded) # fitting dengan data smote

        xgb_cls = XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')
        xgb_cls.fit(X_train_cls_smote, y_train_cls_smote_encoded) # fitting dengan data smote

        # Simpan semua model dan encoder ke cache
        MODELS_CACHE[filename] = {
            'rf_reg': rf_reg,
            'xgb_reg': xgb_reg,
            'rf_cls': rf_cls,
            'xgb_cls': xgb_cls,
            'le_location': le_location,
            'le_class': le_class
        }
        
        # Ambil top 50 lokasi untuk dropdown
        locations = le_location.classes_.tolist()
        top_50_locations = locations[:50]

        return jsonify({
            'success': True, 
            'filename': filename, 
            'locations': top_50_locations
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Terjadi kesalahan saat memproses file: {str(e)}'})

# === Route untuk Prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data.get('filename')

    if not filename or filename not in MODELS_CACHE:
        return jsonify({'success': False, 'error': 'Model belum dilatih. Silakan upload file CSV terlebih dahulu.'})

    try:
        # Ambil model dari cache
        cache = MODELS_CACHE[filename]
        le_location = cache['le_location']
        le_class = cache['le_class']

        # Ambil data input dari request
        mode = data['mode']
        model_choice = data['model']
        bed = int(data['bed'])
        bath = int(data['bath'])
        tanah = float(data['tanah'])
        bangunan = float(data['bangunan'])
        location_name = data['location']

        # Transformasi lokasi
        location_enc = le_location.transform([location_name])[0]
        
        # Buat array input
        input_data = np.array([[bed, bath, tanah, bangunan, location_enc]])

        prediction_result = ""
        if mode == 'regresi':
            model = cache['rf_reg'] if model_choice == 'rf' else cache['xgb_reg']
            pred = model.predict(input_data)[0]
            prediction_result = f"Perkiraan Harga: Rp {int(pred):,}"
        elif mode == 'klasifikasi':
            model = cache['rf_cls'] if model_choice == 'rf' else cache['xgb_cls']
            pred_encoded = model.predict(input_data)[0]
            pred_class = le_class.inverse_transform([pred_encoded])[0]
            prediction_result = f"Kategori Harga: {pred_class}"

        return jsonify({'success': True, 'prediction': prediction_result})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Gagal melakukan prediksi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)