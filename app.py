import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODELS_CACHE = {}

def clean_price(price_str):
    if not isinstance(price_str, str):
        return pd.to_numeric(price_str, errors='coerce')
    try:
        s = str(price_str).lower().strip().replace(" ", "")
        if '-' in s: s = s.split('-')[0]
        s = s.replace(",", ".").replace("rp", "")
        if 'm' in s: return float(s.replace('m', '')) * 1_000_000_000
        if 'jt' in s: return float(s.replace('jt', '')) * 1_000_000
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def clean_area(area_str):
    if isinstance(area_str, str):
        try:
            return float(area_str.replace(" m²", "").replace(",", ""))
        except (ValueError, TypeError):
            return np.nan
    return area_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_train():
    if 'csv-file' not in request.files:
        return jsonify({'success': False, 'error': 'File tidak ditemukan di request.'})
    file = request.files['csv-file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih.'})

    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath, encoding='latin1')
        df['price_numeric'] = df['price'].apply(clean_price)
        df['luas_bangunan_numeric'] = df['listing-floorarea'].apply(clean_area)
        
        max_reasonable_price = 100 * 1_000_000_000
        df = df[df['price_numeric'] <= max_reasonable_price]

        price_bins = [0, 800_000_000, 2_000_000_000, float('inf')]
        price_labels = ["Murah", "Sedang", "Mahal"]
        df['price_category'] = pd.cut(df["price_numeric"], bins=price_bins, labels=price_labels, right=False)
        
        le_location = LabelEncoder()
        df["location_enc"] = le_location.fit_transform(df["listing-location"].astype(str))

        final_cols = ["bed", "bath", "luas_bangunan_numeric", "location_enc", "price_numeric", "price_category"]
        df.dropna(subset=final_cols, inplace=True)

        features = ["bed", "bath", "luas_bangunan_numeric", "location_enc"]
        X = df[features]
        
        y_reg = df["price_numeric"]
        
        y_class = df['price_category']
        le_class = LabelEncoder()
        y_class_encoded = le_class.fit_transform(y_class)

        X_train_reg, _, y_train_reg, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_reg.fit(X_train_reg, y_train_reg)
        
        X_train_cls, _, y_train_cls_encoded, _ = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)
        smote = SMOTE(random_state=42)
        X_train_cls_smote, y_train_cls_smote_encoded = smote.fit_resample(X_train_cls, y_train_cls_encoded)
        rf_cls = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_cls.fit(X_train_cls_smote, y_train_cls_smote_encoded)

        MODELS_CACHE[filename] = { 'rf_reg': rf_reg, 'rf_cls': rf_cls, 'le_location': le_location, 'le_class': le_class }
        
        return jsonify({'success': True, 'filename': filename, 'locations': le_location.classes_.tolist()})

    except Exception as e:
        error_message = f"Terjadi kesalahan saat memproses file: {str(e)}"
        return jsonify({'success': False, 'error': error_message})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data.get('filename')
    model_choice = data.get('model')
    mode = data.get('mode')

    if not filename or filename not in MODELS_CACHE:
        return jsonify({'success': False, 'error': 'Model belum dilatih.'})

    try:
        cache = MODELS_CACHE[filename]
        le_location = cache['le_location']
        le_class = cache['le_class']
        
        bed = int(data['bed'])
        bath = int(data['bath'])
        bangunan = float(data['bangunan'])
        location_name = data['location']
        
        location_enc = le_location.transform([location_name])[0]
        
        input_data = np.array([[bed, bath, bangunan, location_enc]])
        
        prediction_result = ""
        if mode == 'regresi':
            model = cache['rf_reg']
            pred = model.predict(input_data)[0]
            prediction_result = f"Perkiraan Harga: Rp {int(pred):,}"
        elif mode == 'klasifikasi':
            model = cache['rf_cls']
            pred_encoded = model.predict(input_data)[0]
            prediction_result = f"Prediksi Kategori: {le_class.inverse_transform([pred_encoded])[0]}"

        return jsonify({'success': True, 'prediction': prediction_result})

    except Exception as e:
        return jsonify({'success': False, 'error': f"Gagal melakukan prediksi: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)