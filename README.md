## HOW TO RUN 

# 1. Clone Repository
Buka terminal atau Git Bash dan clone repository ini ke komputer Anda.

jalankan: 
git clone https://github.com/NAMA_USER_ANDA/NAMA_REPO_ANDA.git
cd aplikasi_prediksi_rumah


# 2. Buat Virtual Environment
Sangat disarankan untuk membuat lingkungan virtual agar dependensi proyek tidak tercampur dengan library Python sistem Anda.

jalankan: 
python -m venv venv
Aktifkan Virtual Environment

**Windows:**
Bash
venv\Scripts\activate

**macOS / Linux:**
Bash
source venv/bin/activate
Setelah aktif, nama terminal Anda akan diawali dengan (venv).

# 3 Install Dependensi
Install semua library yang dibutuhkan yang tercantum dalam file requirements.txt.

Bash
pip install -r requirements.txt

# 4 Menjalankan Aplikasi
Setelah semua dependensi terinstall, jalankan server Flask dengan perintah berikut di terminal:

Bash
flask run 
