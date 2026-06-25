# ☕ Sistem Klasifikasi Tingkat Sangrai Kopi Berbasis CNN dan PCA

Repositori ini berisi proyek akhir untuk melakukan klasifikasi tingkat kematangan sangrai biji kopi (*coffee roast browning*) 
ke dalam 3 tingkatan kelas utama: **Light**, **Medium**, dan **Dark**. Proyek ini mengimplementasikan dua pendekatan komputasi, 
yaitu **Machine Learning konvensional (PCA + SVM)** dan **Deep Learning (Convolutional Neural Network - CNN)**, 
serta dilengkapi dengan dasbor implementasi berbasis website (Flask).

---

## 📊 Hasil Performa Model
Berdasarkan hasil pengujian pada dataset lokal sebanyak **628 gambar**, berikut adalah perbandingan performa kedua metode:

* **PCA + SVM (Akurasi: 84.92%)**: Sangat cepat dalam proses pelatihan,
  namun memiliki sedikit kendala dalam memisahkan gradasi warna yang tipis antara kelas *Medium* dan *Dark*.
* **CNN (Akurasi: 96.83%)**: Memberikan akurasi yang sangat tinggi dan stabil.
  Lapisan konvolusi terbukti mampu mengekstrak fitur spasial dan tekstur kerutan permukaan biji kopi secara mendalam
  tanpa mengalami *overfitting* (Akurasi Validasi: 96.04%).

---

## 🔗 Tautan Penting & Unduhan

* **Dataset Utama (Kopi Sangrai)**: `[https://universe.roboflow.com/devlong-mwkgm/coffee-roast/dataset/1]`
* **Kode Sumber Backend**: `app.py` 
* **Kode Pelatihan Model**: `train.py`

---

## 📁 Struktur Folder Proyek
Untuk menjalankan proyek ini secara lokal, pastikan struktur folder di komputer Anda tertata sebagai berikut:

```text
📁 Proyek-Klasifikasi-Kopi/
│
├── 📄 app.py                  # Server Backend Flask & Antarmuka Website
├── 📄 train.py                # Script Pelatihan Model (PCA+SVM & CNN)
├── 📄 model_kopi_cnn.h5       # File Bobot Model CNN Terlatih (Disimpan Lokal)
├── 📄 .gitignore              # Mengabaikan file besar (env/ & dataset) dari GitHub
│
└── 📁 coffee-roast-dataset/   # Folder Dataset Gambar (Unduh dari Link di Atas)
    └── 📁 train/
        ├── 📁 Light/
        ├── 📁 Medium/
        └── 📁 Dark/

⚠️ Catatan Dataset: Folder coffee-roast-dataset/ dan env/ tidak dimasukkan ke dalam repositori GitHub ini
    karena pembatasan ukuran file biner. Silakan unduh dataset secara mandiri melalui tautan di atas.

🛠️ Tutorial Langkah Demi Langkah Menjalankan Aplikasi

Instruksi ini ditulis untuk dijalankan pada lingkungan sistem operasi berbasis Linux (Ubuntu/Mint) menggunakan Python 3.12+.
1. Langkah 1: Kloning Repositori dan Masuk ke Folder
    Buka terminal Anda, lalu jalankan perintah berikut untuk mengunduh source code dari GitHub ke komputer lokal:

    git clone [https://github.com/Xian26/klasifikasi-kopi-cnn-dan-pca-dengan-tensorflow.git](https://github.com/Xian26/klasifikasi-kopi-cnn-dan-pca-dengan-tensorflow.git)
    cd klasifikasi-kopi-cnn-dan-pca-dengan-tensorflow

2. Langkah 2: Sediakan Dataset Kopi

    Unduh file dataset melalui tautan yang tertera di bagian Tautan Penting & Unduhan di atas.

    Ekstrak file zip dataset tersebut.

    Letakkan folder hasil ekstrak bernama coffee-roast-dataset langsung di dalam folder proyek Anda (sejajar dengan file app.py).

3. Langkah 3: Membuat dan Mengaktifkan Virtual Environment (venv)
    Gunakan environment terisolasi agar instalasi pustaka Python tidak mengganggu sistem operasi utama Anda:
        Bash
        # 1. Pastikan modul venv Python sistem telah terpasang
        sudo apt update && sudo apt install python3-full -y
        
        # 2. Buat environment baru dengan nama 'env'
        python3 -m venv env
        
        # 3. Aktifkan environment
        source env/bin/activate
        
        *Indikator (env) akan muncul di sebelah kiri baris terminal Anda jika berhasil.

4. Langkah 4: Instalasi Pustaka (Dependencies)

    Perbarui pengelola paket pip dan pasang seluruh pustaka komputasi serta web yang diperlukan proyek:
    Bash
      pip install --upgrade pip
      pip install numpy scikit-learn tensorflow matplotlib Flask Pillow

5. Langkah 5: Melatih Model (Optional / Jika Belum Ada File .h5)
    Jika Anda ingin melakukan pelatihan ulang dari nol atau memverifikasi metrik akurasi (PCA dan CNN) langsung pada dataset, jalankan perintah:
    Bash

      python3 train.py

    Proses ini akan mengevaluasi dataset dan otomatis mengekspor file bobot model baru bernama model_kopi_cnn.h5 setelah training selesai.

6. Langkah 6: Menjalankan Dashboard Website
    Untuk mengoperasikan aplikasi deteksi kopi interaktif lewat browser, hidupkan server backend Flask dengan perintah berikut:
    Bash

    python3 app.py

      Tunggu hingga terminal menampilkan log server aktif, lalu buka browser (Chrome/Firefox) Anda dan akses alamat lokal berikut:
      👉 http://127.0.0.1:5000


🖥️ Fitur Dashboard Website

Website ini dirancang secara rinci untuk menyajikan analisis data komputasi yang informatif, meliputi:

    1. Input Citra Praktis: Mendukung unggahan file gambar langsung lewat browser dengan fitur pratinjau instan.

    2. Softmax Probability Distribution: Menampilkan diagram batang interaktif yang memuat nilai probabilitas kepastian dari ketiga kelas (Light, Medium, Dark).

    3. Karakteristik Fisik & Kimiawi: Menyajikan data edukatif mengenai estimasi suhu sangrai, profil rasa (flavor profile),
        perubahan visual, dan estimasi kadar kafein pada biji kopi yang terdeteksi berdasarkan teori ilmiah.
