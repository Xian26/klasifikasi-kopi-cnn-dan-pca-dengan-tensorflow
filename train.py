import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================================
# LANGKAH 1: MENENTUKAN PATH & MEMUAT DATASET DARI FOLDER EKSTRAK
# =====================================================================
# Ubah sesuaikan dengan nama folder hasil ekstrak manualmu
PATH_DATASET = 'coffee-roast-dataset' 
IMG_SIZE = 128

def load_data_from_directory(base_dir):
    print("====== LANGKAH 1: MEMUAT DATASET LOKAL BROWNING KOPI ======")
    # Membaca data gambar dari sub-folder 'train'
    data_train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, 'train'),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32
    )
    
    class_names = data_train.class_names
    
    X, y = [], []
    for images, labels in data_train:
        X.append(images.numpy())
        y.append(labels.numpy())
        
    X = np.concatenate(X, axis=0) / 255.0  # Normalisasi piksel ke rentang [0, 1]
    y = np.concatenate(y, axis=0)
    return X, y, class_names

# Eksekusi pemuatan data
X_data, y_data, classes = load_data_from_directory(PATH_DATASET)
num_classes = len(classes)

# Membagi data menjadi 80% untuk Training dan 20% untuk Testing
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"✓ Berhasil memuat {X_data.shape[0]} total gambar.")
print(f"✓ Terdeteksi {num_classes} kelas tingkatan sangrai: {classes}\n")


# =====================================================================
# LANGKAH 2: KLASIFIKASI MENGGUNAKAN PCA + SVM (MACHINE LEARNING)
# =====================================================================
print("====== LANGKAH 2: MELATIH MODEL PCA + SVM ======")

# 1. Pipihkan gambar 3D (128x128x3) menjadi 1D array (49.152 fitur)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 2. Reduksi dimensi dengan PCA (ambil 50 komponen utama paling informatif)
n_components = min(50, X_train.shape[0])
pca = PCA(n_components=n_components, whiten=True, random_state=42)

X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)
print(f"✓ Dimensi fitur dipangkas dari {X_train_flat.shape[1]} menjadi {X_train_pca.shape[1]} komponen.")

# 3. Klasifikasi menggunakan algoritma SVM
classifier_svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
classifier_svm.fit(X_train_pca, y_train)

# 4. Evaluasi Hasil PCA + SVM
y_pred_pca = classifier_svm.predict(X_test_pca)
print(f"\n[HASIL EVALUASI PCA + SVM]")
print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred_pca) * 100:.2f}%")
print(classification_report(y_test, y_pred_pca, target_names=classes))


# =====================================================================
# LANGKAH 3: KLASIFIKASI MENGGUNAKAN CNN (DEEP LEARNING)
# =====================================================================
print("\n====== LANGKAH 3: MELATIH MODEL CNN (DEEP LEARNING) ======")

# 1. Membangun Struktur Arsitektur Jaringan CNN
model_cnn = models.Sequential([
    # Input layer & Ekstraksi Fitur Tahap 1
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Ekstraksi Fitur Tahap 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Ekstraksi Fitur Tahap 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Lapisan Klasifikasi Akhir
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Mencegah model menghafal pola gambar (overfitting)
    layers.Dense(num_classes, activation='softmax')  # Output kelas
])

# 2. Mengompilasi (Compile) Model CNN
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Proses Pelatihan (Training) Model CNN
EPOCHS = 10
print(f"Memulai training CNN selama {EPOCHS} Epochs...")
history = model_cnn.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_split=0.2,  # 20% data latih disisihkan untuk validasi berkala
    batch_size=32
)

# 4. Evaluasi Hasil Akhir CNN
print(f"\n[HASIL EVALUASI CNN]")
y_pred_cnn = np.argmax(model_cnn.predict(X_test), axis=1)
print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred_cnn) * 100:.2f}%")
print(classification_report(y_test, y_pred_cnn, target_names=classes)) 

# Simpan model setelah dievaluasi
model_cnn.save('model_kopi_cnn.h5')
print("✓ Model berhasil disimpan dengan nama 'model_kopi_cnn.h5'")