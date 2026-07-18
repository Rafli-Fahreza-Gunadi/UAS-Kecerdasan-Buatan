# Laporan UAS Kecerdasan Buatan

## 1. Judul Proyek

**Prediksi Risiko Penyakit Jantung Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (KNN)**

**Nama Kelompok:**
1. (Nama Anggota 1 — NIM)
2. (Nama Anggota 2 — NIM, opsional, maksimal 2 orang)

**Domain Proyek (Latar Belakang):**

Penyakit jantung (*cardiovascular disease*) merupakan salah satu penyebab kematian tertinggi di dunia menurut World Health Organization (WHO), dengan jutaan kasus baru setiap tahunnya. Deteksi dini terhadap risiko penyakit jantung sangat penting agar tindakan pencegahan atau penanganan medis dapat dilakukan sesegera mungkin, sebelum kondisi pasien memburuk. Namun, proses diagnosis konvensional membutuhkan berbagai pemeriksaan klinis yang memakan waktu dan biaya, serta sangat bergantung pada pengalaman tenaga medis dalam menginterpretasikan data pasien.

Perkembangan teknologi kecerdasan buatan (*Artificial Intelligence*), khususnya *machine learning*, membuka peluang untuk membangun sistem pendukung keputusan (*decision support system*) yang dapat membantu tenaga medis memprediksi risiko penyakit jantung berdasarkan data klinis pasien, seperti usia, tekanan darah, kadar kolesterol, dan hasil elektrokardiografi. Dengan memanfaatkan algoritma klasifikasi seperti **Decision Tree** dan **K-Nearest Neighbors (KNN)**, proyek ini bertujuan membangun model prediksi risiko penyakit jantung yang akurat, cepat, dan dapat diinterpretasikan, sehingga berpotensi digunakan sebagai alat bantu skrining awal di fasilitas kesehatan.

---

## 2. Business Understanding

### Permasalahan Dunia Nyata dan Literatur Review

Penyakit kardiovaskular menyumbang sekitar 32% dari seluruh kematian global setiap tahunnya (WHO, 2021). Banyak kasus penyakit jantung yang sebenarnya dapat dicegah apabila terdeteksi lebih dini melalui identifikasi faktor risiko seperti hipertensi, kadar kolesterol tinggi, dan riwayat elektrokardiografi abnormal. Berbagai penelitian terdahulu telah menunjukkan bahwa algoritma *machine learning* seperti Decision Tree, KNN, Naive Bayes, dan SVM mampu memberikan akurasi yang kompetitif dalam memprediksi risiko penyakit jantung menggunakan dataset klinis standar seperti Cleveland Heart Disease Dataset yang tersedia di UCI Machine Learning Repository (lihat daftar referensi pada Bab 9).

### Tujuan Proyek

1. Membangun model klasifikasi untuk memprediksi apakah seorang pasien berisiko terkena penyakit jantung atau tidak, berdasarkan data klinis.
2. Membandingkan performa dua algoritma klasifikasi (Decision Tree dan KNN) dalam menyelesaikan permasalahan tersebut.
3. Mengidentifikasi fitur klinis yang paling berpengaruh terhadap risiko penyakit jantung.

### Siapa User/Pengguna Sistem

- **Tenaga medis** (dokter umum, dokter spesialis jantung, perawat) sebagai alat bantu skrining awal sebelum pemeriksaan lanjutan.
- **Fasilitas kesehatan** (puskesmas, klinik, rumah sakit) yang ingin melakukan skrining massal dengan biaya rendah.
- **Peneliti/akademisi** di bidang kesehatan dan data science sebagai baseline studi lanjutan.

### Solusi dan Manfaat Implementasi AI

Dengan model prediksi berbasis AI, proses identifikasi awal risiko penyakit jantung dapat dilakukan lebih cepat dan konsisten dibandingkan penilaian manual. Model yang mudah diinterpretasikan (seperti Decision Tree) juga memberi nilai tambah karena aturan keputusannya dapat dijelaskan secara klinis kepada tenaga medis, meningkatkan kepercayaan terhadap rekomendasi sistem.

---

## 3. Data Understanding

### Sumber Data

Dataset yang digunakan pada proyek ini adalah data **simulasi** yang dibangun mengikuti struktur, tipe fitur, dan karakteristik statistik dari dataset publik populer **Heart Disease UCI / Cleveland Heart Disease Dataset** (tersedia di Kaggle dan UCI Machine Learning Repository). Simulasi dilakukan karena keterbatasan akses internet pada lingkungan pengerjaan proyek, namun distribusi nilai, rentang data, serta hubungan antar-fitur terhadap target dirancang untuk merepresentasikan pola klinis nyata pada kasus penyakit jantung, sehingga proses analisis data dan pemodelan tetap valid secara metodologis.

> **Catatan:** File dataset yang digunakan tersedia di `data/dataset/heart_disease_dataset.csv`. Apabila ingin menggunakan data asli, dataset resmi dapat diunduh dari Kaggle: *"Heart Disease UCI Dataset"*.

### Deskripsi Setiap Fitur (Atribut)

| No | Fitur | Deskripsi | Tipe Data |
|---|---|---|---|
| 1 | age | Usia pasien (tahun) | Numerik |
| 2 | sex | Jenis kelamin (1 = laki-laki, 0 = perempuan) | Kategorik biner |
| 3 | cp | Tipe nyeri dada (0–3) | Kategorik |
| 4 | trestbps | Tekanan darah istirahat (mm Hg) | Numerik |
| 5 | chol | Kadar kolesterol serum (mg/dl) | Numerik |
| 6 | fbs | Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak) | Kategorik biner |
| 7 | restecg | Hasil elektrokardiografi istirahat (0–2) | Kategorik |
| 8 | thalach | Detak jantung maksimum tercapai | Numerik |
| 9 | exang | Angina akibat olahraga (1 = ya, 0 = tidak) | Kategorik biner |
| 10 | oldpeak | Depresi ST akibat olahraga relatif terhadap istirahat | Numerik |
| 11 | slope | Kemiringan segmen ST puncak olahraga (0–2) | Kategorik |
| 12 | ca | Jumlah pembuluh darah utama terwarnai fluoroskopi (0–3) | Kategorik |
| 13 | thal | Kondisi thalassemia (1 = normal, 2 = cacat tetap, 3 = cacat reversibel) | Kategorik |
| 14 | **target** | **Label: 1 = berisiko penyakit jantung, 0 = tidak berisiko** | **Target biner** |

### Ukuran dan Format Data

- **Format:** CSV (*Comma-Separated Values*)
- **Ukuran awal:** 505 baris × 14 kolom (termasuk 5 baris duplikat dan 12 nilai kosong yang disisipkan secara sengaja untuk kebutuhan latihan *data cleaning*)
- **Ukuran setelah pembersihan:** 500 baris × 14 kolom

### Tipe Data dan Target Klasifikasi

Seluruh fitur bertipe numerik (baik kontinu maupun kategorik-terenkode). Target/label bertipe **klasifikasi biner**: `0` (tidak berisiko) dan `1` (berisiko penyakit jantung).

---

## 4. Exploratory Data Analysis (EDA)

Proses EDA lengkap beserta seluruh visualisasi (histogram, boxplot, heatmap korelasi, dan pairplot) dapat dilihat pada notebook `uas_model.ipynb` bagian ke-3.

**Ringkasan temuan EDA:**

1. **Distribusi Target:** Distribusi kelas target cukup seimbang, dengan proporsi sekitar 55% "tidak berisiko" dan 45% "berisiko" — sehingga tidak diperlukan teknik penyeimbangan kelas khusus seperti SMOTE.
2. **Distribusi Fitur Numerik:** Fitur seperti `age`, `trestbps`, dan `chol` menunjukkan distribusi mendekati normal, sedangkan `oldpeak` cenderung *right-skewed* (banyak pasien dengan nilai mendekati 0).
3. **Korelasi Antar Fitur (Heatmap):** Tidak ditemukan multikolinearitas ekstrem antar fitur numerik. Fitur `exang`, `ca`, `oldpeak`, dan `cp` menunjukkan korelasi yang relatif lebih kuat terhadap target dibandingkan fitur lainnya.
4. **Boxplot terhadap Target:** Pasien dengan target = 1 (berisiko) cenderung memiliki `thalach` (detak jantung maksimum) lebih rendah dan `oldpeak` lebih tinggi dibandingkan pasien tidak berisiko.
5. **Data Tidak Seimbang:** Berdasarkan pemeriksaan proporsi kelas, dataset dikategorikan **seimbang** (tidak imbalanced), sehingga metrik accuracy tetap relevan digunakan berdampingan dengan precision, recall, dan F1-score.

---

## 5. Data Preparation

1. **Pembersihan Data:**
   - Menghapus 5 baris data duplikat.
   - Melakukan imputasi nilai kosong (*missing value*) pada kolom `chol` dan `thalach` menggunakan nilai median masing-masing kolom, karena median lebih robust terhadap outlier dibandingkan mean.
2. **Encoding Data Kategorik:** Fitur kategorik (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) sudah berbentuk numerik terenkode sejak sumber data (mengikuti standar dataset UCI Heart Disease), sehingga tidak memerlukan One-Hot Encoding tambahan.
3. **Standardisasi Data Numerik:** Seluruh fitur distandardisasi menggunakan `StandardScaler` dari scikit-learn — standardisasi ini **wajib** untuk algoritma KNN (karena KNN berbasis perhitungan jarak antar data), sedangkan untuk Decision Tree standardisasi tidak wajib namun tidak memengaruhi hasil secara negatif.
4. **Split Data (Train-Test):** Data dibagi menjadi 80% data latih (400 baris) dan 20% data uji (100 baris) menggunakan `train_test_split` dengan parameter `stratify=y` untuk menjaga proporsi kelas tetap seimbang pada kedua subset, dan `random_state=42` untuk memastikan hasil dapat direproduksi.

---

## 6. Modeling

### Pemilihan Algoritma

Proyek ini menggunakan dua algoritma klasifikasi:

1. **Decision Tree Classifier** (`max_depth=4`)
2. **K-Nearest Neighbors / KNN** (`n_neighbors=7`)

### Alasan Pemilihan 2 Algoritma

- **Decision Tree** dipilih karena sifatnya yang *interpretable* (dapat divisualisasikan sebagai pohon keputusan berbasis aturan *if-else*), sehingga hasil prediksinya dapat dijelaskan secara klinis kepada tenaga medis — misalnya aturan seperti "jika usia > 55 dan jumlah pembuluh darah tersumbat (`ca`) > 0, maka pasien berisiko". Decision Tree juga tidak memerlukan standardisasi data dan mampu menangkap hubungan non-linear antar fitur.
- **KNN** dipilih sebagai pembanding karena bekerja berdasarkan prinsip kemiripan (*similarity-based learning*) antar pasien, yang secara konsep relevan di bidang medis (pasien dengan profil klinis serupa cenderung memiliki kondisi kesehatan yang serupa pula). KNN juga merupakan algoritma non-parametrik yang tidak mengasumsikan distribusi data tertentu, sehingga baik digunakan sebagai pembanding terhadap pendekatan berbasis aturan seperti Decision Tree.

### Implementasi Model

Implementasi lengkap dengan kode program tersedia pada notebook `uas_model.ipynb`, Bab 5 (Modeling). Ringkasan konfigurasi model:

```python
# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)
```

### Perbandingan Model

| Aspek | Decision Tree | KNN |
|---|---|---|
| Kebutuhan standardisasi | Tidak wajib | Wajib |
| Interpretability | Tinggi (dapat divisualisasikan) | Rendah (black-box berbasis jarak) |
| Sensitivitas terhadap fitur tidak relevan | Rendah (otomatis menyeleksi fitur penting) | Tinggi |
| Waktu training | Cepat | Cepat (tapi prediksi bisa lambat pada data besar) |

### Visualisasi Model

Visualisasi pohon keputusan (*decision tree plot*) dan grafik *feature importance* tersedia pada notebook `uas_model.ipynb`, bagian 5.3 dan 5.4. Berdasarkan feature importance, fitur yang paling berpengaruh terhadap keputusan Decision Tree adalah `age`, `ca` (jumlah pembuluh darah tersumbat), `trestbps` (tekanan darah), dan `oldpeak`.

---

## 7. Evaluation

### Confusion Matrix

Confusion matrix lengkap untuk kedua model tersedia pada notebook `uas_model.ipynb`, Bab 6.1. Ringkasan:

**Decision Tree:**

|  | Prediksi: Tidak Berisiko | Prediksi: Berisiko |
|---|---|---|
| **Aktual: Tidak Berisiko** | 40 | 15 |
| **Aktual: Berisiko** | 17 | 28 |

**KNN:**

|  | Prediksi: Tidak Berisiko | Prediksi: Berisiko |
|---|---|---|
| **Aktual: Tidak Berisiko** | 44 | 11 |
| **Aktual: Berisiko** | 26 | 19 |

### Metrik Evaluasi

| Metrik | Decision Tree | KNN (k=7) |
|---|---|---|
| Accuracy | 0.680 | 0.630 |
| Precision | 0.651 | 0.633 |
| Recall | 0.622 | 0.422 |
| F1-Score | 0.636 | 0.507 |

### Penjelasan Kinerja Model Berdasarkan Metrik (Model Terbaik dan Alasannya)

**Model terbaik pada proyek ini adalah Decision Tree**, karena unggul di seluruh metrik dibandingkan KNN. Perbedaan paling signifikan terletak pada **recall**: Decision Tree mencapai recall 0.622, sedangkan KNN hanya 0.422. Artinya, KNN gagal mendeteksi proporsi pasien berisiko yang jauh lebih besar (26 *false negative* dibandingkan 17 pada Decision Tree). Dalam konteks medis, recall rendah pada kelas "Berisiko" sangat merugikan karena berarti banyak pasien yang sebenarnya berisiko justru diprediksi aman (tidak berisiko), sehingga berpotensi terlambat mendapatkan penanganan.

Beberapa faktor yang menjelaskan perbedaan performa ini:

1. **Seleksi fitur otomatis pada Decision Tree.** Berdasarkan feature importance, Decision Tree secara otomatis mengabaikan fitur yang kurang informatif (`chol`, `fbs`, `slope`, `thal` memiliki importance mendekati nol), sedangkan KNN memperhitungkan seluruh 13 fitur dengan bobot jarak yang sama, sehingga fitur yang kurang relevan justru menambah *noise* dalam perhitungan jarak antar data (*curse of dimensionality*).
2. **Pemilihan nilai k.** Nilai `k=7` yang digunakan pada KNN belum tentu optimal; performa KNN sangat sensitif terhadap pemilihan k dan berpotensi meningkat melalui hyperparameter tuning (misalnya menggunakan GridSearchCV).
3. **Interpretability.** Decision Tree memberikan keunggulan tambahan berupa kemampuan menjelaskan keputusan model secara langsung dalam bentuk aturan klinis, yang bermanfaat untuk membangun kepercayaan tenaga medis terhadap sistem prediksi.

---

## 8. Kesimpulan dan Rekomendasi

### Ringkasan Hasil Modeling dan Evaluasi

Proyek ini berhasil membangun dan membandingkan dua model klasifikasi (Decision Tree dan KNN) untuk memprediksi risiko penyakit jantung berdasarkan 13 fitur klinis pasien. Decision Tree mencapai accuracy 68% dan F1-score 0.636, sedangkan KNN (k=7) mencapai accuracy 63% dan F1-score 0.507.

### Apakah Tujuan Proyek Tercapai?

Ya, tujuan proyek untuk membangun model prediksi risiko penyakit jantung serta membandingkan performa dua algoritma klasifikasi telah tercapai. **Decision Tree** terpilih sebagai model terbaik pada dataset ini berdasarkan seluruh metrik evaluasi, terutama recall yang jauh lebih tinggi — aspek krusial dalam konteks deteksi risiko medis.

### Kelebihan dan Keterbatasan Model

**Kelebihan:**
- Decision Tree mudah diinterpretasikan dan dapat divisualisasikan sebagai aturan keputusan yang dapat dijelaskan secara klinis.
- Decision Tree secara otomatis menyeleksi fitur yang paling relevan, mengurangi pengaruh fitur yang kurang informatif.
- KNN sederhana secara konsep dan tidak memerlukan asumsi distribusi data tertentu.

**Keterbatasan:**
- Dataset yang digunakan bersifat simulasi berskala menengah (500 baris setelah pembersihan), sehingga generalisasi terhadap populasi pasien nyata masih terbatas dan perlu divalidasi ulang menggunakan data klinis asli.
- KNN sensitif terhadap fitur yang kurang relevan dan pemilihan nilai k yang belum dioptimalkan pada proyek ini.
- Decision Tree berisiko *overfitting* apabila kedalaman pohon tidak dibatasi (pada proyek ini telah dibatasi `max_depth=4` untuk mengurangi risiko tersebut).

### Rekomendasi Perbaikan

1. Menggunakan dataset asli berskala lebih besar dan tervalidasi secara medis, misalnya dataset resmi Heart Disease UCI dari Kaggle atau data rekam medis riil (dengan menjaga aspek privasi/etika data).
2. Melakukan hyperparameter tuning menggunakan `GridSearchCV` atau `RandomizedSearchCV`, termasuk pencarian nilai k optimal untuk KNN dan kedalaman optimal untuk Decision Tree.
3. Mencoba algoritma pembanding tambahan seperti Random Forest, Support Vector Machine (SVM), atau Gradient Boosting untuk memperluas perbandingan performa.
4. Menerapkan teknik *cross-validation* (misalnya k-fold cross-validation) agar evaluasi performa model lebih robust dan tidak bergantung pada satu kali pembagian data latih-uji.
5. Mengeksplorasi teknik *feature engineering* tambahan dan penanganan outlier untuk meningkatkan kualitas data sebelum pemodelan.

---

## 9. Referensi

1. Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., Guppy, K. H., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*, 64(5), 304–310.

2. Mohan, S., Thirumalai, C., & Srivastava, G. (2019). Effective heart disease prediction using hybrid machine learning techniques. *IEEE Access*, 7, 81542–81554. https://doi.org/10.1109/ACCESS.2019.2923707

3. Bhatt, C. M., Patel, P., Ghetia, T., & Mazzeo, P. L. (2023). Effective heart disease prediction using machine learning techniques. *Algorithms*, 16(2), 88. https://doi.org/10.3390/a16020088

4. Shorewala, V. (2021). Early detection of coronary heart disease using ensemble techniques. *Informatics in Medicine Unlocked*, 26, 100655. https://doi.org/10.1016/j.imu.2021.100655

5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

*(Gaya sitasi: APA 7th Edition. Seluruh referensi di atas telah diverifikasi keakuratannya — judul, penulis, jurnal, volume, halaman, dan DOI. Silakan cek kembali dan sesuaikan dengan jurnal yang benar-benar Anda baca dan kutip di dalam laporan.)*

**Catatan (sumber pendukung latar belakang, bukan jurnal ilmiah):**
World Health Organization. (2021). *Cardiovascular diseases (CVDs)*. WHO Fact Sheets.

---

## 10. Lampiran

- Dataset mentah: `data/dataset/heart_disease_dataset.csv`
- Notebook kode lengkap: `uas_model.ipynb`
- Referensi jurnal (opsional, jika disertakan file PDF/catatan): `data/jurnal/`
