# UAS Kecerdasan Buatan
## Prediksi Risiko Penyakit Jantung Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (KNN)

Repository ini berisi jawaban Ujian Akhir Semester (UAS) mata kuliah Kecerdasan Buatan, berupa proyek klasifikasi risiko penyakit jantung menggunakan dua algoritma: **Decision Tree** dan **K-Nearest Neighbors (KNN)**.

## Anggota Kelompok

1. (Nama Anggota 1 — NIM)
2. (Nama Anggota 2 — NIM, opsional, maksimal 2 orang)


## Ringkasan Proyek

- **Domain:** Kesehatan — prediksi risiko penyakit jantung
- **Jenis masalah:** Klasifikasi biner (berisiko / tidak berisiko)
- **Dataset:** Data simulasi (505 baris → 500 baris setelah dibersihkan) yang dibangun mengikuti struktur dan karakteristik dataset publik *Heart Disease UCI*
- **Algoritma:** Decision Tree dan KNN
- **Hasil terbaik:** Decision Tree (accuracy 0.68, F1-score 0.636), mengungguli KNN (accuracy 0.63, F1-score 0.507)

Penjelasan lengkap ada di `Laporan_uas.md`.

## Langkah Pengerjaan

1. **Data Understanding** — Memahami struktur dataset: 13 fitur klinis (usia, tekanan darah, kolesterol, dll.) dan 1 target (risiko penyakit jantung: 0/1).
2. **Exploratory Data Analysis (EDA)** — Melihat distribusi target, distribusi fitur numerik (histogram), hubungan fitur terhadap target (boxplot), korelasi antar fitur (heatmap), dan pairplot.
3. **Data Preparation** — Menghapus data duplikat, menangani missing value (imputasi median), memastikan data kategorik sudah dalam bentuk numerik, standardisasi fitur numerik (`StandardScaler`), dan split data latih-uji (80:20).
4. **Modeling** — Melatih dua model: `DecisionTreeClassifier(max_depth=4)` dan `KNeighborsClassifier(n_neighbors=7)`.
5. **Evaluation** — Mengevaluasi kedua model menggunakan confusion matrix, accuracy, precision, recall, dan F1-score, lalu membandingkan performanya.
6. **Kesimpulan** — Decision Tree dipilih sebagai model terbaik karena unggul di seluruh metrik, terutama recall yang penting dalam konteks medis.

Detail kode lengkap ada di `uas_model.ipynb`, dan penjelasan naratif lengkap (10 bagian sesuai instruksi tugas) ada di `Laporan_uas.md`.

## Referensi

Daftar lengkap referensi jurnal ilmiah (minimal 5) tersedia pada `Laporan_uas.md` Bab 9 - Referensi.
