# Laporan Proyek Machine Learning – Paulina Hambali 
## Proyek Predictive Analytics: Diabetes Risk Prediction

## Domain Proyek

Diabetes dikategorikan sebagai penyakit kronis, karena berdampak pada jutaan orang di dunia. Prediksi terhadap potensi diabetes dapat membantu dalam penanganan yang lebih awal dan pencegahan komplikasi serius. Pendekatan dalam machine learning dapat dimanfaatkan untuk mengembangkan sistem prediksi diabetes yang akurat. Menurut World Health Organization (WHO), pada tahun 2021 terdapat lebih dari 2 juta penderita diabetes di seluruh dunia.

Referensi:
- World Health Organization. “Diabetes.” [https://www.who.int/news-room/fact-sheets/detail/diabetes]
- Smith, J. (2020). *Predictive Modelling in Healthcare Using Machine Learning*. Journal of Medical Informatics.

## Business Understanding

### Problem Statements

1. Bagaimana melakukan prediksi seseorang berpotensi menderita penyakit diabetes berdasarkan data klinis?  
2. Algoritma dalam machine learning mana yang paling efektif dalam memprediksi potensi penyakit diabetes? 

### Goals

1. Mengembangkan model klasifikasi machine learning untuk memprediksi potensi diabetes.  
2. Membandingkan performa model algoritma KNN, Random Forest, dan Gradient Boosting.

### Solution Statements

- Membangun tiga model prediksi dengan K-Nearest Neighbors (KNN), Random Forest, dan Gradient Boosting Classifier.  
- Melakukan evaluasi dan membandingkan model menggunakan metrik klasifikasi: akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset yang digunakan adalah **Analysis Diabetes Dataset** yang berasal dari https://www.kaggle.com/datasets/whenamancodes/predict-diabities.

### Fitur pada dataset:

- Pregnancies: Jumlah kehamilan  
- Glucose: Kadar glukosa dalam darah  
- BloodPressure: Tekanan darah diastolik  
- SkinThickness: Ketebalan lipatan kulit triceps  
- Insulin: Kadar insulin serum  
- BMI: Indeks massa tubuh  
- DiabetesPedigreeFunction: Riwayat genetik diabetes  
- Age: Umur  
- Outcome: Label target (1 = Diabetes, 0 = Tidak Diabetes) — target  
Beberapa fitur dalam dataset dengan penjelasan (kegunaan) fitur.

### Exploratory Data Analysis

1. **Rows in Dataset:** Membaca dan menampilkan jumlah baris dalam dataset.  
2. **Total Rows and Columns & Description:** Menampilkan total baris dan kolom serta mendeskripsikan kolom dalam dataset.  
3. **Data Type Check:** Menampilkan tipe data dalam dataset.  
4. **Check Missing Values:** Memeriksa data kosong atau hilang.  
5. **Descriptive Statistics:** Melakukan analisis deskriptif dataset.  
6. **Check Target Class:** Memeriksa distribusi kolom target.  
7. **Visualization: Distribution of Target Class:** Visualisasi dalam bentuk chart kolom Outcome (target).  
8. **Visualization: Distribution of Numeric Features:** Visualisasi dalam bentuk histogram kolom numerik.  
9. **Visualization: Correlation Between Feature and Target & Outlier Detection (numeric features):** Visualisasi korelasi antara Outcome (target) dengan kolom numerik dalam bentuk boxplot.

## Data Preparation

Langkah-langkah yang dilakukan:

1. **Feature and Target:** Mengatur fitur dan target berdasarkan hasil EDA.  
2. **Train-Test Split:** Membagi data menjadi 80% training dan 20% testing.  
3. **Feature Scaling:** Menggunakan MinMaxScaler agar semua fitur berada dalam skala 0–1.  

**Alasan:** Model sensitif terhadap skala data sehingga diperlukan normalisasi. Tidak dilakukan PCA karena dataset hanya memiliki delapan fitur sehingga tidak diperlukan reduksi dimensi.

## Model Development (Modelling)
Sebelum model digunakan, model diinisialisasi terlebih dahulu, yakni 
- knn = KNeighborsClassifier(n_neighbors=10): KNN bekerja dengan mencari k tetangga terdekat dari data baru berdasarkan jarak, disini n_neighbord berarti 10 tetangga terdekat dan mengklasifikasikan berdasarkan mayoritas kelas dari tetangga-tetangga tersebut.
- rf = RandomForestClassifier(random_state=42): algoritma berbasis ensemble yang membangun banyak pohon keputusan (decision trees) dan menggabungkan hasilnya untuk prediksi akhir. random_state = 42 berarti selama membangun model akan dilakukan proses acak dengan seed = 42.
- gb = GradientBoostingClassifier(random_state=42): Gradient Boosting membangun model secara bertahap. Tiap model baru dibuat untuk memperbaiki kesalahan dari model sebelumnya. Diterapkan juga random_state=42 dilakukan proses acak dengan seed = 42

Lanjut tahap training data, dimana setiap model dilatih menggunakan data pelatihan (X_train, y_train) dan akan dicatat akurasinya, penerapannya: 
- models.loc['train_acc', 'KNN'] = accuracy_score(y_train, knn.predict(X_train))
- models.loc['train_acc', 'RandomForest'] = accuracy_score(y_train, rf.predict(X_train))
- models.loc['train_acc', 'Boosting'] = accuracy_score(y_train, gb.predict(X_train))

Setelah model dilatih menggunakan data training, langkah selanjutnya adalah melakukan prediksi terhadap data testing. Tujuannya adalah untuk mengetahui seberapa baik model dalam memprediksi data yang belum pernah dilihat sebelumnya.
- models.loc['test_acc', 'KNN'] = accuracy_score(y_test, knn.predict(X_test))
- models.loc['test_acc', 'RandomForest'] = accuracy_score(y_test, rf.predict(X_test)) melakukan prediksi pada test data dan mencatat akurasinya.
- models.loc['test_acc', 'Boosting'] = accuracy_score(y_test, gb.predict(X_test)) melakukan prediksi pada trest data dan mencatat akurasinya.

Penjelasan tiga model digunakan:

1. **K-Nearest Neighbors (KNN):**
Model ini bekerja dengan cara melihat data baru yang akan diprediksi, lalu mencari “tetangga” atau data terdekat dari data tersebut       sebanyak k (di sini 10 tetangga). Setelah itu, KNN menentukan kelas data baru berdasarkan mayoritas kelas dari tetangga-tetangga          terdekat tersebut. Jadi, kalau sebagian besar tetangga berlabel “diabetes”, maka data baru juga akan diprediksi sebagai “diabetes”.
   - `n_neighbors=10`  
   - **Kelebihan:** Model sederhana dan cepat, cocok untuk dataset dengan fitur sedikit.  
   - **Kekurangan:** Performa menurun pada dataset dengan banyak fitur (curse of dimensionality).

3. **Random Forest:**
Model ini terdiri dari kumpulan banyak pohon keputusan (decision trees). Setiap pohon akan dilatih dengan data yang berbeda secara acak, dan akan  membuat prediksi sendiri-sendiri. Setelah semua pohon membuat prediksi akan diambil  suara terbanyak (mayoritas) dari pohon-pohon tersebut sebagai hasil akhir. 
   - `random_state=42`  
   - **Kelebihan:**  
     - Mampu menangani data dengan fitur yang banyak dan kompleks.  
     - Minim risiko overfitting karena teknik bagging.  
     - Dapat memberikan estimasi pentingnya fitur (feature importance).  
   - **Kekurangan:**  
     - Lebih sulit diinterpretasi dibandingkan model sederhana.  
     - Membutuhkan lebih banyak sumber daya komputasi.

5. **Gradient Boosting Classifier:**
Model ini membangun secara bertahap dan bertingkat. Model pertama dibuat untuk memprediksi data, kemudian model selanjutkan akan dibuat untuk memperbaiki kesalahan dari model sebelumnya. Setiap model baru akan fokus dalam menangani data yang sudah diprediksi sebelumnya. Proses ini dilakukan berulang kali hingga model cukup bagus dalam memprediksi.
   - `random_state=42`  
   - **Kelebihan:** Performa tinggi dalam prediksi.  
   - **Kekurangan:** Waktu training lebih lama dibanding model lain.

## Evaluation

Metrik yang digunakan:

- **Accuracy:** (TP + TN) / Total  
- **Precision:** TP / (TP + FP)  
- **Recall:** TP / (TP + FN)  
- **F1-score:** Rata-rata antara precision dan recall  

| Model           | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | Accuracy |
|-----------------|---------------|------------|--------------|---------------|------------|--------------|----------|
| KNN             | 0.802         | 0.907      | 0.851        | 0.697         | 0.489      | 0.575        | 0.779    |
| Random Forest   | 0.879         | 0.879      | 0.879        | 0.723         | 0.723      | 0.723        | 0.831    |
| Gradient Boosting| 0.868         | 0.860      | 0.864        | 0.688         | 0.702      | 0.695        | 0.812    |

Dari hasil laporan di atas, dapat disimpulkan bahwa model Random Forest memiliki performa terbaik dibanding dua model lainnya, dengan F1-Score tertinggi untuk kelas 1 (pengidap diabetes) yaitu sebesar 0.723 atau 72.3% serta akurasi tertinggi yaitu 0.831 atau 83.1%. Hal ini menunjukkan model ini cukup akurat dalam memprediksi potensi diabetes. Untuk skor precision tertinggi pada kelas 1 dicapai oleh Random Forest sebesar 0.723 atau 72.3%, sedangkan skor recall tertinggi pada kelas 1 juga dimiliki oleh Random Forest dengan nilai yang sama, yaitu 0.723 atau 72.3%. Precision yang tinggi menunjukkan bahwa dari semua prediksi positif (mengidap diabetes), sebanyak 72.3% prediksi tersebut benar-benar akurat dan bukan false positive. Sedangkan recall yang tinggi menunjukkan model mampu mendeteksi sebanyak 72.3% dari seluruh pasien yang benar-benar mengidap diabetes, sehingga meminimalkan kasus false negative.

Lalu, prediksi terhadap bukan pengidap diabetes (kelas 0) juga sangat baik, dengan F1-Score tertinggi sebesar 0.879 atau 87.9%, yang menunjukkan kemampuan model dalam mengenali pasien sehat dengan sangat baik. Untuk kelas 0, skor precision tertinggi ada pada Random Forest dengan nilai 0.879 atau 87.9%, artinya dari semua pasien yang diprediksi sehat, 87.9% benar-benar sehat. Sedangkan skor recall tertinggi pada kelas 0 dimiliki oleh KNN dengan nilai 0.907 atau 90.7%, yang menunjukkan KNN berhasil mendeteksi 90.7% dari seluruh pasien sehat secara tepat.

Berikut kesimpulan keseluruhan proyek prediksi potensi diabetes dengan memperhatikan business understanding yang telah dijabarkan sebelumnya. 

**Jawaban Pertanyaan 1:** Proyek ini dapat memprediksi potensi diabetes dengan dibangun menggunakan dataset yang didalamnya terdapat beberapa fitur yang dapat digunakan untuk memprediksi apakah seseorang mengidap penyakit diabetes atau tidak, yakni  
- Pregnancies: Jumlah kehamilan  
- Glucose: Kadar glukosa dalam darah  
- BloodPressure: Tekanan darah diastolik  
- SkinThickness: Ketebalan lipatan kulit triceps  
- Insulin: Kadar insulin serum  
- BMI: Indeks massa tubuh  
- DiabetesPedigreeFunction: Riwayat genetik diabetes  
- Age: Umur  
- Outcome: Label target (1 = Diabetes, 0 = Tidak Diabetes)
  
**Jawaban Pertanyaan 2:** Menggunakan tiga model, dan disimpulkan model yang terbaik untuk dipakai dalam memprediksi, yakni Random Forest karena memiliki akurasi tertinggi dalam memprediksi potensi diabetes maupun bukan pengidap diabetes. 

**Goals:** telah tercapai, yakni mampu untuk membangun model klasifikasi machine learning dalam memprediksi potensi diabetes dan telah membandingkan performa model KNN, Random Forest, dan GradientBoosting berdasarkan classification report. 

**Problem solution:** telah diterapkan dan berhasil dalam membangun model serta memprediksi potensi diabetes dengan menerapkan tiga model KNN, Random Forest, dan Gradient Boosting. Serta menerapkan metrik evaluasi untuk melakukan evaluasi terhadap model yang digunakan. 

---
