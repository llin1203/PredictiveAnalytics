# Machine Learning Project Report – Paulina Hambali 

## Nama Proyek  
Proyek Predictive Analytics: Diabetes Risk Prediction

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

## Modeling

Tiga model digunakan:

1. **K-Nearest Neighbors (KNN):**  
   - `n_neighbors=10`  
   - **Kelebihan:** Model sederhana dan cepat, cocok untuk dataset dengan fitur sedikit.  
   - **Kekurangan:** Performa menurun pada dataset dengan banyak fitur (curse of dimensionality).

2. **Random Forest:**  
   - `random_state=42`  
   - **Kelebihan:**  
     - Mampu menangani data dengan fitur yang banyak dan kompleks.  
     - Minim risiko overfitting karena teknik bagging.  
     - Dapat memberikan estimasi pentingnya fitur (feature importance).  
   - **Kekurangan:**  
     - Lebih sulit diinterpretasi dibandingkan model sederhana.  
     - Membutuhkan lebih banyak sumber daya komputasi.

3. **Gradient Boosting Classifier:**  
   - `random_state=42`  
   - **Kelebihan:** Performa tinggi dalam prediksi.  
   - **Kekurangan:** Waktu training lebih lama dibanding model lain.

## Evaluation

Metrik yang digunakan:

- **Accuracy:** (TP + TN) / Total  
- **Precision:** TP / (TP + FP)  
- **Recall:** TP / (TP + FN)  
- **F1-score:** Rata-rata harmonis antara precision dan recall  

| Model         | Accuracy | Precision (1) | Recall (1) | F1-score (1) |
|---------------|----------|---------------|------------|--------------|
| KNN           | 0.779    | 0.697         | 0.489      | 0.575        |
| Random Forest | 0.831    | 0.723         | 0.723      | 0.723        |
| Boosting      | 0.812    | 0.688         | 0.702      | 0.695        |

Dari hasil laporan di atas, dapat disimpulkan bahwa model **Random Forest** memiliki performa terbaik dari dua model lainnya, dengan akurasi dan F1-Score tertinggi, yakni 0.723 atau 72% dalam mendeteksi potensi diabetes.

---
