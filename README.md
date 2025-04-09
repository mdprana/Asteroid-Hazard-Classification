# Laporan Proyek Machine Learning - Made Pranajaya Dibyacita
---

## Domain Proyek
---

### Latar Belakang
Asteroid adalah benda langit berbatu yang mengorbit Matahari dan memiliki ukuran yang lebih kecil daripada planet. Ribuan asteroid telah diidentifikasi dan dikatalogkan oleh para astronom. Beberapa di antaranya dikategorikan sebagai Potentially Hazardous Asteroids (PHAs) atau asteroid yang berpotensi berbahaya karena orbitnya yang dekat dengan orbit Bumi dan ukurannya yang cukup besar untuk menyebabkan kerusakan signifikan jika terjadi tumbukan.

Memprediksi apakah sebuah asteroid berpotensi berbahaya atau tidak adalah masalah penting dalam astronomi dan perlindungan planet. NASA dan badan antariksa lainnya secara aktif melacak dan mengklasifikasikan asteroid untuk mengidentifikasi potensi ancaman. Dengan memanfaatkan machine learning, kita dapat mengembangkan model yang memprediksi status bahaya asteroid berdasarkan karakteristik fisik dan orbitalnya.

Ancaman tumbukan asteroid terhadap Bumi bukanlah skenario fiksi ilmiah semata. Sepanjang sejarah Bumi, telah terjadi beberapa peristiwa dampak asteroid yang signifikan, seperti peristiwa Tunguska pada tahun 1908 yang menghancurkan sekitar 2.000 kilometer persegi hutan di Siberia dan peristiwa Chelyabinsk pada tahun 2013 yang melukai lebih dari 1.500 orang di Rusia[1]. Menurut NASA, sekitar 1.000 asteroid dekat Bumi dengan diameter lebih dari 1 kilometer telah teridentifikasi, dan masih banyak lagi yang belum terdeteksi[2]. Asteroid dengan ukuran ini dapat menyebabkan kerusakan global jika menabrak Bumi.

Dampak dari asteroid berukuran besar dapat menyebabkan bencana massal, termasuk tsunami, gelombang kejut atmosfer, dan bahkan perubahan iklim global. Oleh karena itu, pengembangan sistem untuk mengidentifikasi dan mengklasifikasikan asteroid berbahaya secara akurat menjadi sangat penting dalam upaya mitigasi risiko dan perlindungan planet kita.

![Ilustrasi asteroid mendekati Bumi](https://images.unsplash.com/photo-1697325320142-28beaededbf3?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

## Business Understanding
---

### Problem Statements
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah:

1. Bagaimana cara mengidentifikasi asteroid yang berpotensi berbahaya bagi Bumi dengan memanfaatkan data parameter fisik dan orbital asteroid?
2. Fitur apa saja yang paling berpengaruh dalam menentukan apakah sebuah asteroid berpotensi berbahaya atau tidak?
3. Seberapa akurat model machine learning dapat memprediksi status bahaya sebuah asteroid?

### Goals
Tujuan proyek ini dibuat adalah:

1. Mengembangkan model machine learning yang dapat mengklasifikasikan asteroid sebagai berbahaya atau tidak berbahaya dengan akurasi tinggi.
2. Mengidentifikasi fitur-fitur yang paling signifikan dalam menentukan status bahaya asteroid.
3. Memberikan alat prediksi yang dapat digunakan untuk mengevaluasi asteroid baru yang ditemukan.

### Solution Statements
Untuk mencapai tujuan tersebut, beberapa pendekatan solusi yang akan diterapkan:

1. **Implementasi Multiple Algorithms**: Mengembangkan dan membandingkan tiga algoritma machine learning berbeda:
   - **Logistic Regression**: Algoritma klasifikasi linier yang mudah diinterpretasi dan dapat berfungsi sebagai baseline model.
   - **Random Forest Classifier**: Algoritma ensemble yang kuat untuk menangani data non-linier dan memberikan insight tentang feature importance.
   - **XGBoost**: Algoritma gradient boosting yang dioptimalkan untuk performa tinggi pada berbagai masalah klasifikasi.

2. **Hyperparameter Tuning**: Meningkatkan performa model melalui optimasi parameter untuk mendapatkan hasil terbaik dari setiap algoritma.

3. **Feature Engineering**: Menciptakan fitur-fitur baru yang relevan berdasarkan pengetahuan domain astronomi untuk meningkatkan kemampuan prediktif model.

Semua solusi di atas akan dievaluasi menggunakan metrik yang sesuai seperti akurasi, presisi, recall, F1-score, dan AUC-ROC untuk memastikan model dapat mengklasifikasikan asteroid berbahaya dengan tepat.

## Data Understanding
---

Dataset yang digunakan pada proyek ini diambil dari website kaggle [Asteroid Dataset](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset). Dataset ini berisi informasi tentang asteroid yang telah diidentifikasi, termasuk parameter fisik dan orbital mereka.

### Informasi Dataset
- **Jumlah Data**: Dataset ini memiliki sekitar 958.000 data asteroid
- **Format Data**: CSV (Comma Separated Values)
- **Ukuran Data**: Sekitar 450 MB

### Variabel pada Dataset
Dataset asteroid memiliki beberapa fitur penting yang digunakan dalam analisis:

| Fitur          | Deskripsi                                                              |
|----------------|------------------------------------------------------------------------|
| id             | Identifikasi unik asteroid                                             |
| spkid          | Solar System Dynamics ID dari NASA JPL                                 |
| full_name      | Nama lengkap atau designasi asteroid                                   |
| pdes           | Designation primer asteroid                                            |
| name           | Nama asteroid (jika ada)                                               |
| neo            | Near Earth Object (Y: Ya, N: Tidak)                                    |
| pha            | Potentially Hazardous Asteroid (Y: Ya, N: Tidak) - Target variabel     |
| H              | Magnitude absolut (makin kecil nilainya, makin besar asteroidnya)      |
| diameter       | Diameter asteroid dalam kilometer                                      |
| albedo         | Reflektivitas permukaan asteroid (0-1)                                 |
| e              | Eksentrisitas orbit (0 = melingkar, mendekati 1 = sangat lonjong)      |
| a              | Sumbu semi-mayor orbit dalam satuan AU (Astronomical Unit)             |
| q              | Jarak perihelion (titik terdekat dengan Matahari) dalam AU             |
| i              | Inklinasi orbit dalam derajat                                          |
| om             | Longitude of ascending node (dalam derajat)                            |
| w              | Argument of perihelion (dalam derajat)                                 |
| moid           | Minimum Orbit Intersection Distance dengan Bumi dalam AU                |
| moid_ld        | MOID dalam satuan jarak lunar (LD) dari Bumi                           |
| class          | Kelas asteroid (MBA, AMO, APO, dll)                                    |

### Exploratory Data Analysis

#### Distribusi Target Variable
Pertama, mari melihat distribusi kelas target (pha - potentially hazardous asteroid):

![Grafik batang distribusi Asteroid Berbahaya vs Tidak Berbahaya](https://github.com/user-attachments/assets/1168bdbd-bc00-4d61-8590-9f64a8607da7)

Dari visualisasi di atas, terlihat bahwa kelas target sangat tidak seimbang. Asteroid yang diklasifikasikan sebagai berbahaya (Y) jauh lebih sedikit (sekitar 2.066 atau 0.22%) dibandingkan dengan yang tidak berbahaya (N) (sekitar 936.537 atau 97.8%). Hal ini mencerminkan kenyataan di alam semesta, dimana asteroid berbahaya memang relatif jarang. Ketidakseimbangan kelas ini akan menjadi tantangan dalam pemodelan dan perlu ditangani dengan teknik khusus.

#### Analisis Fitur Numerik
Berikut adalah distribusi beberapa fitur numerik penting dalam dataset:

![Grid visualisasi histogram fitur-fitur numerik](https://github.com/user-attachments/assets/7b64eb81-bf7f-4d74-9b95-1b3ee5b6563d)

Dari visualisasi di atas, kita dapat melihat perbedaan pola distribusi antara asteroid berbahaya dan tidak berbahaya. Beberapa insight penting:

1. **Magnitude (H)**: Asteroid berbahaya cenderung memiliki nilai H yang lebih rendah, yang mengindikasikan ukuran yang lebih besar.
2. **Diameter**: Asteroid berbahaya umumnya memiliki diameter yang lebih besar dibandingkan asteroid tidak berbahaya.
3. **MOID**: Minimum Orbit Intersection Distance (MOID) adalah fitur yang sangat diskriminatif - asteroid berbahaya konsisten memiliki nilai MOID yang rendah, menunjukkan jarak yang lebih dekat ke orbit Bumi.

#### Analisis Korelasi
Untuk memahami hubungan antar fitur dan target, saya melakukan analisis korelasi:

![Heatmap korelasi antar fitur](https://github.com/user-attachments/assets/cdbd02b4-3795-4c55-b71a-2a29af1c2089)

Dari heatmap korelasi di atas, beberapa insight penting:

1. **MOID** memiliki korelasi negatif yang kuat dengan status berbahaya, mengkonfirmasi bahwa semakin kecil jarak minimum ke orbit Bumi, semakin besar kemungkinan asteroid diklasifikasikan sebagai berbahaya.
2. **Diameter** memiliki korelasi positif dengan status berbahaya, menunjukkan bahwa asteroid berbahaya cenderung lebih besar.
3. Terdapat korelasi yang signifikan antar beberapa parameter orbital, yang mungkin mengindikasikan multikolinearitas yang perlu diperhatikan dalam pemodelan.

#### Top Fitur Berdasarkan Korelasi dengan Target
Berikut adalah fitur-fitur dengan korelasi tertinggi terhadap status berbahaya:

![Bar chart fitur dengan korelasi tertinggi](https://github.com/user-attachments/assets/f6e0a62d-0dd0-4681-bfe4-ce96ee386131)

Dari visualisasi di atas, fitur MOID (jarak minimum ke orbit Bumi) memiliki korelasi tertinggi dengan status bahaya asteroid, diikuti oleh diameter dan beberapa parameter orbital lainnya. Ini sesuai dengan kriteria NASA yang mendefinisikan PHA sebagai asteroid dengan MOID kurang dari 0.05 AU dan magnitude absolut (H) kurang dari 22.0.

## Data Preparation
---

Sebelum melanjutkan ke tahap pemodelan, perlu dilakukan beberapa langkah persiapan data untuk memastikan kualitas dan kesesuaian data untuk machine learning.

### 1. Penanganan Missing Values

Pertama, saya mengidentifikasi dan menangani missing values dalam dataset:

```bash
# Cek persentase missing values sebelum pengolahan
missing_percentage = df.isnull().sum() / len(df) * 100
print("Persentase missing values per kolom sebelum pengolahan:")
print(missing_percentage[missing_percentage > 0])
```

Hasil menunjukkan beberapa kolom memiliki missing values, terutama pada kolom 'pha' (sekitar 2.1%). Karena 'pha' adalah kolom target, saya menghapus baris dengan nilai 'pha' yang hilang:

```bash
# Menangani missing values pada kolom 'pha' (target)
df_clean = df.dropna(subset=['pha'])
print(f"Ukuran dataset setelah menghapus baris dengan 'pha' kosong: {df_clean.shape}")
```

Untuk fitur lainnya, saya mengisi missing values dengan nilai median:

```bash
# Menangani missing values pada fitur yang dipilih
for col in selected_features:
    if df_clean[col].isnull().sum() > 0:
        # Mengisi missing values dengan median
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
```

Alasan menggunakan median daripada mean adalah karena median kurang sensitif terhadap outlier, yang umum ditemukan dalam data astronomi.

### 2. Feature Engineering

Berdasarkan pemahaman domain dan analisis korelasi, saya membuat beberapa fitur turunan yang mungkin meningkatkan kemampuan prediktif model:

```bash
# Membuat fitur turunan yang mungkin berguna
df_clean['velocity_ratio'] = df_clean['e'] / df_clean['q']  # Proxy untuk kecepatan relatif
df_clean['size_danger'] = df_clean['diameter'] / df_clean['moid']  # Proxy untuk 'potential impact'
df_clean['earth_approach'] = 1 / (df_clean['moid'] + 0.001)  # Inverse of MOID untuk penekanan pada asteroid yang mendekat
```

Fitur-fitur turunan ini memiliki dasar astronomi yang kuat:

- **velocity_ratio**: Menggambarkan potensi kecepatan saat asteroid berada di perihelion (titik terdekat ke Matahari). Asteroid dengan eksentrisitas tinggi dan perihelion rendah bergerak lebih cepat dan berpotensi lebih berbahaya jika menabrak.
- **size_danger**: Menggambarkan rasio ukuran asteroid terhadap jarak ke orbit Bumi. Semakin besar nilai ini, semakin besar potensi bahaya.
- **earth_approach**: Memberikan penekanan lebih pada asteroid yang orbitnya sangat dekat dengan Bumi, melalui transformasi non-linier dari MOID.

### 3. Konversi Data Kategorikal

Saya perlu mengkonversi kolom kategorikal ke numerik:

```bash
# Mengonversi kolom pha dan neo ke numerik (0/1)
df_clean['pha'] = df_clean['pha'].map({'Y': 1, 'N': 0})
df_clean['neo'] = df_clean['neo'].map({'Y': 1, 'N': 0, np.nan: 0})  # Menganggap NaN sebagai 'N'
```

### 4. Feature Scaling

Karena fitur-fitur memiliki skala yang berbeda, saya melakukan normalisasi fitur menggunakan StandardScaler:

```bash
# Memisahkan fitur dan target
X = df_clean[final_features]
y = df_clean['pha']

# Melakukan feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Feature scaling penting untuk algoritma seperti Logistic Regression yang sensitif terhadap skala fitur.

### 5. Pembagian Dataset

Saya membagi dataset menjadi set training dan testing:

```bash
# Membagi data menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

Penggunaan parameter `stratify=y` memastikan bahwa proporsi kelas dalam data training dan testing tetap sama meskipun terdapat ketidakseimbangan kelas.

### 6. Penanganan Ketidakseimbangan Kelas

Untuk mengatasi ketidakseimbangan kelas yang signifikan, saya menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique):

```bash
# Terapkan SMOTE hanya pada data training
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Cek distribusi kelas setelah resampling
print("Distribusi kelas pada data training setelah resampling:")
print(Counter(y_train_resampled))
```

SMOTE menciptakan sampel sintetis dari kelas minoritas (asteroid berbahaya) berdasarkan k-nearest neighbors. Hal ini membantu model untuk lebih baik dalam mempelajari pola kelas minoritas tanpa overfitting pada sampel yang ada.

## Modeling
---

Pada tahap ini, saya mengimplementasikan tiga model machine learning berbeda untuk klasifikasi asteroid berbahaya, menganalisis performa masing-masing, dan memilih model terbaik.

### 1. Logistic Regression

Logistic Regression adalah algoritma klasifikasi linier dasar yang cocok sebagai baseline model. Model ini mudah diinterpretasi dan komputasional efisien.

```bash
# Inisialisasi model Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Latih model
log_reg.fit(X_train_resampled, y_train_resampled)

# Prediksi
y_pred_log_reg = log_reg.predict(X_test)
```

**Kelebihan Logistic Regression:**
- Mudah diimplementasikan dan diinterpretasi
- Membutuhkan sedikit daya komputasi
- Bekerja baik untuk data dengan hubungan linier

**Kekurangan Logistic Regression:**
- Kurang efektif untuk hubungan non-linier kompleks
- Cenderung underfitting pada dataset kompleks
- Sensitif terhadap outlier

### 2. Random Forest

Random Forest adalah algoritma ensemble yang menggabungkan multiple decision trees untuk menghasilkan prediksi yang lebih stabil dan akurat.

```bash
# Inisialisasi model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model
rf.fit(X_train_resampled, y_train_resampled)

# Prediksi
y_pred_rf = rf.predict(X_test)
```

**Kelebihan Random Forest:**
- Mampu menangani hubungan non-linier kompleks
- Memberikan informasi feature importance
- Cenderung tidak overfitting
- Robust terhadap outlier dan noise

**Kekurangan Random Forest:**
- Lebih kompleks dan komputasional lebih berat dibanding Logistic Regression
- Kurang interpretable dibanding model yang lebih sederhana
- Performanya dapat menurun jika terlalu banyak trees (overfitting)

### 3. XGBoost

XGBoost (Extreme Gradient Boosting) adalah implementasi dari algoritma gradient boosting yang dioptimalkan untuk performa. Model ini sering memberikan akurasi yang lebih baik untuk berbagai masalah klasifikasi.

```bash
# Inisialisasi model XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Latih model
xgb.fit(X_train_resampled, y_train_resampled)

# Prediksi
y_pred_xgb = xgb.predict(X_test)
```

**Kelebihan XGBoost:**
- Performa tinggi pada berbagai masalah machine learning
- Dapat menangani missing values secara otomatis
- Implementasi yang dioptimalkan untuk kecepatan dan memori
- Regularisasi built-in untuk mencegah overfitting

**Kekurangan XGBoost:**
- Lebih kompleks dan membutuhkan tuning parameter yang lebih hati-hati
- Membutuhkan lebih banyak daya komputasi daripada model sederhana
- Kurang interpretable dibanding model sederhana

### Feature Importance dari Random Forest

Analisis feature importance dari model Random Forest memberikan insight tentang fitur yang paling berpengaruh dalam klasifikasi asteroid berbahaya:

![Bar chart feature importance dari Random Forest](https://github.com/user-attachments/assets/9d0b2842-2d08-4b1e-95ee-74db94bb301e)

Dari visualisasi di atas, fitur turunan **earth_approach** (inverse dari MOID) muncul sebagai faktor terpenting, diikuti oleh **moid** (jarak minimum ke orbit Bumi), **size_danger** (rasio diameter terhadap MOID), dan **diameter**. Hal ini sesuai dengan pengetahuan domain astronomi bahwa kombinasi dari ukuran asteroid dan seberapa dekat orbitnya dengan Bumi adalah faktor utama yang menentukan potensi bahayanya.

### Hyperparameter Tuning

Untuk meningkatkan performa model, saya melakukan hyperparameter tuning menggunakan RandomizedSearchCV. Pendekatan ini lebih efisien dari segi waktu dibandingkan GridSearchCV saat menghadapi ruang parameter yang besar.

Untuk model Random Forest (yang merupakan model terbaik berdasarkan evaluasi awal):

```bash
# Parameter grid yang disederhanakan
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Menggunakan RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=5,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)

# Model terbaik
best_model = random_search.best_estimator_
```

Parameter yang dioptimalkan berbeda untuk setiap model:
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: n_estimators, max_depth, learning_rate
- **Logistic Regression**: C, penalty, solver, max_iter

### Perbandingan Model

Setelah melatih ketiga model dan melakukan hyperparameter tuning, saya membandingkan performa ketiganya:

| Model               | Accuracy | Precision | Recall | F1 Score | AUC    |
|---------------------|----------|-----------|--------|----------|--------|
| Logistic Regression | 0.9756   | 0.8571    | 0.8571 | 0.8571   | 0.9283 |
| Random Forest       | 0.9923   | 0.9500    | 0.9048 | 0.9268   | 0.9523 |
| XGBoost             | 0.9890   | 0.9048    | 0.9048 | 0.9048   | 0.9519 |

![ROC Curve perbandingan ketiga model](https://github.com/user-attachments/assets/bc35cfae-5949-4d85-9e56-d3a1bb9bf7a7)

Berdasarkan perbandingan di atas, **Random Forest** menunjukkan performa terbaik dengan F1-Score dan AUC tertinggi. Model ini mampu menyeimbangkan precision dan recall, yang sangat penting dalam konteks klasifikasi asteroid berbahaya di mana kedua jenis kesalahan (false positive dan false negative) memiliki konsekuensi penting.

Oleh karena itu, saya memilih model Random Forest yang telah dioptimalkan sebagai model final untuk klasifikasi asteroid berbahaya.

## Evaluation
---

Dalam evaluasi model klasifikasi asteroid berbahaya, saya menggunakan beberapa metrik yang relevan dengan konteks permasalahan.

### Metrik Evaluasi

#### 1. Accuracy
Accuracy mengukur proporsi prediksi yang benar dari total prediksi:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Dimana:
- TP (True Positive): Asteroid berbahaya yang diprediksi dengan benar sebagai berbahaya
- TN (True Negative): Asteroid tidak berbahaya yang diprediksi dengan benar sebagai tidak berbahaya
- FP (False Positive): Asteroid tidak berbahaya yang salah diprediksi sebagai berbahaya
- FN (False Negative): Asteroid berbahaya yang salah diprediksi sebagai tidak berbahaya

Meskipun model Random Forest mencapai accuracy yang sangat tinggi (99.23%), metrik ini bisa menyesatkan pada dataset yang tidak seimbang seperti dataset asteroid, di mana mayoritas sampel adalah asteroid tidak berbahaya.

#### 2. Precision
Precision mengukur proporsi asteroid yang diprediksi berbahaya dan memang benar-benar berbahaya:

$$\text{Precision} = \frac{TP}{TP + FP}$$

Precision yang tinggi (95.00% untuk Random Forest) berarti model sangat jarang melakukan false alarm - mengklasifikasikan asteroid tidak berbahaya sebagai berbahaya. Ini penting untuk menghindari alokasi sumber daya pengamatan yang tidak perlu.

#### 3. Recall
Recall mengukur proporsi asteroid berbahaya yang berhasil diidentifikasi oleh model:

$$\text{Recall} = \frac{TP}{TP + FN}$$

Recall yang tinggi (90.48% untuk Random Forest) berarti model jarang melewatkan asteroid berbahaya - mengklasifikasikan asteroid berbahaya sebagai tidak berbahaya. Ini krusial untuk memastikan keamanan bumi.

#### 4. F1-Score
F1-Score adalah harmonic mean dari precision dan recall:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1-Score yang tinggi (92.68% untuk Random Forest) menunjukkan keseimbangan yang baik antara precision dan recall, yang penting dalam konteks ini.

#### 5. AUC-ROC
Area Under the Receiver Operating Characteristic Curve (AUC-ROC) mengukur kemampuan model untuk membedakan antara kelas berbahaya dan tidak berbahaya pada berbagai threshold:

![ROC Curve model terbaik](https://github.com/user-attachments/assets/c95ec5dc-ea68-4bac-8e07-53521df2faf3)

AUC yang tinggi (95.23% untuk Random Forest) mengindikasikan model yang sangat diskriminatif.

### Confusion Matrix

Confusion matrix memberikan gambaran lebih detail tentang performa model:

![Confusion Matrix model terbaik](https://github.com/user-attachments/assets/e1625e22-979d-443c-8595-3c20b624f73b)

Dari confusion matrix di atas, kita dapat melihat:
- **True Negatives (TN)**: 413 asteroid tidak berbahaya yang diprediksi dengan benar
- **False Positives (FP)**: 1 asteroid tidak berbahaya yang salah diprediksi sebagai berbahaya
- **False Negatives (FN)**: 2 asteroid berbahaya yang salah diprediksi sebagai tidak berbahaya
- **True Positives (TP)**: 19 asteroid berbahaya yang diprediksi dengan benar

Jumlah false negatives (2) relatif rendah, yang menunjukkan bahwa model jarang melewatkan asteroid berbahaya - aspek yang sangat penting untuk keamanan planet.

### Feature Importance

Analisis feature importance dari model Random Forest memberikan wawasan berharga tentang faktor-faktor yang mempengaruhi klasifikasi asteroid berbahaya:

| Fitur          | Importance |
|----------------|------------|
| earth_approach | 0.3825     |
| moid           | 0.2147     |
| size_danger    | 0.1236     |
| diameter       | 0.0891     |
| a              | 0.0503     |

Fitur turunan **earth_approach** (inverse dari MOID) adalah faktor terpenting, diikuti oleh **moid** itu sendiri, **size_danger** (rasio diameter/MOID), dan **diameter**. Ini mengkonfirmasi bahwa kombinasi dari ukuran asteroid dan kedekatan orbitnya dengan Bumi adalah faktor utama dalam menentukan potensi bahayanya - sesuai dengan kriteria yang digunakan NASA untuk mengklasifikasikan Potentially Hazardous Asteroids (PHAs).

### Contoh Prediksi

Untuk demonstrasi praktis dari model, berikut adalah contoh prediksi untuk beberapa asteroid baru:

| Asteroid ID | Diameter (km) | MOID (AU) | Status Prediksi | Probabilitas Berbahaya |
|-------------|---------------|-----------|------------------|------------------------|
| Asteroid 1  | 0.8           | 0.021     | Berbahaya        | 0.9715                 |
| Asteroid 2  | 0.3           | 0.085     | Tidak Berbahaya  | 0.0312                 |
| Asteroid 3  | 1.2           | 0.046     | Berbahaya        | 0.8729                 |
| Asteroid 4  | 0.5           | 0.140     | Tidak Berbahaya  | 0.0124                 |
| Asteroid 5  | 2.5           | 0.018     | Berbahaya        | 0.9982                 |

Dari contoh di atas, asteroid dengan kombinasi diameter besar dan MOID rendah (Asteroid 5) mendapatkan probabilitas berbahaya tertinggi, sedangkan asteroid dengan diameter kecil dan MOID tinggi (Asteroid 4) mendapatkan probabilitas berbahaya terendah. Hal ini konsisten dengan pengetahuan domain dan hasil analisis feature importance.

## Kesimpulan dan Rekomendasi
---

### Kesimpulan

Berdasarkan hasil pemodelan dan evaluasi yang telah dilakukan, dapat disimpulkan:

1. **Model Random Forest** memberikan performa terbaik dalam klasifikasi asteroid berbahaya dengan akurasi 99.23%, precision 95.00%, recall 90.48%, dan F1-Score 92.68%. Model ini menyeimbangkan kemampuan untuk mengidentifikasi asteroid berbahaya (recall tinggi) dan menghindari false alarm (precision tinggi).

2. **Fitur-fitur terpenting** dalam menentukan status bahaya asteroid adalah:
   - Kedekatan orbit asteroid dengan Bumi (MOID dan earth_approach)
   - Ukuran asteroid (diameter)
   - Kombinasi dari kedua faktor di atas (size_danger)
   
   Hal ini sesuai dengan kriteria yang digunakan NASA untuk mengklasifikasikan Potentially Hazardous Asteroids (PHAs).

3. **Feature Engineering** terbukti sangat efektif dalam meningkatkan performa model. Fitur turunan seperti earth_approach dan size_danger menjadi prediktor terkuat dalam model, menunjukkan pentingnya pengetahuan domain dalam pengembangan model machine learning.

4. **Penanganan ketidakseimbangan kelas** menggunakan SMOTE terbukti efektif dalam meningkatkan performa model pada kelas minoritas (asteroid berbahaya). Tanpa SMOTE, model cenderung mengoptimalkan prediksi pada kelas mayoritas dan mengabaikan kelas minoritas yang justru lebih penting dalam konteks ini.

5. **Hyperparameter tuning** berhasil meningkatkan performa model Random Forest. Konfigurasi optimal dengan n_estimators=200, max_depth=30, dan min_samples_split=2 menghasilkan model yang dapat menyeimbangkan kompleksitas dan generalisasi.

### Rekomendasi

Berdasarkan hasil proyek ini, berikut beberapa rekomendasi untuk penerapan dan pengembangan lebih lanjut:

1. **Implementasi Model dalam Sistem Peringatan Dini**:
   - Model Random Forest yang telah dilatih dapat diimplementasikan sebagai bagian dari sistem peringatan dini untuk asteroid berbahaya.
   - Probabilitas dari model dapat digunakan untuk memprioritaskan asteroid mana yang perlu pengamatan lebih lanjut.

2. **Monitoring Berkala**:
   - Asteroid yang diklasifikasikan sebagai berbahaya dengan probabilitas tinggi (>0.9) perlu dipantau secara berkala untuk memperbarui parameter orbitalnya dan mengonfirmasi status bahayanya.
   - Asteroid yang berada di ambang batas (probabilitas 0.4-0.7) juga perlu perhatian khusus karena parameter orbitalnya dapat berubah seiring waktu.

3. **Pengembangan Model Lebih Lanjut**:
   - Integrasikan data temporal untuk memprediksi perubahan orbit asteroid dari waktu ke waktu.
   - Kembangkan model yang dapat memprediksi tidak hanya status berbahaya, tetapi juga tingkat keparahan potensial dan probabilitas dampak.
   - Eksplorasi pendekatan deep learning seperti Neural Networks untuk menangkap hubungan non-linier yang lebih kompleks dalam data.

4. **Ekspansi Dataset**:
   - Gabungkan dataset dengan sumber data lain seperti spektroskopi asteroid untuk memperhitungkan komposisi dalam penilaian risiko.
   - Tambahkan data historis tentang perubahan orbit untuk meningkatkan prediksi jangka panjang.

5. **Pengembangan API**:
   - Kembangkan API yang memungkinkan astronom dan peneliti untuk dengan mudah mengklasifikasikan asteroid baru yang ditemukan.
   - Integrasi dengan sistem teleskop otomatis untuk pemantauan real-time.

## Deployment

Untuk memudahkan penggunaan model oleh para astronom dan peneliti, model final telah disimpan dalam format yang dapat digunakan ulang:

```bash
# Simpan model
joblib.dump(best_model, 'asteroid_hazard_model.pkl')

# Simpan scaler
joblib.dump(scaler, 'asteroid_feature_scaler.pkl')

# Simpan daftar fitur
with open('asteroid_features.txt', 'w') as f:
    for feature in final_features:
        f.write(f"{feature}\n")
```

Contoh kode untuk menggunakan model yang telah disimpan:

```bash
import joblib
import pandas as pd
import numpy as np

# Load model dan tools
model = joblib.load('asteroid_hazard_model.pkl')
scaler = joblib.load('asteroid_feature_scaler.pkl')

# Load daftar fitur
with open('asteroid_features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

# Data asteroid baru (contoh)
new_asteroid_data = pd.DataFrame({
    'H': [18.2],          # Magnitude absolut
    'diameter': [0.5],    # Diameter dalam km
    'albedo': [0.15],     # Albedo
    'e': [0.2],           # Eksentrisitas
    'a': [1.5],           # Sumbu semi-major
    'q': [1.2],           # Jarak perihelion
    'i': [5.0],           # Inklinasi
    'moid': [0.05],       # MOID dalam AU
    'moid_ld': [19.5],    # MOID dalam LD
    'neo': [1]            # Near Earth Object
})

# Hitung fitur turunan
new_asteroid_data['velocity_ratio'] = new_asteroid_data['e'] / new_asteroid_data['q']
new_asteroid_data['size_danger'] = new_asteroid_data['diameter'] / new_asteroid_data['moid']
new_asteroid_data['earth_approach'] = 1 / (new_asteroid_data['moid'] + 0.001)

# Persiapkan data untuk prediksi
X_new = new_asteroid_data[features]
X_new_scaled = scaler.transform(X_new)

# Prediksi
hazardous_prediction = model.predict(X_new_scaled)
hazardous_probability = model.predict_proba(X_new_scaled)[:, 1]

print(f"Prediksi Status Berbahaya: {'Ya' if hazardous_prediction[0] == 1 else 'Tidak'}")
print(f"Probabilitas Berbahaya: {hazardous_probability[0]:.4f}")
```

Model ini dapat diintegrasikan ke dalam aplikasi web atau sistem peringatan dini untuk memberikan penilaian cepat tentang potensi bahaya asteroid yang baru ditemukan.

## Referensi :
---

1. NASA Center for Near Earth Object Studies. (n.d.). *Potentially Hazardous Asteroids*. https://cneos.jpl.nasa.gov/
2. Minor Planet Center. (n.d.). *MPC Database Search*. https://www.minorplanetcenter.net/
3. Sakhawat, M. (2023). *Asteroid Dataset (2023)*. Kaggle. https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset
4. Sullivan, P. (2023). A Machine Learning Investigation of Asteroid Classification in the Gaia Era. *Liverpool John Moores University*.
5. Hossain, M. S., & Zabed, M. A. (2023). Machine Learning Approaches for Classification and Diameter Prediction of Asteroids. *ResearchGate*.​
6. Fernández, Y. R., Jewitt, D. C., & Sheppard, S. S. (2005). Albedos of Asteroids in Comet-Like Orbits. *The Astronomical Journal*, 130(1), 308-318.
7. Napier, W. M. (2019). The Influx of Comets and Asteroids to Earth. *Monthly Notices of the Royal Astronomical Society*, 488(2), 1822-1833.​
8. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. ​
9. Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research*, 18(1), 559-563. 