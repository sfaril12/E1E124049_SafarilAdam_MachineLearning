#  clustering wilayah berdasarkan risiko penyakit menggunakan K-means dan PCA untuk mendukung SDGS 3 ( Good health and well-being)

> Unsupervised Machine Learning untuk Identifikasi Profil Kesehatan Global

**Safaril Adam** — E1E124049

---

## Deskripsi Proyek

Proyek ini menggunakan teknik *unsupervised learning* untuk mengelompokkan negara-negara di dunia berdasarkan indikator kesehatan, dalam kerangka **Sustainable Development Goals 3 (SDGs 3)** — memastikan kehidupan yang sehat dan mendukung kesejahteraan bagi semua usia.

Dengan menganalisis data dari **193 negara** selama periode **2000–2015**, proyek ini mengidentifikasi klaster risiko kesehatan yang dapat digunakan untuk:
-  Menentukan prioritas distribusi bantuan kesehatan global
-  Mengidentifikasi negara yang membutuhkan intervensi mendesak
-  Membandingkan efektivitas algoritma clustering (*K-Means, Hierarchical, DBSCAN*)

---

## Struktur File

```
Machine Learning/
├── Clustering_Safaril_Adam_E1E124049l.ipynb   # Notebook utama (analisis lengkap)
├── Life Expectancy Data.csv                    # Dataset WHO
├── dokumentasi_clustering.tex                  # Dokumentasi LaTeX
├── clustering_model.pkl                        # Model tersimpan (output notebook)
├── clustering_results.csv                      # Hasil clustering (output notebook)
└── README.md                                   # File ini
```

---

## Dataset

| Atribut | Detail |
|---------|--------|
| **Sumber** | [Life Expectancy (WHO)](https://raw.githubusercontent.com/sfaril12/dataset-machine-learning/refs/heads/main/Life%20Expectancy%20Data.csv) |
| **Volume** | 2.938 observasi |
| **Kolom** | 22 fitur |
| **Cakupan** | 193 negara |
| **Rentang** | 2000 – 2015 |

**14 fitur** dipilih untuk analisis, mencakup: Life Expectancy, Adult Mortality, Infant Deaths, Under-5 Deaths, HIV/AIDS, Measles, BMI, Polio, Diphtheria, Hepatitis B, Thinness 1-19y, GDP, Income Composition, dan Schooling.

---

## Pipeline Preprocessing

```
Dataset Awal (2.938 baris, 193 negara)
    │
    ▼
Feature Selection (14 fitur)
    │
    ▼
Agregasi per Negara (mean 16 tahun) → 147 negara
    │
    ▼
Log1p Transform (6 fitur skewed: Infant Deaths, Under-5 Deaths,
                  HIV/AIDS, Measles, Thinness 1-19y, GDP)
    │
    ▼
IQR Outlier Removal (3x Tukey Fence) → 141 negara
    │
    ▼
RobustScaler (standardisasi)
    │
    ▼
PCA (14 → 2 komponen, 72.81% variance)
    │
    ▼
Clustering (K-Means / Hierarchical / DBSCAN)
```

---

## Hasil Utama

### Performa Algoritma

| Algoritma | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|-----------|:---:|:---:|:---:|
| **K-Means** | **0.6316** ✅ | **174.55** ✅ | **0.4802** ✅ |
| Hierarchical | 0.5813 | 149.73 | 0.5340 |
| DBSCAN | 0.5742 | — | — |

### Cluster Optimal: k = 2

- **Risiko Rendah** — Life Expectancy tinggi, cakupan vaksinasi tinggi, GDP tinggi
- **Risiko Tinggi** — Life Expectancy rendah, HIV/AIDS tinggi, cakupan vaksinasi rendah

### Konsistensi Model (Data Splitting Test)

| Split | Train Silhouette | Test Silhouette | Gap |
|:---:|:---:|:---:|:---:|
| 70/30 | ~0.63 | ~0.62 | ~0.01 |
| 80/20 | ~0.63 | ~0.62 | ~0.01 |
| 90/10 | ~0.63 | ~0.63 | ~0.00 |

> Gap yang sangat kecil menunjukkan model **stabil dan tidak overfitting**.

---

## Perbandingan Cluster

Berdasarkan model K-Means dengan k=2, negara-negara dikelompokkan menjadi dua cluster berdasarkan profil kesehatannya:

### Cluster 0: Risiko Rendah

Cluster ini berisi negara-negara dengan profil kesehatan **baik**, umumnya negara-negara maju dan berkembang menengah ke atas:

| Indikator | Karakteristik |
|-----------|--------------|
| Life Expectancy | **Tinggi** (rata-rata >70 tahun) |
| Adult Mortality | **Rendah** — probabilitas kematian usia produktif rendah |
| Infant & Under-5 Deaths | **Rendah** — sistem kesehatan ibu-anak memadai |
| HIV/AIDS | **Rendah** — prevalensi terkontrol |
| Cakupan Vaksinasi (Polio, Diphtheria, Hep B) | **Tinggi** (>90%) — program imunisasi kuat |
| BMI | **Tinggi** — nutrisi tercukupi |
| GDP | **Tinggi** — kapasitas ekonomi mendukung sistem kesehatan |
| Income Composition | **Tinggi** — pembangunan manusia baik |
| Schooling | **Tinggi** — akses pendidikan luas |
| Thinness 1-19y | **Rendah** — malnutrisi minim |

> Contoh negara: Jepang, Norwegia, Australia, Jerman, Korea Selatan

### Cluster 1: Risiko Tinggi

Cluster ini berisi negara-negara dengan profil kesehatan **buruk**, umumnya negara-negara berkembang dengan tantangan struktural:

| Indikator | Karakteristik |
|-----------|--------------|
| Life Expectancy | **Rendah** (rata-rata <65 tahun) |
| Adult Mortality | **Tinggi** — kematian usia produktif signifikan |
| Infant & Under-5 Deaths | **Tinggi** — mortalitas bayi dan balita masih menjadi masalah |
| HIV/AIDS | **Tinggi** — epidemi belum terkontrol |
| Cakupan Vaksinasi (Polio, Diphtheria, Hep B) | **Rendah** (<80%) — program imunisasi belum merata |
| BMI | **Rendah** — indikasi malnutrisi |
| GDP | **Rendah** — keterbatasan ekonomi membatasi layanan kesehatan |
| Income Composition | **Rendah** — pembangunan manusia tertinggal |
| Schooling | **Rendah** — akses pendidikan terbatas |
| Thinness 1-19y | **Tinggi** — prevalensi kekurusan pada anak dan remaja |

> Contoh negara: Nigeria, Chad, Afghanistan, Sierra Leone, Myanmar

### Perbedaan Kunci Antar Cluster

Faktor yang paling membedakan kedua cluster (berdasarkan kontribusi PCA):

1. **Life Expectancy** — perbedaan rata-rata ~15-20 tahun antar cluster
2. **Income Composition** — indikator pembangunan manusia paling kontras
3. **Schooling** — akses pendidikan berbeda sangat signifikan
4. **HIV/AIDS** — prevalensi jauh lebih tinggi di cluster Risiko Tinggi
5. **Cakupan Vaksinasi** — gap besar pada Polio, Diphtheria, dan Hepatitis B

---

## Hubungan dengan SDGs 3

| Target SDGs | Temuan |
|-------------|--------|
| SDGs 3.2 | Kematian bayi & balita berbeda signifikan antar cluster |
| SDGs 3.3 | HIV/AIDS lebih tinggi di cluster Risiko Tinggi |
| SDGs 3.4 | Life Expectancy berbeda signifikan antar cluster |
| SDGs 3.8 | Cakupan vaksinasi lebih rendah di cluster Risiko Tinggi |

---

## Keterbatasan

1. **Dimensi data terbatas** — Hanya 2 komponen PCA (72.81% varians) dipertahankan
2. **Temporal limitation** — Data diagregasi menjadi rata-rata 16 tahun
3. **Data loss** — 52 negara hilang karena *listwise deletion*
4. **Outlier removal** — 6 negara HIV/AIDS parah dihapus (Botswana, Lesotho, Malawi, South Africa, Swaziland, Zimbabwe)


---

## Library yang digunakan

- **Python 3.x**
- `pandas` — manipulasi data
- `numpy` — komputasi numerik
- `scikit-learn` — preprocessing, PCA, clustering, evaluasi
- `matplotlib` & `seaborn` — visualisasi
- `scipy` — analisis statistik & dendrogram
- `pickle` — serialisasi model

---

## Dokumentasi Lengkap

Dokumentasi teknis lengkap tersedia dalam format:
- **Notebook** → `Clustering_Safaril_Adam_E1E124049l.ipynb` (seluruh kode + hasil)

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik.

---

*Dibuat berdasarkan notebook `Clustering_Safaril_Adam_E1E124049l.ipynb`*
