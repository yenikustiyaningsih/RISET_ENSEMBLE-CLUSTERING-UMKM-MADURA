# Streamlit — Ensemble Clustering UMKM (Unweighted)

Aplikasi berbasis **Streamlit** untuk implementasi **Ensemble Clustering Unweighted** pada data UMKM. Aplikasi ini merupakan implementasi dari notebook penelitian untuk melakukan clustering menggunakan beberapa algoritma dan menggabungkan hasil clustering melalui **Majority Voting**.

## Tujuan

Aplikasi ini digunakan untuk melakukan segmentasi UMKM berdasarkan karakteristik usaha menggunakan beberapa metode clustering, kemudian menggabungkan hasil clustering tersebut menjadi satu hasil konsensus.

## Pipeline

```text
Dataset UMKM
    ↓
Preprocessing
    ↓
K-Means
K-Prototypes
Gower K-Medoids
    ↓
Hungarian Label Alignment
    ↓
Majority Voting (Equal Vote)
    ↓
Multi-Index Validation
    ↓
K Final
    ↓
Profil & Visualisasi Cluster
```

## Metode yang Digunakan

- 8 indikator UMKM
- K-Means
- K-Prototypes
- Gower K-Medoids
- Hungarian Label Alignment
- Majority Voting dengan Equal Vote
- Silhouette Coefficient
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Filter ukuran cluster minimum
- Multi-Index Validation / Mean Rank

## Fitur Aplikasi

- Upload dataset UMKM dalam format Excel
- Preprocessing data numerik dan kategorikal
- Clustering menggunakan K-Means
- Clustering menggunakan K-Prototypes
- Clustering menggunakan Gower K-Medoids
- Penyamaan label cluster menggunakan Hungarian Label Alignment
- Penggabungan hasil clustering menggunakan Majority Voting
- Evaluasi kualitas cluster menggunakan beberapa indeks
- Pemilihan K final berdasarkan Multi-Index Validation / Mean Rank
- Visualisasi hasil clustering
- Profiling karakteristik setiap cluster
- Download hasil analisis dalam format Excel

## Struktur

```text
streamlit_ensemble_unweighted/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── clustering.py
    ├── ensemble.py
    └── visualization.py
```

## Teknologi

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- K-Modes / K-Prototypes
- Gower Distance
- K-Medoids

## Setup Windows PowerShell

Clone repository:

```powershell
git clone https://github.com/yenikustiyaningsih/RISET_ENSEMBLE-CLUSTERING-UMKM-MADURA.git
cd RISET_ENSEMBLE-CLUSTERING-UMKM-MADURA
```

Buat virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependency:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jalankan aplikasi:

```powershell
streamlit run app.py
```

Jika PowerShell memblokir aktivasi:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset

Dataset diunggah melalui sidebar aplikasi dalam format Excel.

Untuk data mentah, kolom yang harus tersedia:

### Numerik

- `Jml. Tenaga Kerja`
- `Kapasitas Produksi/Thn`
- `Omset/Thn`
- `Aset`

### Kategorikal

- `NIB/SKU`
- `Sosmed`
- `Marketplace`
- `Kepemilikan Lahan`

Aplikasi juga mengenali workbook preprocessing yang memiliki sheet `MinMax_2896` dengan nama kolom seperti pada notebook penelitian.

## Preprocessing

Tahapan preprocessing yang digunakan meliputi:

- IQR clipping pada fitur numerik.
- Normalisasi fitur numerik menggunakan MinMax Scaling.
- Transformasi `NIB/SKU` menjadi fitur biner.
- Transformasi `Kepemilikan Lahan` menjadi fitur biner.
- Transformasi `Sosmed` dan `Marketplace` menjadi proporsi jumlah channel.

## Catatan Kesesuaian dengan Notebook

Implementasi utama mengikuti notebook penelitian:

- K-Means menggunakan representasi 8 fitur baseline.
- Numerik baseline menggunakan IQR clipping + MinMax [0,1].
- NIB/SKU dan Kepemilikan Lahan menjadi biner.
- Sosmed dan Marketplace menjadi proporsi jumlah channel.
- K-Prototypes menggunakan StandardScaler pada empat fitur numerik dan `gamma = 0.150004`.
- Gower K-Medoids menggunakan matriks Gower dan 10 inisialisasi `fasterpam`.
- Hungarian menggunakan K-Means sebagai reference.
- Majority Voting menggunakan equal vote dan K-Means sebagai tie-break.
- Evaluasi konsensus menggunakan Gower untuk Silhouette Coefficient serta ruang evaluasi gabungan untuk Davies-Bouldin Index dan Calinski-Harabasz Index.
- K final dipilih menggunakan Mean Rank setelah filter ukuran cluster minimum `max(20, 1% × N)`.

## Evaluasi Clustering

### Silhouette Coefficient

Mengukur seberapa baik suatu objek berada di dalam clusternya dibandingkan dengan cluster lainnya. Nilai yang lebih tinggi menunjukkan kualitas clustering yang lebih baik.

### Davies-Bouldin Index

Mengukur kemiripan antar-cluster berdasarkan jarak dan penyebaran cluster. Nilai yang lebih rendah menunjukkan kualitas clustering yang lebih baik.

### Calinski-Harabasz Index

Mengukur rasio dispersi antar-cluster terhadap dispersi dalam cluster. Nilai yang lebih tinggi menunjukkan kualitas clustering yang lebih baik.

### Multi-Index Validation

Hasil dari beberapa indeks evaluasi digunakan secara bersama-sama untuk menentukan konfigurasi clustering terbaik berdasarkan **Mean Rank**.

## Output

Aplikasi dapat mengunduh hasil analisis dalam file:

```text
hasil_ensemble_unweighted.xlsx
```

File hasil memiliki beberapa sheet:

- `Ensemble_All_K`
- `Evaluasi_Ensemble`
- `Multi_Index_Ranking`
- `Perbandingan_Final`
- `Distribusi_Cluster`

## Deployment

Aplikasi dapat dijalankan secara online menggunakan **Streamlit Community Cloud** dengan menghubungkan repository GitHub ini.

Main file:

```text
app.py
```

Branch:

```text
main
```

## Repository

[GitHub Repository](https://github.com/yenikustiyaningsih/RISET_ENSEMBLE-CLUSTERING-UMKM-MADURA)

## Author

**Yeni Kustiyaningsih**

Ensemble Clustering UMKM — Unweighted Scenario
