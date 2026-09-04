# Streamlit — Ensemble Clustering UMKM  

Aplikasi ini merupakan implementasi **skenario Ensemble Clustering Unweighted** dari notebook penelitian.

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

### Yang digunakan

- 8 indikator UMKM
- K-Means
- K-Prototypes
- Gower K-Medoids
- Hungarian Label Alignment
- Majority Voting Equal Vote
- Silhouette Coefficient
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Filter ukuran cluster minimum
- Multi-Index Validation / Mean Rank

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

## Setup Windows PowerShell

```powershell
cd streamlit_ensemble_unweighted

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Jika PowerShell memblokir aktivasi:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset

Upload file Excel melalui sidebar.

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

Aplikasi juga mengenali workbook preprocessing yang memiliki sheet `MinMax_2896` dengan nama kolom seperti pada notebook.   

## Catatan Kesesuaian dengan Notebook

Implementasi utama mengikuti notebook:

- K-Means menggunakan representasi 8 fitur baseline.
- Numerik baseline menggunakan IQR clipping + MinMax [0,1].
- NIB/SKU dan Kepemilikan Lahan menjadi biner.
- Sosmed dan Marketplace menjadi proporsi jumlah channel.
- K-Prototypes menggunakan StandardScaler pada empat fitur numerik dan `gamma = 0.150004`.
- Gower K-Medoids menggunakan matriks Gower dan 10 inisialisasi `fasterpam`.
- Hungarian menggunakan K-Means sebagai reference.
- Majority Voting menggunakan equal vote dan K-Means sebagai tie-break.
- Evaluasi konsensus menggunakan Gower untuk SC serta ruang evaluasi gabungan untuk DBI/CHI.
- K final dipilih menggunakan Mean Rank setelah filter ukuran cluster minimum `max(20, 1% × N)`.     

## Output

Aplikasi dapat mengunduh hasil berikut:

`hasil_ensemble_unweighted.xlsx`

dengan sheet:

- `Ensemble_All_K`
- `Evaluasi_Ensemble`
- `Multi_Index_Ranking`
- `Perbandingan_Final`
- `Distribusi_Cluster`
