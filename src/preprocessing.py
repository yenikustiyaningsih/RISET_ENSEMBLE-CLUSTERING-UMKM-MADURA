from io import BytesIO
import numpy as np
import pandas as pd
import gower

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_COLUMNS = [
    "Jml. Tenaga Kerja",
    "Kapasitas Produksi/Thn",
    "Omset/Thn",
    "Aset",
]

CATEGORICAL_COLUMNS = [
    "NIB/SKU",
    "Sosmed",
    "Marketplace",
    "Kepemilikan Lahan",
]

EXPECTED_FEATURE_ORDER = [
    "NIB/SKU",
    "Jml. Tenaga Kerja",
    "Kapasitas Produksi/Thn",
    "Omset/Thn",
    "Aset",
    "Sosmed",
    "Marketplace",
    "Kepemilikan Lahan",
]


def _count_channels(value):
    text = str(value).strip()
    if text == "" or text.lower() in ("tidak ada", "-", "nan", "none"):
        return 0
    return len([item for item in text.split(",") if item.strip()])


def _rename_preprocessed_columns(df):
    return df.rename(columns={
        "NIB_SKU": "NIB/SKU",
        "Jml_Tenaga_Kerja": "Jml. Tenaga Kerja",
        "Kapasitas_Produksi": "Kapasitas Produksi/Thn",
        "Omset_Log1p": "Omset/Thn",
        "Aset_Log1p": "Aset",
        "Kepemilikan_Lahan": "Kepemilikan Lahan",
    })


def build_raw_pipeline(df_raw):
    df_clean = df_raw.drop_duplicates(
        subset=[c for c in df_raw.columns if c != "No."]
    ).reset_index(drop=True)

    identity_candidates = [
        "No.", "Nama Pemilik", "Nama Usaha", "NIK", "Tgl. Lahir",
        "Jalan", "Desa", "Sektor", "Identitas_UMKM"
    ]
    identity_columns = [c for c in identity_candidates if c in df_clean.columns]
    if not identity_columns:
        identity_columns = [df_clean.columns[0]]

    missing = [
        c for c in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
        if c not in df_clean.columns
    ]
    if missing:
        raise KeyError(f"Kolom wajib tidak ditemukan: {missing}")

    typeaware = df_clean[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS].copy()

    for col in NUMERICAL_COLUMNS:
        typeaware[col] = (
            typeaware[col].astype(str)
            .str.replace("Rp", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        typeaware[col] = pd.to_numeric(typeaware[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        typeaware[col] = (
            typeaware[col].astype(str).str.strip().str.title()
        )

    if typeaware[NUMERICAL_COLUMNS].isnull().any().any():
        bad = typeaware[NUMERICAL_COLUMNS].isnull().sum()
        raise ValueError(f"Ada nilai numerik yang gagal dikonversi: {bad[bad > 0].to_dict()}")

    baseline = pd.DataFrame(index=df_clean.index)
    baseline["NIB/SKU"] = (
        typeaware["NIB/SKU"].astype(str).str.strip().str.title().eq("Ada").astype(int)
    )

    for col in NUMERICAL_COLUMNS:
        x = typeaware[col].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        x = x.clip(lower, upper)
        x_min, x_max = x.min(), x.max()
        baseline[col] = 0.0 if x_max == x_min else (x - x_min) / (x_max - x_min)

    sosmed_count = typeaware["Sosmed"].apply(_count_channels)
    marketplace_count = typeaware["Marketplace"].apply(_count_channels)

    baseline["Sosmed"] = (
        0.0 if sosmed_count.max() == 0 else sosmed_count / sosmed_count.max()
    )
    baseline["Marketplace"] = (
        0.0 if marketplace_count.max() == 0
        else marketplace_count / marketplace_count.max()
    )
    baseline["Kepemilikan Lahan"] = (
        typeaware["Kepemilikan Lahan"].astype(str)
        .str.strip().str.title().eq("Sendiri").astype(int)
    )

    baseline = baseline[EXPECTED_FEATURE_ORDER]
    df_identity = df_clean[identity_columns].copy().reset_index(drop=True)
    typeaware = typeaware.reset_index(drop=True)
    baseline = baseline.reset_index(drop=True)
    return df_identity, typeaware, baseline, "Data mentah + preprocessing notebook"


def build_preprocessed_pipeline(df):
    df = _rename_preprocessed_columns(df.copy())
    missing = [
        c for c in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Kolom wajib tidak ditemukan pada data preprocessing: {missing}")

    id_col = "UMKM_ID" if "UMKM_ID" in df.columns else df.columns[0]
    df_identity = df[[id_col]].copy().reset_index(drop=True)
    baseline = df[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS].copy().reset_index(drop=True)
    typeaware = baseline.copy()
    return df_identity, typeaware, baseline, "Data sudah dipreprocessing"


def build_evaluation_spaces(typeaware):
    gower_matrix = gower.gower_matrix(
        typeaware[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    X_eval = preprocessor.fit_transform(typeaware)
    X_eval = np.asarray(
        X_eval.todense() if hasattr(X_eval, "todense") else X_eval
    )
    return gower_matrix, X_eval


def load_and_prepare_data(file_bytes, filename):
    raw = pd.read_excel(BytesIO(file_bytes))

    # Jika workbook memiliki sheet MinMax_2896, perlakukan sebagai
    # data preprocessing sesuai notebook.
    if "MinMax_2896" in pd.ExcelFile(BytesIO(file_bytes)).sheet_names:
        prep = pd.read_excel(BytesIO(file_bytes), sheet_name="MinMax_2896")
        # Hanya gunakan mode preprocessed bila struktur tersebut memang cocok.
        if all(
            c in prep.columns
            for c in [
                "NIB_SKU", "Jml_Tenaga_Kerja", "Kapasitas_Produksi",
                "Omset_Log1p", "Aset_Log1p", "Sosmed",
                "Marketplace", "Kepemilikan_Lahan"
            ]
        ):
            identity, typeaware, baseline, mode = build_preprocessed_pipeline(prep)
        else:
            identity, typeaware, baseline, mode = build_raw_pipeline(raw)
    else:
        identity, typeaware, baseline, mode = build_raw_pipeline(raw)

    gower_matrix, X_eval = build_evaluation_spaces(typeaware)

    return {
        "df_identity": identity,
        "typeaware": typeaware,
        "baseline": baseline,
        "gower_matrix": gower_matrix,
        "X_eval_combined": X_eval,
        "source_mode": mode,
    }
