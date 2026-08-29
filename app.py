import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    load_and_prepare_data,
)
from src.clustering import run_base_clustering
from src.ensemble import (
    align_all_labels,
    build_consensus,
    evaluate_consensus,
    rank_multi_index,
    final_comparison,
)
from src.visualization import (
    plot_consensus_metrics,
    plot_mean_rank,
    plot_pca,
    plot_cluster_distribution,
    plot_cluster_heatmap,
    build_profile_tables,
)

# =============================================================
# ICON HELPER
# -------------------------------------------------------------
# Streamlit (>=1.31) mendukung icon "Material Symbols" bawaan lewat
# sintaks ":material/nama_icon:" pada banyak elemen teks & widget
# (st.subheader, st.markdown, st.caption, st.tabs, st.button,
# st.download_button, icon=... pada st.info/success/warning/error).
# Tidak perlu instalasi tambahan apa pun untuk fitur ini.
#
# Untuk elemen yang dirender sebagai HTML mentah (hero header,
# feature card) dipakai SVG inline bergaya "Lucide" (stroke-based,
# clean, minimal) supaya tetap konsisten tanpa emoji Unicode.
# =============================================================
_ICON_PATHS = {
    "chart": '<path d="M3 3v18h18"/><path d="M7 16v-4"/><path d="M12 16V8"/><path d="M17 16v-7"/>',
    "flask": '<path d="M9 2h6"/><path d="M10 2v6.34L4.24 18.3A2 2 0 0 0 6 21.5h12a2 2 0 0 0 1.76-3.2L14 8.34V2"/>',
    "warning": '<path d="M10.3 3.9 2.3 18a2 2 0 0 0 1.7 3h16a2 2 0 0 0 1.7-3l-8-14.1a2 2 0 0 0-3.4 0Z"/>'
               '<path d="M12 9.5v4"/><path d="M12 17h.01"/>',
}


def svg_icon(name: str, size: int = 20, color: str = "#FFFFFF", stroke_width: float = 2) -> str:
    """Render icon SVG inline bergaya Lucide (stroke-based, clean, minimal)."""
    path = _ICON_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-4px;margin-right:8px;">{path}</svg>'
    )


def method_dot(color: str) -> str:
    """Titik warna kecil pengganti emoji bulat sebagai penanda metode base clustering."""
    return (
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{color};margin-right:8px;vertical-align:middle;"></span>'
    )


# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Ensemble Clustering UMKM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# GLOBAL STYLE
# =============================================================
st.markdown("""
<style>

/* =============================================================
   0. FORCE LIGHT COLOR SCHEME — mencegah browser/OS dark-mode
   membuat elemen native (input, select, dropdown) jadi putih di
   atas putih. Ini akar dari masalah "tulisan putih background putih".
   ============================================================= */
html, body, [class*="css"] {
    color-scheme: light !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* App background: abu sangat muda supaya kartu putih tetap kontras */
[data-testid="stAppViewContainer"] {
    background: #F5F6FA;
}
[data-testid="stHeader"] {
    background: rgba(245, 246, 250, 0.0);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* =============================================================
   1. HERO HEADER
   ============================================================= */
.hero-box {
    background: linear-gradient(120deg, #4338CA 0%, #6D28D9 55%, #7C3AED 100%);
    padding: 2rem 2.4rem;
    border-radius: 20px;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(76, 29, 149, 0.28);
    position: relative;
    overflow: hidden;
}
.hero-box::after {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 260px;
    height: 260px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 50%;
    pointer-events: none;
}
.hero-box h1 {
    color: #FFFFFF !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    line-height: 1.25;
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
}
.hero-box p {
    color: rgba(255, 255, 255, 0.92) !important;
    font-size: 1rem;
    margin: 0;
    max-width: 640px;
    line-height: 1.55;
    position: relative;
    z-index: 1;
}
.hero-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.18);
    color: #fff !important;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.95rem;
    letter-spacing: 0.02em;
    position: relative;
    z-index: 1;
}

/* =============================================================
   2. METRIC CARDS
   ============================================================= */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EF;
    border-radius: 14px;
    padding: 0.95rem 1.1rem 0.75rem 1.1rem;
    box-shadow: 0 2px 10px rgba(20, 20, 43, 0.05);
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #3730A3 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #6B7280 !important;
}

/* =============================================================
   3. TABS — dibuat konsisten rounded (pill), termasuk state selected
   ============================================================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #ECEDF5;
    padding: 8px;
    border-radius: 14px;
    flex-wrap: nowrap;
    overflow-x: auto;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    min-height: 50px;
    border-radius: 10px !important;
    padding: 12px 36px !important;
    box-sizing: border-box !important;
    font-weight: 600;
    color: #52525B !important;
    background: transparent;
    display: flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    flex-shrink: 0;
    transition: background 0.15s ease, color 0.15s ease;
    overflow: visible;
}
.stTabs [data-baseweb="tab"] p {
    margin: 0 auto;
    padding: 4px 2px;
    font-size: 0.92rem;
    line-height: 1.3;
    color: inherit !important;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
}
.stTabs [data-baseweb="tab"] [data-testid="stIconMaterial"] {
    font-size: 1.05rem;
    margin: 0;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(67, 56, 202, 0.10);
    color: #4338CA !important;
}
.stTabs [aria-selected="true"] {
    background: #4338CA !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(67, 56, 202, 0.35);
}
.stTabs [aria-selected="true"]:hover {
    background: #4338CA !important;
    color: #FFFFFF !important;
}
.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* =============================================================
   4. SIDEBAR
   ============================================================= */
section[data-testid="stSidebar"] {
    background: #14142B;
}
section[data-testid="stSidebar"] * {
    color: #F3F4F6 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #B4B4C6 !important;
}

/* Number / text inputs in sidebar: solid white bg + dark text */
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #35355A !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] input[type="text"] {
    background-color: #FFFFFF !important;
    color: #14142B !important;
    -webkit-text-fill-color: #14142B !important;
    caret-color: #14142B !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] div,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* +/- step buttons pada number_input */
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] {
    background: #F3F4F6 !important;
    border: 1px solid #35355A !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] svg,
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] svg {
    fill: #14142B !important;
}

/* ---- File uploader: dropzone (state kosong / sebelum upload) ----
   Background dibuat terang solid supaya teks hitam pasti kontras,
   apa pun kondisi rendering browser. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #94A3B8 !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
    fill: #0F172A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #EEF2FF !important;
    color: #0F172A !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: 1px solid #C7D2FE !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}

/* ---- File uploader: chip setelah file ter-upload ----
   Sama-sama background terang + teks hitam supaya konsisten dan
   tetap terbaca meski di atas sidebar gelap. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] li[class*="uploadedFile"],
section[data-testid="stSidebar"] div[class*="uploadedFile"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.7rem !important;
    margin-top: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
section[data-testid="stSidebar"] li[class*="uploadedFile"] *,
section[data-testid="stSidebar"] div[class*="uploadedFile"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] small {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] svg {
    fill: #0F172A !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* PENEGASAN TERAKHIR — SOLUSI PALING PASTI:
   Daripada terus coba override warna ikon bawaan Streamlit (yang
   strukturnya bisa beda-beda dan sulit ditebak dari luar), ikon
   ASLI-nya kita sembunyikan total, lalu diganti dengan ikon custom
   sendiri lewat background-image (SVG dengan warna sudah dibakar
   langsung di dalam datanya). Cara ini 100% tidak bergantung pada
   struktur/warna internal komponen Streamlit, jadi dijamin selalu
   kontras apa pun yang terjadi di dalam sana. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] > *:first-child svg,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] > *:first-child img,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileIcon"] svg,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileIcon"] img {
    opacity: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] > *:first-child,
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileIcon"] {
    background-color: transparent !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234338CA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><polyline points='14 2 14 8 20 8'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 22px 22px !important;
    min-width: 30px !important;
    min-height: 36px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Nama file & ukuran (bukan ikon) tetap gelap supaya tetap kontras
   di atas chip putih — dikecualikan lagi di sini karena aturan svg
   di atas bisa ikut menimpa teks jika strukturnya nested. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] > div:not(:first-child),
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] > div:not(:first-child) * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    fill: initial !important;
    stroke: initial !important;
}

/* Label "Upload dataset Excel" di atas dropzone tetap putih (kontras
   dengan background sidebar gelap), dropzone-nya sendiri sudah terang. */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Tombol utama di sidebar */
section[data-testid="stSidebar"] .stButton button {
    border-radius: 10px;
    font-weight: 700;
    background: linear-gradient(120deg, #6D28D9, #7C3AED);
    border: none;
    color: #FFFFFF !important;
    padding: 0.6rem 1rem;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(120deg, #7C3AED, #8B5CF6);
    color: #fff !important;
}
section[data-testid="stSidebar"] .stButton button * {
    color: #FFFFFF !important;
}

/* Spacing antar komponen sidebar */
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.35rem;
}

/* =============================================================
   5. FORM CONTROLS DI AREA UTAMA (bukan sidebar) — jaga kontras
   ============================================================= */
[data-testid="stAppViewContainer"] .main input,
[data-testid="stAppViewContainer"] .main textarea,
[data-testid="stAppViewContainer"] .main select {
    background-color: #FFFFFF !important;
    color: #14142B !important;
    -webkit-text-fill-color: #14142B !important;
}
/* Dropdown / popover BaseWeb (selectbox, multiselect) di manapun muncul */
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
    background: #FFFFFF !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li,
div[data-baseweb="popover"] li *,
div[data-baseweb="menu"] li * {
    color: #14142B !important;
    -webkit-text-fill-color: #14142B !important;
}

/* Tombol umum di area utama (mis. tombol download) */
.stDownloadButton button,
[data-testid="stAppViewContainer"] .stButton button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.1rem !important;
}

/* =============================================================
   6. TYPOGRAFI & SPACING AREA UTAMA
   ============================================================= */
h1, h2, h3 {
    color: #1F2937;
}
h2 {
    margin-top: 0.2rem;
    margin-bottom: 1rem;
}
.stMarkdown h3 {
    font-size: 1.35rem;
    font-weight: 800;
    color: #1F2937;
    margin-top: 1.1rem;
    margin-bottom: 0.6rem;
}
.stMarkdown p {
    color: #374151;
    font-size: 1rem;
    line-height: 1.6;
}
.stMarkdown ol {
    padding-left: 1.1rem;
}
.stMarkdown ol li {
    color: #374151;
    font-size: 1.02rem;
    margin-bottom: 0.35rem;
    line-height: 1.55;
}
.stMarkdown ol li strong {
    color: #4338CA;
}
.stMarkdown blockquote {
    background: #F5F3FF;
    border-left: 4px solid #7C3AED;
    padding: 0.7rem 1.1rem;
    border-radius: 8px;
    color: #4B5563 !important;
    margin: 1rem 0;
    display: flex;
    align-items: flex-start;
}
.stMarkdown blockquote p {
    color: #4B5563 !important;
    margin: 0;
    font-size: 0.95rem;
}

/* Beri jarak konsisten antar blok konten */
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.15rem;
}

/* =============================================================
   7. DATAFRAME / TABLE
   ============================================================= */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E5E7EF;
    margin-bottom: 0.8rem;
}

/* =============================================================
   8. ALERT / INFO / WARNING / SUCCESS / ERROR BOXES
   ============================================================= */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(20,20,43,0.06);
}
div[data-testid="stAlert"] p {
    color: #1F2937 !important;
    font-weight: 500;
}

/* =============================================================
   9. EXPANDER
   ============================================================= */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EF !important;
    border-radius: 12px !important;
    overflow: hidden;
    margin-bottom: 1rem;
    background: #FFFFFF;
}
[data-testid="stExpander"] summary {
    font-weight: 700;
    color: #1F2937 !important;
    padding: 0.6rem 0.9rem !important;
}

/* =============================================================
   10. STATUS WIDGET (st.status)
   ============================================================= */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
    border: 1px solid #E5E7EF !important;
    background: #FFFFFF !important;
}
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {
    color: #1F2937 !important;
}

/* =============================================================
   11. DIVIDER
   ============================================================= */
hr {
    margin: 1.7rem 0;
    border-color: #E5E7EF;
}

/* =============================================================
   12. INTRO FEATURE CARD (halaman sebelum upload)
   ============================================================= */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EF;
    border-radius: 16px;
    padding: 1.7rem 2rem;
    box-shadow: 0 3px 14px rgba(20, 20, 43, 0.06);
    margin-top: 0.5rem;
}
.feature-card h3 {
    color: #1F2937 !important;
    margin-top: 0 !important;
    display: flex;
    align-items: center;
}
.feature-card p {
    color: #374151 !important;
}
.feature-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 1.1rem 0 1.3rem 0;
}
.feature-step {
    background: linear-gradient(120deg, #EEF2FF, #F5F3FF);
    border: 1px solid #E0E0F5;
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #4338CA !important;
    display: inline-flex;
    align-items: center;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #4338CA;
    color: #FFFFFF !important;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 8px;
    flex-shrink: 0;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# HERO HEADER
# =============================================================
st.markdown(f"""
<div class="hero-box">
    <h1>{svg_icon("chart", size=28, color="#FFFFFF")}Ensemble Clustering UMKM</h1>
    <p>Ensemble Clustering Unweighted — K-Means, K-Prototypes, dan Gower K-Medoids</p>
    <span class="hero-pill">Hungarian Alignment · Majority Voting · Multi-Index Validation</span>
</div>
""", unsafe_allow_html=True)

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## :material/tune: Konfigurasi")
    st.caption("Atur parameter clustering sebelum menjalankan pipeline.")

    uploaded = st.file_uploader(
        ":material/upload_file: Upload dataset Excel",
        type=["xlsx", "xls"],
        help="Gunakan data_umkm.xlsx atau data hasil preprocessing.",
    )

    st.markdown("**Rentang jumlah cluster (K)**")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        k_min = st.number_input("K min", min_value=2, max_value=9, value=2)
    with col_k2:
        k_max = st.number_input("K max", min_value=3, max_value=10, value=10)

    random_state = st.number_input(":material/casino: Random State", min_value=0, value=42, step=1)

    st.markdown("")
    run_button = st.button(
        "Jalankan Ensemble",
        icon=":material/play_arrow:",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**:material/alt_route: Pipeline**")
    st.markdown(
        "1. Data → 2. Preprocessing → 3. Base Clustering → "
        "4. Hungarian Alignment → 5. Majority Voting → 6. Evaluasi → 7. Profil"
    )
    st.divider()
    st.caption("Ensemble Clustering App · Skenario Unweighted")

# =============================================================
# VALIDATION
# =============================================================
if k_min >= k_max:
    st.error("K minimum harus lebih kecil daripada K maksimum.", icon=":material/error:")
    st.stop()

if uploaded is None:
    st.info("Upload dataset Excel dari sidebar untuk memulai.", icon=":material/arrow_back:")

    st.markdown(f"""
<div class="feature-card">
    <h3>{svg_icon("flask", size=22, color="#7C3AED")}Aplikasi ini menjalankan skenario <span style="color:#7C3AED;">&nbsp;Unweighted</span></h3>
    <p>Tiga base clustering yang digunakan:</p>
    <ol>
        <li><strong>K-Means</strong> — clustering berbasis jarak Euclidean untuk fitur numerik.</li>
        <li><strong>K-Prototypes</strong> — kombinasi fitur numerik dan kategorikal sekaligus.</li>
        <li><strong>Gower K-Medoids</strong> — berbasis matriks jarak Gower untuk data campuran.</li>
    </ol>
    <p style="margin-top:1rem;"><strong>Alur proses ensemble:</strong></p>
    <div class="feature-steps">
        <span class="feature-step"><span class="step-num">1</span>Hungarian Label Alignment</span>
        <span class="feature-step"><span class="step-num">2</span>Majority Voting (Equal Vote)</span>
        <span class="feature-step"><span class="step-num">3</span>Multi-Index Validation</span>
    </div>
    <blockquote>
        {svg_icon("warning", size=18, color="#7C3AED")}<span>Fuzzy Entropy, Feature Selection, dan pembobotan fitur <strong>tidak digunakan</strong> pada aplikasi ini.</span>
    </blockquote>
</div>
""", unsafe_allow_html=True)
    st.stop()

# =============================================================
# DATA LOADING
# =============================================================
@st.cache_data(show_spinner=False)
def cached_prepare(file_bytes, filename):
    return load_and_prepare_data(file_bytes, filename)

try:
    prepared = cached_prepare(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Gagal membaca/preprocessing dataset: {exc}", icon=":material/error:")
    st.stop()

df_identity = prepared["df_identity"]
typeaware = prepared["typeaware"]
baseline = prepared["baseline"]
gower_matrix = prepared["gower_matrix"]
X_eval = prepared["X_eval_combined"]
source_mode = prepared["source_mode"]

if "results" not in st.session_state:
    st.session_state.results = None

# =============================================================
# RUN PIPELINE
# =============================================================
if run_button:
    with st.status("Menjalankan ensemble clustering...", expanded=True) as status:
        st.write("1/5 Menjalankan K-Means...")
        with st.spinner("K-Means K=2..10"):
            base = run_base_clustering(
                baseline=baseline,
                typeaware=typeaware,
                gower_matrix=gower_matrix,
                X_eval_combined=X_eval,
                k_values=list(range(int(k_min), int(k_max) + 1)),
                random_state=int(random_state),
            )

        st.write("2/5 Hungarian Label Alignment...")
        aligned = align_all_labels(base["labels"], list(range(int(k_min), int(k_max) + 1)))

        st.write("3/5 Majority Voting Equal Vote...")
        consensus_labels = build_consensus(aligned, list(range(int(k_min), int(k_max) + 1)))

        st.write("4/5 Evaluasi konsensus...")
        eval_consensus = evaluate_consensus(
            consensus_labels, gower_matrix, X_eval,
            list(range(int(k_min), int(k_max) + 1))
        )
        kept, dropped, threshold = rank_multi_index(eval_consensus, len(df_identity))

        best_k = int(kept.iloc[0]["K"])
        best_labels = consensus_labels[best_k]

        st.write("5/5 Menyusun profil dan perbandingan...")
        comparison = final_comparison(
            base["labels"], best_labels, best_k,
            gower_matrix, X_eval
        )

        st.session_state.results = {
            "base": base,
            "aligned": aligned,
            "consensus_labels": consensus_labels,
            "evaluation": eval_consensus,
            "evaluation_kept": kept,
            "evaluation_dropped": dropped,
            "threshold": threshold,
            "best_k": best_k,
            "best_labels": best_labels,
            "comparison": comparison,
            "prepared": prepared,
        }
        status.update(label=f"Selesai — K final = {best_k}", state="complete")

res = st.session_state.results

# =============================================================
# TABS
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    ":material/database: Dataset",
    ":material/hub: Base Clustering",
    ":material/how_to_vote: Alignment & Voting",
    ":material/monitoring: Evaluasi",
    ":material/groups: Profil Cluster",
    ":material/bar_chart: Visualisasi",
])

with tab1:
    st.subheader(":material/database: Dataset dan Preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah UMKM", f"{len(df_identity):,}")
    c2.metric("Fitur", len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS))
    c3.metric("Fitur Numerik", len(NUMERICAL_COLUMNS))
    c4.metric("Fitur Kategorikal", len(CATEGORICAL_COLUMNS))

    st.info(f"Mode data: **{source_mode}**", icon=":material/info:")

    with st.expander(":material/checklist: Fitur yang digunakan pada skenario Unweighted", expanded=True):
        st.dataframe(
            pd.DataFrame({
                "Tipe": ["Numerik"] * 4 + ["Kategorikal"] * 4,
                "Fitur": NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS,
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**:material/search: Preview data**")
    st.dataframe(typeaware.head(10), use_container_width=True)

    if res is not None:
        st.success(f"Ensemble sudah dijalankan. K final = **{res['best_k']}**", icon=":material/check_circle:")

with tab2:
    st.subheader(":material/hub: Evaluasi Base Clustering")
    if res is None:
        st.warning("Klik **Jalankan Ensemble** terlebih dahulu.", icon=":material/hourglass_empty:")
    else:
        base = res["base"]
        method_colors = {"kmeans": "#3B82F6", "kproto": "#8B5CF6", "kmedoids": "#10B981"}
        for method, title in [
            ("kmeans", "K-Means"),
            ("kproto", "K-Prototypes"),
            ("kmedoids", "Gower K-Medoids"),
        ]:
            st.markdown(
                f'#### {method_dot(method_colors[method])}{title}',
                unsafe_allow_html=True,
            )
            table = base["evaluation"][method].copy()
            show_cols = [c for c in [
                "K", "Inertia", "Cost", "Loss", "Silhouette", "DBI", "CHI",
                "Min_Cluster_Size", "Max_Cluster_Size"
            ] if c in table.columns]
            st.dataframe(
                table[show_cols].round(4),
                use_container_width=True,
                hide_index=True,
            )

with tab3:
    st.subheader(":material/how_to_vote: Hungarian Label Alignment & Majority Voting")
    if res is None:
        st.warning("Klik **Jalankan Ensemble** terlebih dahulu.", icon=":material/hourglass_empty:")
    else:
        k = res["best_k"]
        aligned = res["aligned"][k]
        labels = res["consensus_labels"][k]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("K Final", k)
        c2.metric("K-Means", len(np.unique(aligned["KM"])))
        c3.metric("K-Prototypes Aligned", len(np.unique(aligned["KP"])))
        c4.metric("Gower K-Medoids Aligned", len(np.unique(aligned["GM"])))

        st.markdown("**:material/table_rows: Contoh label setelah alignment dan voting**")
        preview = pd.DataFrame({
            "K-Means": aligned["KM"] + 1,
            "K-Prototypes (Aligned)": aligned["KP"] + 1,
            "Gower K-Medoids (Aligned)": aligned["GM"] + 1,
            "Label Konsensus": labels + 1,
        })
        if "UMKM_ID" in df_identity.columns:
            preview.insert(0, "UMKM_ID", df_identity["UMKM_ID"].values)
        st.dataframe(
            pd.concat([preview.head(10), preview.tail(10)]),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Equal Vote: setiap metode memiliki satu suara. Jika ketiga metode "
            "memberikan label berbeda, label K-Means digunakan sebagai tie-break.",
            icon=":material/info:",
        )

with tab4:
    st.subheader(":material/monitoring: Multi-Index Validation Ensemble")
    if res is None:
        st.warning("Klik **Jalankan Ensemble** terlebih dahulu.", icon=":material/hourglass_empty:")
    else:
        best = res["evaluation_kept"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("K Final", res["best_k"])
        c2.metric("Silhouette", f"{best['Silhouette']:.4f}")
        c3.metric("DBI", f"{best['DBI']:.4f}")
        c4.metric("CHI", f"{best['CHI']:,.2f}")

        st.markdown("#### :material/table_view: Evaluasi semua K")
        st.dataframe(
            res["evaluation"].round(4),
            use_container_width=True,
            hide_index=True,
        )

        if len(res["evaluation_dropped"]) > 0:
            st.caption(
                f":material/warning: K yang tidak lolos filter ukuran cluster minimum "
                f"(threshold = {res['threshold']}) : "
                f"{', '.join(map(str, res['evaluation_dropped']['K'].tolist()))}"
            )

        st.markdown("#### :material/leaderboard: Peringkat Multi-Index")
        st.dataframe(
            res["evaluation_kept"].round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### :material/balance: Perbandingan 4 Model pada K Final")
        st.dataframe(
            res["comparison"].round(4),
            use_container_width=True,
            hide_index=True,
        )

with tab5:
    st.subheader(":material/groups: Profil Cluster Ensemble")
    if res is None:
        st.warning("Klik **Jalankan Ensemble** terlebih dahulu.", icon=":material/hourglass_empty:")
    else:
        numeric_profile, categorical_profile = build_profile_tables(
            typeaware, res["best_labels"], NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
        )
        st.markdown(f"### :material/pie_chart: Distribusi Cluster (K={res['best_k']})")
        sizes = pd.Series(res["best_labels"]).value_counts().sort_index()
        dist = pd.DataFrame({
            "Cluster": [f"C{i+1}" for i in sizes.index],
            "Jumlah UMKM": sizes.values,
            "Persentase (%)": (sizes.values / len(res["best_labels"]) * 100).round(1),
        })
        st.dataframe(dist, use_container_width=True, hide_index=True)

        st.markdown("### :material/numbers: Profil Numerik")
        st.dataframe(numeric_profile, use_container_width=True, hide_index=True)

        st.markdown("### :material/label: Profil Kategorikal / Akses Digital")
        st.dataframe(categorical_profile, use_container_width=True, hide_index=True)

with tab6:
    st.subheader(":material/bar_chart: Visualisasi")
    if res is None:
        st.warning("Klik **Jalankan Ensemble** terlebih dahulu.", icon=":material/hourglass_empty:")
    else:
        eval_df = res["evaluation"].sort_values("K")
        st.pyplot(plot_consensus_metrics(eval_df), clear_figure=True)
        st.pyplot(plot_mean_rank(res["evaluation_kept"], res["best_k"]), clear_figure=True)

        st.markdown("### :material/scatter_plot: PCA: Base Clustering vs Ensemble")
        fig = plot_pca(
            baseline,
            res["aligned"][res["best_k"]],
            res["best_labels"],
            res["best_k"],
            int(random_state),
        )
        st.pyplot(fig, clear_figure=True)

        st.markdown("### :material/bar_chart_4_bars: Distribusi Anggota Cluster")
        st.pyplot(plot_cluster_distribution(res["best_labels"], res["best_k"]), clear_figure=True)

        st.markdown("### :material/grid_on: Heatmap Rata-rata Fitur")
        st.pyplot(plot_cluster_heatmap(typeaware, res["best_labels"], NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS), clear_figure=True)

# =============================================================
# DOWNLOAD RESULTS
# =============================================================
if res is not None:
    st.divider()
    st.subheader(":material/download: Unduh Hasil")

    result_df = df_identity.copy()
    for k in sorted(res["consensus_labels"]):
        result_df[f"Cluster_K{k}"] = res["consensus_labels"][k]
    result_df["Cluster_Final"] = res["best_labels"]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Ensemble_All_K", index=False)
        res["evaluation"].to_excel(writer, sheet_name="Evaluasi_Ensemble", index=False)
        res["evaluation_kept"].to_excel(writer, sheet_name="Multi_Index_Ranking", index=False)
        res["comparison"].to_excel(writer, sheet_name="Perbandingan_Final", index=False)
        sizes = pd.Series(res["best_labels"]).value_counts().sort_index()
        pd.DataFrame({
            "Cluster": [f"C{i+1}" for i in sizes.index],
            "Jumlah UMKM": sizes.values,
            "Persentase (%)": sizes.values / len(res["best_labels"]) * 100,
        }).to_excel(writer, sheet_name="Distribusi_Cluster", index=False)

    st.download_button(
        "Download hasil_ensemble_unweighted.xlsx",
        data=buffer.getvalue(),
        file_name="hasil_ensemble_unweighted.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
        use_container_width=True,
    )