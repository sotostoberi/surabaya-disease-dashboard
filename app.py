"""
Dashboard Analisis & Prediksi Kasus Penyakit Kota Surabaya
=============================================================
Dibangun dengan Streamlit. Menyajikan eksplorasi data, pemetaan
kerawanan kecamatan (clustering), forecasting SARIMA on-the-fly,
evaluasi 4 model prediksi (Linear Regression, Random Forest,
XGBoost, SARIMA), serta asisten AI (Gemini) yang terintegrasi.

Desain UI: minimalis modern bergaya SaaS — palet teal gelap & oranye
terracotta, tipografi Lora (display) + Inter (body), kartu bersudut
membulat dengan bayangan halus, mengikuti brief desain yang diberikan.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
import google.generativeai as genai

# -------------------------------------------------------------------
# KONFIGURASI HALAMAN (harus jadi perintah Streamlit pertama)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Penyakit Surabaya",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    AI_ENABLED = True

except Exception as e:
    gemini_model = None
    AI_ENABLED = False

# -------------------------------------------------------------------
# PALET WARNA & STYLE
# -------------------------------------------------------------------
COLOR_PRIMARY = "#0F4C5C"       # teal gelap - identitas utama
COLOR_PRIMARY_SOFT = "#156D80"  # teal sedikit lebih terang (gradien)
COLOR_ACCENT = "#E36414"        # oranye terracotta - aksen/AI
COLOR_HIGH = "#C0392B"          # merah - rawan tinggi
COLOR_HIGH_BG = "#FDECEA"
COLOR_MED = "#E67E22"           # oranye - rawan sedang
COLOR_MED_BG = "#FEF3E7"
COLOR_LOW = "#27AE60"           # hijau - rawan rendah
COLOR_LOW_BG = "#EAFBF1"
COLOR_BG_CARD = "#F7F5F0"
COLOR_MUTED = "#5b6b6e"

MODEL_COLORS = {
    "LR": "#9B59B6",
    "RF": "#C0392B",
    "XGB": "#E67E22",
    "SARIMA": "#27AE60",
}

RISK_COLOR_MAP = {
    "Rawan Tinggi": COLOR_HIGH,
    "Rawan Sedang": COLOR_MED,
    "Rawan Rendah": COLOR_LOW,
}
RISK_BADGE_MAP = {
    "Rawan Tinggi": (COLOR_HIGH, COLOR_HIGH_BG),
    "Rawan Sedang": (COLOR_MED, COLOR_MED_BG),
    "Rawan Rendah": (COLOR_LOW, COLOR_LOW_BG),
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Lora', serif;
        color: {COLOR_PRIMARY};
    }}

    /* ---------- Header ---------- */
    .main-header {{
        font-family: 'Lora', serif;
        font-size: 2.05rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        margin-bottom: 0.1rem;
        letter-spacing: -0.01em;
    }}
    .sub-header {{
        font-size: 0.98rem;
        color: {COLOR_MUTED};
        margin-bottom: 1.4rem;
    }}
    .panel-title {{
        font-family: 'Lora', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: {COLOR_PRIMARY};
        margin-bottom: 0.6rem;
    }}

    /* ---------- Kartu generik via st.container(border=True) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px !important;
        border: 1px solid #EAE8E1 !important;
        box-shadow: 0 2px 14px rgba(15, 76, 92, 0.05);
        padding: 0.2rem 0.2rem 0.4rem 0.2rem;
        background: #FFFFFF;
    }}

    /* ---------- Metric card kustom ---------- */
    .metric-card-v2 {{
        background: linear-gradient(155deg, #FFFFFF 0%, {COLOR_BG_CARD} 100%);
        border-radius: 14px;
        border-left: 4px solid {COLOR_PRIMARY};
        box-shadow: 0 2px 10px rgba(15, 76, 92, 0.06);
        padding: 0.95rem 1.1rem;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }}
    .metric-card-v2:hover {{
        box-shadow: 0 6px 18px rgba(15, 76, 92, 0.10);
        transform: translateY(-1px);
    }}
    .metric-icon {{ font-size: 1.25rem; margin-bottom: 0.35rem; opacity: 0.9; }}
    .metric-label {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
        color: {COLOR_MUTED}; text-transform: uppercase; margin-bottom: 0.15rem;
    }}
    .metric-value {{ font-size: 1.55rem; font-weight: 700; color: {COLOR_PRIMARY}; line-height: 1.1; }}
    .metric-delta {{ font-size: 0.78rem; font-weight: 600; margin-top: 0.25rem; }}

    /* ---------- Kartu wawasan AI ---------- */
    .ai-card {{
        position: relative;
        background: linear-gradient(155deg, #FFF8F2 0%, #FDF1E6 100%);
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 14px;
        padding: 1rem 1.2rem 1.7rem 1.2rem;
        margin-top: 0.6rem;
        overflow: hidden;
    }}
    .ai-card-label {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
        color: {COLOR_ACCENT}; text-transform: uppercase; margin-bottom: 0.4rem;
    }}
    .ai-card-text {{ font-size: 0.92rem; color: #4a3c30; line-height: 1.5; }}
    .ai-card-icon {{
        position: absolute; right: 0.7rem; bottom: 0.4rem;
        font-size: 1.5rem; opacity: 0.35;
    }}

    /* ---------- Kartu insight besar (kecamatan / model) ---------- */
    .insight-card-big {{
        background: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #EAE8E1;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 76, 92, 0.05);
        height: 100%;
    }}
    .insight-card-big .ic-label {{
        font-size: 0.75rem; font-weight: 600; color: {COLOR_MUTED};
        text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.5rem;
    }}
    .insight-card-big .ic-value {{
        font-family: 'Lora', serif; font-size: 1.5rem; font-weight: 700;
        color: {COLOR_PRIMARY}; margin-bottom: 0.6rem;
    }}
    .insight-card-big .ic-sub {{ font-size: 0.85rem; color: {COLOR_MUTED}; margin-top: 0.6rem; }}

    .risk-badge {{
        display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
    }}

    /* ---------- Insight / warning box (gaya lama, masih dipakai) ---------- */
    .insight-box {{
        background-color: #EFF6F4;
        border-left: 5px solid {COLOR_PRIMARY};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}
    .warning-box {{
        background-color: #FDF1E6;
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}

    div[data-testid="stMetric"] {{
        background-color: {COLOR_BG_CARD};
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border-left: 5px solid {COLOR_PRIMARY};
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{ background-color: #fbfaf8; }}
    .sidebar-logo {{ text-align: center; padding: 0.7rem 0 1.1rem 0; }}
    .sidebar-logo .icon {{ font-size: 2.1rem; line-height: 1; }}
    .sidebar-logo .name {{
        font-family: 'Lora', serif; font-weight: 700; font-size: 1.1rem;
        color: {COLOR_PRIMARY}; margin-top: 0.25rem;
    }}
    .sidebar-logo .tag {{
        font-size: 0.66rem; color: #94a3a6; letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.15rem; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-radius: 10px; padding: 0.45rem 0.6rem; transition: background 0.15s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: #eef3f2; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_PRIMARY_SOFT});
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
        color: #ffffff !important; font-weight: 600;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {{
        accent-color: {COLOR_PRIMARY};
    }}

    /* ---------- Chat AI Assistant ---------- */
    div[data-testid="stChatMessage"] {{ border-radius: 14px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# HELPER UI — komponen kartu kustom dipakai berulang di banyak halaman
# -------------------------------------------------------------------
def metric_card_html(icon, label, value, delta_pct=None):
    """Kartu metrik: ikon kecil, label tipis, nilai besar, opsional indikator tren."""
    delta_html = ""
    if delta_pct is not None:
        favorable = delta_pct <= 0  # untuk jumlah kasus, turun = baik
        arrow = "▼" if delta_pct <= 0 else "▲"
        color = COLOR_LOW if favorable else COLOR_HIGH
        delta_html = (
            f'<div class="metric-delta" style="color:{color};">'
            f'{arrow} {abs(delta_pct):.1f}% vs bulan lalu</div>'
        )
    return f"""
    <div class="metric-card-v2">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def risk_badge_html(label):
    color, bg = RISK_BADGE_MAP.get(label, (COLOR_PRIMARY, COLOR_BG_CARD))
    return f'<span class="risk-badge" style="color:{color};background:{bg};">{label.upper()}</span>'

def ai_card_html(text):
    return f"""
    <div class="ai-card">
        <div class="ai-card-label">⚡ Wawasan AI </div>
        <div class="ai-card-text">{text}</div>
        <div class="ai-card-icon">🤖</div>
    </div>
    """

# -------------------------------------------------------------------
# LOAD DATA (cached) — identik dengan versi sebelumnya
# -------------------------------------------------------------------
@st.cache_data
def load_main_data():
    df = pd.read_csv(DATA_DIR / "data_bersih.csv.gz", compression="gzip", parse_dates=["Tanggal"])
    return df

@st.cache_data
def load_clustering():
    return pd.read_excel(DATA_DIR / "Hasil_Clustering_Kecamatan.xlsx")

@st.cache_data
def load_top5():
    return pd.read_excel(DATA_DIR / "Top5_Penyakit.xlsx")

@st.cache_data
def load_evaluasi():
    return pd.read_csv(DATA_DIR / "Tabel_Evaluasi_4_Model.csv")

@st.cache_data
def load_adf():
    return pd.read_excel(DATA_DIR / "Ringkasan_ADF.xlsx")

df_main = load_main_data()
df_cluster = load_clustering()
df_top5 = load_top5()
df_eval = load_evaluasi()
df_adf = load_adf()

# -------------------------------------------------------------------
# AI HELPERS (Gemini) — wawasan otomatis + fallback statis bila gagal
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_ai_insight(kec_nama, kec_total, kec_label, peny_nama, peny_total, model_terbaik, win_count, total_eval):
    fallback = (
        f"Kecamatan {kec_nama} mencatat kerawanan tertinggi dengan {kec_total:,.0f} kasus, "
        f"didominasi {peny_nama}. Prioritaskan intervensi di wilayah ini dan gunakan model "
        f"{model_terbaik} untuk memantau tren ke depan."
    )
    if not AI_ENABLED:
        return fallback
    prompt = (
        "Anda adalah analis kesehatan masyarakat untuk Dinas Kesehatan Kota Surabaya. "
        "Tulis SATU wawasan ringkas (maksimal 2 kalimat, Bahasa Indonesia, gaya langsung dan "
        "actionable, tanpa salam pembuka, tanpa markdown) berdasarkan data ini:\n"
        f"- Kecamatan paling rawan: {kec_nama} ({kec_total:,.0f} kasus, kategori {kec_label})\n"
        f"- Penyakit dominan: {peny_nama} ({peny_total:,.0f} kasus)\n"
        f"- Model prediksi terbaik: {model_terbaik}, menang {win_count} dari {total_eval} kategori penyakit"
    )
    try:
        resp = gemini_model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else fallback
    except Exception:
        return fallback

@st.cache_data(show_spinner=False)
def run_sarima_forecast(penyakit, n_periods=6):
    """Latih SARIMA on-the-fly dari total kasus bulanan kota untuk satu jenis penyakit."""
    series = (
        df_main[df_main["jenis_penyakit"] == penyakit]
        .groupby("Tanggal")["jumlah_kasus"].sum()
        .sort_index()
    )
    if len(series) < 18:
        return None
    sarima = SARIMAX(
        series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fitted = sarima.fit(disp=False)
    forecast_res = fitted.get_forecast(steps=n_periods)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.2)
    return series, forecast_mean, conf_int

# -------------------------------------------------------------------
# SIDEBAR — LOGO & NAVIGASI
# -------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <div class="icon">🏥</div>
        <div class="tag">Surabaya Disease Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Pilih halaman",
    [
        "🏠 Ringkasan Umum",
        "📊 Eksplorasi Data",
        "🗺️ Pemetaan Kerawanan",
        "📈 Forecasting & Evaluasi Model",
        "🤖 AI Assistant",
        "📝 Insight & Kesimpulan",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber data: Data Kasus Penyakit Kota Surabaya per Kecamatan & Fasilitas "
    "Kesehatan, periode 2022–2026."
)
st.sidebar.caption("Dibuat dengan Streamlit · Analisis & Prediksi oleh Tim Data Dinas Kesehatan Kota Surabaya · 2024")

# =====================================================================
# HALAMAN 1 — RINGKASAN UMUM
# =====================================================================
if page == "🏠 Ringkasan Umum":
    st.markdown('<div class="main-header">Analisis & Prediksi Kasus Penyakit Kota Surabaya</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Dashboard interaktif hasil analisis data kasus penyakit '
        '31 kecamatan di Kota Surabaya, periode 2022–2026</div>',
        unsafe_allow_html=True,
    )

    total_kasus = df_main["jumlah_kasus"].sum()
    n_kecamatan = df_main["kecamatan"].nunique()
    n_penyakit = df_main["jenis_penyakit"].nunique()
    n_faskes = df_main["nama_faskes"].nunique()
    tahun_min, tahun_max = int(df_main["tahun"].min()), int(df_main["tahun"].max())

    tren = df_main.groupby("Tanggal", as_index=False)["jumlah_kasus"].sum().sort_values("Tanggal")
    delta_pct = None
    if len(tren) >= 2 and tren["jumlah_kasus"].iloc[-2]:
        last, prev = tren["jumlah_kasus"].iloc[-1], tren["jumlah_kasus"].iloc[-2]
        delta_pct = (last - prev) / prev * 100

    # --- A. Baris metrik ringkas ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(metric_card_html("🏘️", "Kecamatan", f"{n_kecamatan}"), unsafe_allow_html=True)
    col2.markdown(metric_card_html("🦠", "Jenis Penyakit", f"{n_penyakit}"), unsafe_allow_html=True)
    col3.markdown(metric_card_html("🏨", "Fasilitas Kesehatan", f"{n_faskes}"), unsafe_allow_html=True)
    col4.markdown(metric_card_html("📋", "Total Kasus", f"{total_kasus:,.0f}", delta_pct), unsafe_allow_html=True)
    col5.markdown(metric_card_html("📅", "Rentang Tahun", f"{tahun_min}–{tahun_max}"), unsafe_allow_html=True)

    st.markdown("###")
    c1, c2 = st.columns([1.3, 1])

    # --- B. Tren (kiri) & Top 5 + Wawasan AI (kanan) ---
    with c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Tren Total Kasus Bulanan — Kota Surabaya</div>', unsafe_allow_html=True)
            fig = px.area(tren, x="Tanggal", y="jumlah_kasus", color_discrete_sequence=[COLOR_PRIMARY])
            fig.update_traces(line_width=2.4, fillcolor="rgba(15,76,92,0.15)")
            fig.update_layout(
                height=362, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="", yaxis_title="",
                plot_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f0ec"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Top 5 Penyakit Dominan</div>', unsafe_allow_html=True)
            fig2 = px.bar(
                df_top5.sort_values("Total_Kasus"), x="Total_Kasus", y="Penyakit", orientation="h",
                color_discrete_sequence=[COLOR_ACCENT], text="Total_Kasus",
            )
            fig2.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig2.update_layout(
                height=185, margin=dict(l=10, r=25, t=10, b=10),
                xaxis_title="", yaxis_title="", plot_bgcolor="white",
                xaxis=dict(showgrid=False), yaxis=dict(tickfont=dict(size=10.5)),
            )
            st.plotly_chart(fig2, use_container_width=True)

        kec_top = df_cluster.sort_values("total_kasus", ascending=False).iloc[0]
        best_model = df_eval["Model_Terbaik"].value_counts().idxmax()
        win_count = df_eval["Model_Terbaik"].value_counts().max()
        peny_top_row = df_top5.iloc[0]

        ai_text = generate_ai_insight(
            kec_top["kecamatan"], kec_top["total_kasus"], kec_top["risk_label"],
            peny_top_row["Penyakit"], peny_top_row["Total_Kasus"],
            best_model, win_count, len(df_eval),
        )
        st.markdown(ai_card_html(ai_text), unsafe_allow_html=True)

    st.markdown("###")
    # --- C. Status kerawanan & model terbaik ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""<div class="insight-card-big">
                <div class="ic-label">📍 Kecamatan Paling Rawan</div>
                <div class="ic-value">{kec_top['kecamatan'].upper()}</div>
                {risk_badge_html(kec_top['risk_label'])}
                <div class="ic-sub">Total {kec_top['total_kasus']:,.0f} kasus tercatat berdasarkan hasil clustering K-Means.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""<div class="insight-card-big">
                <div class="ic-label">🏆 Model Prediksi Terbaik</div>
                <div class="ic-value">{best_model}</div>
                <div class="ic-sub">Memenangkan <b>{win_count} dari {len(df_eval)}</b> kategori penyakit utama dalam evaluasi RMSE / MAE / R².</div>
            </div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 2 — EKSPLORASI DATA
# =====================================================================
elif page == "📊 Eksplorasi Data":
    st.markdown('<div class="main-header">Eksplorasi Data Kasus Penyakit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Telusuri data berdasarkan kecamatan, jenis penyakit, dan periode waktu</div>',
        unsafe_allow_html=True,
    )

    # --- Filter dalam satu baris horizontal ---
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        kec_opts = ["Semua Kecamatan"] + sorted(df_main["kecamatan"].unique().tolist())
        sel_kec = st.selectbox("Kecamatan", kec_opts)
    with fc2:
        peny_opts = ["Semua Penyakit"] + sorted(df_main["jenis_penyakit"].unique().tolist())
        sel_peny = st.selectbox("Jenis Penyakit", peny_opts)
    with fc3:
        tahun_opts = ["Semua Tahun"] + sorted(df_main["tahun"].unique().tolist(), reverse=True)
        sel_tahun = st.selectbox("Tahun", tahun_opts)

    df_f = df_main.copy()
    if sel_kec != "Semua Kecamatan":
        df_f = df_f[df_f["kecamatan"] == sel_kec]
    if sel_peny != "Semua Penyakit":
        df_f = df_f[df_f["jenis_penyakit"] == sel_peny]
    if sel_tahun != "Semua Tahun":
        df_f = df_f[df_f["tahun"] == sel_tahun]

    st.markdown("###")
    m1, m2, m3 = st.columns(3)
    m1.markdown(metric_card_html("📊", "Total Kasus (filter aktif)", f"{df_f['jumlah_kasus'].sum():,.0f}"), unsafe_allow_html=True)
    m2.markdown(metric_card_html("🧾", "Jumlah Baris Data", f"{len(df_f):,}"), unsafe_allow_html=True)
    m3.markdown(
        metric_card_html("📈", "Rata-rata Kasus / Periode", f"{df_f['jumlah_kasus'].mean():,.1f}" if len(df_f) else "0"),
        unsafe_allow_html=True,
    )

    st.markdown("###")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Tren Kasus dari Waktu ke Waktu</div>', unsafe_allow_html=True)
            tren_f = df_f.groupby("Tanggal", as_index=False)["jumlah_kasus"].sum()
            if len(tren_f):
                fig = px.line(tren_f, x="Tanggal", y="jumlah_kasus", markers=True,
                              color_discrete_sequence=[COLOR_PRIMARY])
                fig.update_layout(height=370, plot_bgcolor="white",
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="", yaxis_title="Jumlah Kasus",
                                  xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0ec"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data untuk kombinasi filter ini.")

    with c2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Distribusi Kasus per Jenis Penyakit</div>', unsafe_allow_html=True)
            dist = df_f.groupby("jenis_penyakit", as_index=False)["jumlah_kasus"].sum().sort_values("jumlah_kasus", ascending=True).tail(10)
            if len(dist):
                fig = px.bar(dist, x="jumlah_kasus", y="jenis_penyakit", orientation="h",
                            color_discrete_sequence=[COLOR_ACCENT])
                fig.update_layout(height=370, plot_bgcolor="white",
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_title="Jumlah Kasus", yaxis_title="",
                                  xaxis=dict(showgrid=False),
                                  yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data untuk kombinasi filter ini.")

    st.markdown("###")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Perbandingan Antar Kecamatan</div>', unsafe_allow_html=True)
        kec_compare = df_f.groupby("kecamatan", as_index=False)["jumlah_kasus"].sum().sort_values("jumlah_kasus", ascending=False)
        if len(kec_compare):
            fig = px.bar(kec_compare, x="kecamatan", y="jumlah_kasus",
                        color="jumlah_kasus", color_continuous_scale=["#27AE60", "#E67E22", "#C0392B"])
            fig.update_layout(height=400, plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="", yaxis_title="Jumlah Kasus",
                              coloraxis_showscale=False)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("###")
    with st.expander("🔍 Lihat Data Mentah (Tabel)"):
        st.dataframe(
            df_f[["kecamatan", "jenis_penyakit", "nama_faskes", "periode", "jumlah_kasus", "is_anomaly"]]
            .sort_values("periode", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=350,
        )
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh data terfilter (CSV)", csv, "data_filtered.csv", "text/csv")

    n_anomaly = int(df_f["is_anomaly"].sum())
    if n_anomaly > 0:
        st.markdown(
            f"""<div class="warning-box">⚠️ Terdeteksi <b>{n_anomaly}</b> titik data anomali
            (lonjakan/penurunan kasus tidak wajar) pada kombinasi filter saat ini.</div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 3 — PEMETAAN KERAWANAN (CLUSTERING)
# =====================================================================
elif page == "🗺️ Pemetaan Kerawanan":
    st.markdown('<div class="main-header">Pemetaan Kerawanan Kecamatan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hasil clustering K-Means (k=3) berdasarkan total kasus penyakit per kecamatan</div>',
        unsafe_allow_html=True,
    )

    silhouette = 0.4389  
    n_tinggi = (df_cluster["risk_label"] == "Rawan Tinggi").sum()
    n_sedang = (df_cluster["risk_label"] == "Rawan Sedang").sum()
    n_rendah = (df_cluster["risk_label"] == "Rawan Rendah").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card_html("📐", "Silhouette Score", f"{silhouette:.4f}"), unsafe_allow_html=True)
    c2.markdown(metric_card_html("🔴", "Rawan Tinggi", f"{n_tinggi} kecamatan"), unsafe_allow_html=True)
    c3.markdown(metric_card_html("🟠", "Rawan Sedang", f"{n_sedang} kecamatan"), unsafe_allow_html=True)
    c4.markdown(metric_card_html("🟢", "Rawan Rendah", f"{n_rendah} kecamatan"), unsafe_allow_html=True)

    st.markdown("###")
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Peta Kerawanan: Total Kasus vs Variabilitas Bulanan</div>',
            unsafe_allow_html=True,
        )
        fig = px.scatter(
            df_cluster, x="total_kasus", y="std_kasus",
            color="risk_label", color_discrete_map=RISK_COLOR_MAP,
            hover_name="kecamatan",
            size="total_kasus", size_max=34,
            labels={
                "total_kasus": "Total Kasus", "std_kasus": "Std. Deviasi Kasus Bulanan",
                "risk_label": "Kategori",
            },
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
        fig.update_layout(
            height=500, plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title="Kategori Kerawanan",
            xaxis=dict(showgrid=True, gridcolor="#eef0ef"),
            yaxis=dict(showgrid=True, gridcolor="#eef0ef"),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ukuran titik proporsional terhadap total kasus. Arahkan kursor ke titik untuk melihat nama kecamatan.")

    st.markdown("###")
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Tabel Detail Clustering</div>', unsafe_allow_html=True)
            df_show = df_cluster.copy().sort_values("total_kasus", ascending=False)
            df_show["total_kasus"] = df_show["total_kasus"].map("{:,.0f}".format)
            df_show["rata_rata_bulanan"] = df_show["rata_rata_bulanan"].map("{:,.1f}".format)
            df_show["std_kasus"] = df_show["std_kasus"].map("{:,.1f}".format)
            st.dataframe(
                df_show[["kecamatan", "total_kasus", "rata_rata_bulanan", "std_kasus", "risk_label"]]
                .rename(columns={
                    "kecamatan": "Kecamatan", "total_kasus": "Total Kasus",
                    "rata_rata_bulanan": "Rata-rata Bulanan", "std_kasus": "Std. Deviasi",
                    "risk_label": "Kategori",
                }),
                use_container_width=True, height=400, hide_index=True,
            )

    with col2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Proporsi Kecamatan per Kategori</div>', unsafe_allow_html=True)
            risk_count = df_cluster["risk_label"].value_counts().reset_index()
            risk_count.columns = ["risk_label", "jumlah"]
            fig2 = px.pie(
                risk_count, names="risk_label", values="jumlah",
                color="risk_label", color_discrete_map=RISK_COLOR_MAP,
                hole=0.55,
            )
            fig2.update_traces(textinfo="label+percent", textfont_size=11)
            fig2.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        top3 = df_cluster.sort_values("total_kasus", ascending=False).head(3)
        badges = " ".join(f"{r['kecamatan']} {risk_badge_html(r['risk_label'])}" for _, r in top3.iterrows())
        st.markdown(
            f"""<div class="warning-box">⚠️ <b>3 kecamatan paling rawan</b> — perlu prioritas
            intervensi kesehatan masyarakat:<br><br>{badges}</div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 4 — FORECASTING & EVALUASI MODEL
# =====================================================================
elif page == "📈 Forecasting & Evaluasi Model":
    st.markdown('<div class="main-header">Forecasting & Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Perbandingan performa 4 model prediksi: Linear Regression, Random Forest, XGBoost, dan SARIMA</div>',
        unsafe_allow_html=True,
    )

    best_overall = df_eval["Model_Terbaik"].value_counts().idxmax()
    win_count = df_eval["Model_Terbaik"].value_counts().max()
    avg_rmse_best = df_eval["RF_RMSE"].mean()  # RF adalah model terbaik keseluruhan

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card_html("🏆", "Model Terbaik Keseluruhan", best_overall), unsafe_allow_html=True)
    c2.markdown(metric_card_html("✅", "Jumlah Kemenangan", f"{win_count} dari {len(df_eval)} penyakit"), unsafe_allow_html=True)
    c3.markdown(metric_card_html("📉", "Rata-rata RMSE (RF)", f"{avg_rmse_best:,.2f}"), unsafe_allow_html=True)

    st.markdown("###")
    st.subheader("Pilih Penyakit untuk Analisis Detail")
    peny_list = df_eval["Nama_Penyakit"].tolist()
    sel_peny_eval = st.selectbox("Jenis Penyakit", peny_list, label_visibility="collapsed")
    row = df_eval[df_eval["Nama_Penyakit"] == sel_peny_eval].iloc[0]

    metrics_df = pd.DataFrame({
        "Model": ["LR", "RF", "XGB", "SARIMA"],
        "RMSE": [row["LR_RMSE"], row["RF_RMSE"], row["XGB_RMSE"], row["SARIMA_RMSE"]],
        "MAE": [row["LR_MAE"], row["RF_MAE"], row["XGB_MAE"], row["SARIMA_MAE"]],
        "R2": [row["LR_R2"], row["RF_R2"], row["XGB_R2"], row["SARIMA_R2"]],
    })

    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True):
            st.markdown(f'<div class="panel-title">RMSE & MAE — {sel_peny_eval}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=metrics_df["Model"], y=metrics_df["RMSE"], name="RMSE", marker_color=COLOR_PRIMARY))
            fig.add_trace(go.Bar(x=metrics_df["Model"], y=metrics_df["MAE"], name="MAE", marker_color=COLOR_ACCENT))
            fig.update_layout(
                barmode="group", height=350, plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0ec"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with cc2:
        with st.container(border=True):
            st.markdown(f'<div class="panel-title">R² Score per Model — {sel_peny_eval}</div>', unsafe_allow_html=True)
            colors_r2 = [MODEL_COLORS[m] for m in metrics_df["Model"]]
            fig2 = go.Figure(go.Bar(
                x=metrics_df["Model"], y=metrics_df["R2"], marker_color=colors_r2,
                text=metrics_df["R2"].round(3), textposition="outside",
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(height=350, plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="R² Score",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0ec"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""<div class="insight-box">
        Model terbaik untuk <b>{sel_peny_eval}</b> adalah <b>{row['Model_Terbaik']}</b>,
        dengan RMSE terendah pada model tersebut dibanding 3 model lainnya.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("###")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Tabel Lengkap Evaluasi 4 Model — Top 5 Penyakit</div>', unsafe_allow_html=True)
        df_eval_show = df_eval.copy()
        for c in df_eval_show.columns:
            if "RMSE" in c or "MAE" in c:
                df_eval_show[c] = df_eval_show[c].map("{:,.2f}".format)
            elif "R2" in c:
                df_eval_show[c] = df_eval_show[c].map("{:,.4f}".format)
        st.dataframe(
            df_eval_show.drop(columns=["Penyakit"]).rename(columns={"Nama_Penyakit": "Penyakit", "Model_Terbaik": "Model Terbaik"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("###")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Uji Stasioneritas (Augmented Dickey-Fuller Test)</div>', unsafe_allow_html=True)
        df_adf_show = df_adf.copy()
        df_adf_show["ADF_Statistic"] = df_adf_show["ADF_Statistic"].map("{:.4f}".format)
        df_adf_show["P_Value"] = df_adf_show["P_Value"].map("{:.2e}".format)
        df_adf_show["Stasioner"] = df_adf_show["Stasioner"].map({True: "✅ Ya", False: "❌ Tidak"})
        st.dataframe(
            df_adf_show.rename(columns={
                "Penyakit": "Penyakit", "ADF_Statistic": "Statistik ADF",
                "P_Value": "P-Value", "Stasioner": "Stasioner?",
                "Differencing_d": "Differencing (d)",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Seluruh data deret waktu kasus penyakit dinyatakan **stasioner** (p-value < 0.05) "
            "tanpa perlu differencing tambahan (d=0), sehingga layak dimodelkan langsung dengan SARIMA."
        )

    # --- Simulasi forecasting SARIMA aktual vs prediksi (on-the-fly) ---
    st.markdown("###")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Simulasi Forecasting SARIMA — Aktual vs Prediksi</div>', unsafe_allow_html=True)
        sel_peny_forecast = st.selectbox("Pilih penyakit untuk simulasi forecasting", peny_list, key="forecast_select")
        n_periods = st.slider("Jumlah bulan ke depan", min_value=3, max_value=12, value=6, key="forecast_horizon")

        result = None
        with st.spinner("Melatih model SARIMA & menghasilkan prediksi..."):
            try:
                result = run_sarima_forecast(sel_peny_forecast, n_periods)
            except Exception as e:
                st.warning(f"Model SARIMA tidak dapat dilatih untuk data ini: {e}")

        if result is not None:
            series, forecast_mean, conf_int = result
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode="lines", name="Aktual",
                line=dict(color=COLOR_PRIMARY, width=2.2),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_mean.index, y=forecast_mean.values, mode="lines", name="Prediksi SARIMA",
                line=dict(color=MODEL_COLORS["SARIMA"], width=2.2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=list(forecast_mean.index) + list(forecast_mean.index[::-1]),
                y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
                fill="toself", fillcolor="rgba(39,174,96,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="Interval Keyakinan 80%",
            ))
            fig.update_layout(
                height=420, plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="", yaxis_title="Jumlah Kasus",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0ec"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Forecast {n_periods} bulan ke depan menggunakan SARIMA(1,1,1)(1,1,1,12) yang dilatih "
                "langsung dari data historis penyakit terpilih (agregat seluruh kecamatan)."
            )
        else:
            st.info("Data historis untuk penyakit ini terlalu sedikit untuk forecasting SARIMA (minimal 18 periode bulanan).")

# =====================================================================
# HALAMAN 5 —  AI ASSISTANT
# =====================================================================
elif page == "🤖 AI Assistant":
    st.markdown('<div class="main-header"> AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Tanya jawab interaktif seputar data kasus penyakit Kota Surabaya, didukung Gemini</div>',
        unsafe_allow_html=True,
    )

    if not AI_ENABLED:
        st.warning(
            "Asisten AI belum aktif. Tambahkan `GEMINI_API_KEY` pada `st.secrets` "
            "untuk mengaktifkan fitur chat berbasis Gemini."
        )
    else:
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if "rapids_chat" not in st.session_state:
            kec_top_chat = df_cluster.sort_values("total_kasus", ascending=False).iloc[0]
            best_model_chat = df_eval["Model_Terbaik"].value_counts().idxmax()
            system_context = (
                "Anda adalah AI Assistant, asisten analis data untuk dashboard kasus "
                "penyakit Kota Surabaya. Jawab pertanyaan pengguna secara ringkas dan informatif "
                "dalam Bahasa Indonesia, berdasarkan ringkasan data berikut:\n"
                f"- Periode data: {int(df_main['tahun'].min())}–{int(df_main['tahun'].max())}\n"
                f"- Jumlah kecamatan: {df_main['kecamatan'].nunique()}, jenis penyakit: {df_main['jenis_penyakit'].nunique()}\n"
                f"- Total kasus tercatat: {df_main['jumlah_kasus'].sum():,.0f}\n"
                f"- Kecamatan paling rawan: {kec_top_chat['kecamatan']} ({kec_top_chat['risk_label']}, "
                f"{kec_top_chat['total_kasus']:,.0f} kasus)\n"
                f"- Model prediksi terbaik secara keseluruhan: {best_model_chat}\n"
                "Jika ditanya hal di luar topik data kesehatan Surabaya, jawab singkat lalu arahkan "
                "kembali ke topik dashboard ini."
            )
            st.session_state.rapids_chat = gemini_model.start_chat(history=[
                {"role": "user", "parts": [system_context]},
                {"role": "model", "parts": ["Baik, saya siap membantu menganalisis data kasus penyakit Kota Surabaya."]},
            ])

        def _process_message(text):
            st.session_state.chat_messages.append({"role": "user", "content": text})
            try:
                resp = st.session_state.rapids_chat.send_message(text)
                answer = resp.text
            except Exception as e:
                answer = f"Maaf, terjadi kendala saat menghubungi AI Assistant: {e}"
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        if not st.session_state.chat_messages:
            st.markdown("**Coba tanyakan:**")
            qc1, qc2, qc3 = st.columns(3)
            quick_prompts = [
                "Kecamatan mana yang paling rawan dan kenapa?",
                "Penyakit apa yang paling dominan di Surabaya?",
                "Model prediksi mana yang paling akurat?",
            ]
            for col, q in zip([qc1, qc2, qc3], quick_prompts):
                if col.button(q, use_container_width=True):
                    _process_message(q)

        for msg in st.session_state.chat_messages:
            avatar = "🤖" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        user_input = st.chat_input("Tanyakan sesuatu tentang data kasus penyakit Surabaya...")
        if user_input:
            _process_message(user_input)
            st.rerun()

# =====================================================================
# HALAMAN 6 — INSIGHT & KESIMPULAN
# =====================================================================
elif page == "📝 Insight & Kesimpulan":
    st.markdown('<div class="main-header">Insight & Kesimpulan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ringkasan naratif hasil analisis & prediksi kasus penyakit Kota Surabaya</div>',
        unsafe_allow_html=True,
    )

    total_kasus_pernafasan = df_top5.iloc[0]["Total_Kasus"]
    kec_top = df_cluster.sort_values("total_kasus", ascending=False).iloc[0]

    with st.container(border=True):
        st.markdown(
            f"""
            ### 📌 Ringkasan Eksekutif

            Analisis ini mencakup **31 kecamatan** dan **22 jenis penyakit** di Kota Surabaya
            sepanjang periode **2022–2026**.

            - **Penyakit dominan**: *{df_top5.iloc[0]['Penyakit']}* dengan total
              **{total_kasus_pernafasan:,.0f}** kasus — jauh di atas penyakit lainnya.
            - **Kecamatan paling rawan**: **{kec_top['kecamatan']}**, dengan total kasus
              **{kec_top['total_kasus']:,.0f}**, dikategorikan sebagai *{kec_top['risk_label']}*.
            - **Clustering K-Means (k=3)** menghasilkan silhouette score **0.4389**, menunjukkan
              pemisahan kelompok yang cukup baik antara kecamatan rawan tinggi, sedang, dan rendah.
            - **Model prediksi terbaik secara keseluruhan**: **Random Forest (RF)**, memenangkan
              2 dari 5 kategori penyakit utama, dengan rata-rata RMSE terbaik **2.857,99**.
            """
        )

    st.markdown("###")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="panel-title">🦠 Top 5 Penyakit Dominan</div>', unsafe_allow_html=True)
            for i, r in df_top5.iterrows():
                st.markdown(f"**{i+1}. {r['Penyakit']}** — {r['Total_Kasus']:,.0f} kasus")

    with col2:
        with st.container(border=True):
            st.markdown('<div class="panel-title">🗺️ Top 5 Kecamatan Paling Rawan</div>', unsafe_allow_html=True)
            top5_kec = df_cluster.sort_values("total_kasus", ascending=False).head(5)
            for i, r in top5_kec.reset_index(drop=True).iterrows():
                st.markdown(
                    f"**{i+1}. {r['kecamatan']}** — {r['total_kasus']:,.0f} kasus {risk_badge_html(r['risk_label'])}",
                    unsafe_allow_html=True,
                )

    st.markdown("###")
    with st.container(border=True):
        st.markdown('<div class="panel-title">🤖 Model Terbaik per Jenis Penyakit</div>', unsafe_allow_html=True)
        for _, r in df_eval.iterrows():
            st.markdown(f"- **{r['Nama_Penyakit']}** → model terbaik: **{r['Model_Terbaik']}**")

    st.markdown("###")
    st.markdown(
        """
        ### 💡 Rekomendasi

        1. **Prioritaskan intervensi kesehatan** di kecamatan dengan kategori *Rawan Tinggi*
           (Kenjeran, Sawahan, Semampir, Wonokromo, Wonocolo), terutama untuk penyakit sistem
           pernafasan dan pencernaan yang mendominasi.
        2. **Gunakan model Random Forest** sebagai model utama untuk prediksi kasus penyakit
           pernafasan dan pencernaan, karena memberikan akurasi (RMSE) terbaik dibanding
           Linear Regression, XGBoost, maupun SARIMA.
        3. **Gunakan model SARIMA** untuk kategori penyakit dengan pola musiman yang lebih
           kuat, seperti faktor status kesehatan umum dan penyakit musculoskeletal.
        4. **Pantau anomali data secara berkala** — terdapat sejumlah titik anomali yang
           terdeteksi dan dapat menjadi sinyal dini lonjakan kasus di lapangan.
        """
    )

    st.markdown("---")
    st.caption(
        "Dashboard ini dibangun berdasarkan hasil analisis (Rangkaian Analisis)"
        "Prediksi & Identifikasi Daerah Sensitif) terhadap data kasus penyakit Kota Surabaya."
    )