# Dashboard Analisis & Prediksi Kasus Penyakit Kota Surabaya

Dashboard interaktif berbasis **Streamlit** untuk eksplorasi data, pemetaan
kerawanan kecamatan (clustering), serta evaluasi & hasil *forecasting*
kasus penyakit di Kota Surabaya periode 2022–2026.

## 📁 Struktur Folder

```
dashboard_surabaya/
├── app.py                          # Aplikasi utama Streamlit
├── requirements.txt                # Daftar dependency Python
├── .streamlit/
│   └── config.toml                 # Tema warna dashboard
└── data/
    ├── data_bersih.csv.gz          # Data utama (hasil cleaning)
    ├── Hasil_Clustering_Kecamatan.xlsx
    ├── Tabel_Evaluasi_4_Model.csv
    ├── Top5_Penyakit.xlsx
    └── Ringkasan_ADF.xlsx
```

## 🚀 Cara Menjalankan secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard akan terbuka otomatis di `http://localhost:8501`.

## ☁️ Cara Deploy ke Streamlit Community Cloud

1. **Buat repository baru di GitHub**, lalu upload seluruh isi folder ini
   (termasuk folder `data/` dan `.streamlit/`).
   - Pastikan struktur folder tetap sama persis seperti di atas.
   - File `data_bersih.csv.gz` sengaja dikompresi (≈1.4 MB) agar repo ringan
     dan upload ke GitHub lancar.

2. **Buka [share.streamlit.io](https://share.streamlit.io)** dan login
   dengan akun GitHub Anda.

3. Klik **"New app"**, lalu pilih:
   - **Repository**: repo yang baru dibuat
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. Klik **Deploy**. Tunggu beberapa menit sampai proses build selesai.

5. Dashboard akan mendapat link publik seperti:
   `https://nama-app-anda.streamlit.app`

## 📊 Konten Dashboard

| Halaman | Isi |
|---|---|
| 🏠 Ringkasan Umum | KPI utama, tren kasus bulanan, top 5 penyakit |
| 📊 Eksplorasi Data | Filter interaktif kecamatan/penyakit/tahun, tabel data mentah |
| 🗺️ Pemetaan Kerawanan | Hasil clustering K-Means (Rawan Tinggi/Sedang/Rendah) |
| 📈 Forecasting & Evaluasi Model | Perbandingan LR, RF, XGBoost, SARIMA + uji ADF |
| 📝 Insight & Kesimpulan | Ringkasan naratif & rekomendasi |

## 🔧 Catatan Teknis

- Data dimuat dengan `@st.cache_data` agar dashboard tetap responsif.
- Semua chart menggunakan **Plotly** (interaktif: zoom, hover, pan).
- Jika ingin mengganti/menambah data, cukup ganti file di folder `data/`
  dan sesuaikan path pembacaan data di `app.py`.
