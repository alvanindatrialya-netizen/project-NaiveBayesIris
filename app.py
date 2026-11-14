import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Klasifikasi Iris | Naive Bayes", page_icon="🌿", layout="centered")

# ---------- Gaya ----------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 760px;}
.hero {
    background: linear-gradient(135deg, #14b8a6 0%, #3b82f6 60%, #6366f1 100%);
    border-radius: 20px; padding: 28px 26px; color: #fff; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(59,130,246,.28);
}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 6px 0 0; opacity: .92; font-size: .98rem;}
.badge {
    display: inline-block; background: rgba(255,255,255,.22); border-radius: 999px;
    padding: 3px 12px; font-size: .78rem; margin-top: 12px; margin-right: 6px;
}
.result {
    border-radius: 18px; padding: 22px; text-align: center; color: #fff;
    margin: 14px 0 8px; box-shadow: 0 6px 18px rgba(0,0,0,.15);
}
.result .emoji {font-size: 2.6rem; line-height: 1;}
.result .name {font-size: 1.7rem; font-weight: 700; margin-top: 6px;}
.result .conf {opacity: .95; font-size: .95rem; margin-top: 2px;}
.barwrap {margin: 10px 0;}
.barlabel {display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: 4px;}
.bar {height: 12px; border-radius: 999px; background: rgba(128,128,128,.18); overflow: hidden;}
.fill {height: 100%; border-radius: 999px;}
.stButton > button {border-radius: 12px; font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Data ----------
@st.cache_resource
def load_model():
    return joblib.load("model_nb_iris.pkl")

data = load_model()
SKENARIO = ["80:20", "70:30", "60:40"]

INFO = {
    "Iris-setosa": {"emoji": "🌿", "warna": "#10b981", "nama": "Iris Setosa",
                    "ket": "Kelopak kecil dan pendek, paling mudah dibedakan."},
    "Iris-versicolor": {"emoji": "🌼", "warna": "#f59e0b", "nama": "Iris Versicolor",
                        "ket": "Ukuran menengah, sering mirip dengan Virginica."},
    "Iris-virginica": {"emoji": "🌻", "warna": "#3b82f6", "nama": "Iris Virginica",
                       "ket": "Kelopak paling besar dan panjang."},
}
FITUR = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
st.session_state.setdefault("riwayat", [])

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <h1>🌿 Klasifikasi Bunga Iris</h1>
  <p>Masukkan ukuran bunga, lalu model akan menebak jenisnya.</p>
  <span class="badge">Naive Bayes</span><span class="badge">GaussianNB</span><span class="badge">150 data</span>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🔍 Prediksi", "📊 Perbandingan Model"])

# ---------- Tab Prediksi ----------
with tab1:
    skenario = st.selectbox("Skenario pembagian data (Train:Test)", SKENARIO)
    st.caption(f"Akurasi model skenario {skenario}: **{data['akurasi'][skenario]:.2%}**")

    st.markdown("**Ukuran bunga (cm)**")
    a, b = st.columns(2)
    sl = a.slider("Sepal Length (panjang kelopak luar)", 4.0, 8.0, 5.1, 0.1)
    sw = b.slider("Sepal Width (lebar kelopak luar)", 2.0, 4.5, 3.5, 0.1)
    pl = a.slider("Petal Length (panjang mahkota)", 1.0, 7.0, 1.4, 0.1)
    pw = b.slider("Petal Width (lebar mahkota)", 0.1, 2.5, 0.2, 0.1)

    if st.button("🔮 Prediksi Sekarang", type="primary", width="stretch"):
        model = data[skenario]
        X = np.array([[sl, sw, pl, pw]])
        hasil = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        info = INFO.get(hasil, {"emoji": "🌼", "warna": "#10b981", "nama": hasil, "ket": ""})

        st.session_state["last"] = [sl, sw, pl, pw]
        st.session_state["riwayat"].append(
            {"Sepal L": sl, "Sepal W": sw, "Petal L": pl, "Petal W": pw,
             "Skenario": skenario, "Hasil": info["nama"], "Keyakinan": f"{proba.max():.1%}"}
        )

        st.markdown(
            f"""
<div class="result" style="background: linear-gradient(135deg, {info['warna']}, {info['warna']}bb);">
  <div class="emoji">{info['emoji']}</div>
  <div class="name">{info['nama']}</div>
  <div class="conf">Keyakinan model: {proba.max():.1%}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(info["ket"])

        st.markdown("**Probabilitas tiap jenis**")
        for kelas, p in sorted(zip(model.classes_, proba), key=lambda t: -t[1]):
            w = INFO.get(kelas, {"warna": "#10b981", "nama": kelas})
            st.markdown(
                f"""
<div class="barwrap">
  <div class="barlabel"><span>{w['nama']}</span><b>{p:.1%}</b></div>
  <div class="bar"><div class="fill" style="width:{p*100:.1f}%; background:{w['warna']};"></div></div>
</div>
""",
                unsafe_allow_html=True,
            )

    if st.session_state["riwayat"]:
        st.divider()
        h1, h2 = st.columns([3, 1])
        h1.markdown("**🕘 Riwayat prediksi**")
        if h2.button("Hapus", width="stretch"):
            st.session_state["riwayat"] = []
            st.rerun()
        st.dataframe(
            pd.DataFrame(st.session_state["riwayat"][::-1]),
            width="stretch", hide_index=True,
        )

# ---------- Tab Perbandingan ----------
with tab2:
    st.markdown("Perbandingan akurasi ketiga skenario pembagian data.")
    df = pd.DataFrame(
        {"Skenario": SKENARIO, "Akurasi (%)": [round(data["akurasi"][s] * 100, 2) for s in SKENARIO]}
    )
    m1, m2, m3 = st.columns(3)
    for col, s in zip([m1, m2, m3], SKENARIO):
        col.metric(f"Train:Test {s}", f"{data['akurasi'][s]:.2%}")
    st.bar_chart(df.set_index("Skenario"), color="#14b8a6")
    terbaik = max(SKENARIO, key=lambda s: data["akurasi"][s])
    st.info(f"Skenario dengan akurasi tertinggi: **{terbaik}** ({data['akurasi'][terbaik]:.2%}).")