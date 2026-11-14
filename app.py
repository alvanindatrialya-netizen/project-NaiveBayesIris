import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Klasifikasi Iris - Naive Bayes", page_icon="🌸")

@st.cache_resource
def load_model():
    return joblib.load("model_nb_iris.pkl")

data = load_model()
skenario_list = ["80:20", "70:30", "60:40"]

st.title("🌸 Klasifikasi Bunga Iris")
st.write("Prediksi jenis bunga Iris menggunakan algoritma **Naive Bayes (GaussianNB)**.")

skenario = st.selectbox("Pilih skenario Train:Test", skenario_list)
st.caption(f"Akurasi skenario {skenario}: {data['akurasi'][skenario]:.2%}")

st.subheader("Masukkan ukuran bunga (cm)")
sepal_l = st.number_input("Sepal Length", 4.0, 8.0, 5.1, 0.1)
sepal_w = st.number_input("Sepal Width", 2.0, 4.5, 3.5, 0.1)
petal_l = st.number_input("Petal Length", 1.0, 7.0, 1.4, 0.1)
petal_w = st.number_input("Petal Width", 0.1, 2.5, 0.2, 0.1)

if st.button("Prediksi", type="primary", use_container_width=True):
    model = data[skenario]
    X = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    hasil = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.success(f"Hasil prediksi: **{hasil}**")
    st.subheader("Probabilitas tiap kelas")
    for kelas, p in zip(model.classes_, proba):
        st.write(f"{kelas}: {p:.2%}")
        st.progress(float(p))