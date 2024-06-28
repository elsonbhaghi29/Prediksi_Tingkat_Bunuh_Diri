import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Judul aplikasi
st.title("Prediksi Tingkat Bunuh Diri di Amerika Serikat")

# Memuat model
model = joblib.load('model.pkl')

# Input tahun dari pengguna
year = st.number_input('Masukkan Tahun:', min_value=1950, max_value=2024, step=1)

# Prediksi
if st.button('Prediksi'):
    age_num = 0  # Asumsi age_num = 0 untuk "All ages"
    prediction = model.predict([[year, age_num]])[0]
    st.write(f"Prediksi Tingkat Bunuh Diri per 100,000 penduduk pada tahun {year} adalah {prediction:.2f}")
