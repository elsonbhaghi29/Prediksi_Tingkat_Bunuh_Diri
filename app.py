# Import Library
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

# Pilihan grup usia
age_group = st.selectbox('Pilih Grup Usia:', ['All Ages', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])

# Mengubah nama grup usia menjadi angka sesuai data yang Anda gunakan
age_num_map = {
    'All Ages': 0,
    '15-24 years': 1,
    '25-34 years': 2,
    '35-54 years': 3,
    '55-74 years': 4,
    '75+ years': 5
}
age_num = age_num_map[age_group]

# Prediksi
if st.button('Prediksi'):
    prediction = model.predict([[year, age_num]])[0]
    st.write(f"Prediksi Tingkat Bunuh Diri per 100,000 penduduk pada tahun {year} untuk grup usia {age_group} adalah {prediction:.2f}")

# Visualisasi Tingkat Bunuh Diri Historis
st.subheader('Visualisasi Tingkat Bunuh Diri Historis')
historical_data = pd.read_csv('data.csv')  # Pastikan file CSV tersedia dan path-nya benar
st.line_chart(historical_data.set_index('Year'))
