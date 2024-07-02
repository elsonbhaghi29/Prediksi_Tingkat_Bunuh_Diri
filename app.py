# Import library
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("Prediksi Tingkat Bunuh Diri di Amerika Serikat")

# Memuat model
model = joblib.load('model.pkl')

# Input tahun dari pengguna
year = st.number_input('Masukkan Tahun:', min_value=1950, max_value=2024, step=1)

# Input grup usia dari pengguna
age_groups = {0: 'All Ages', 1: '0-14 years', 2: '15-24 years', 3: '25-34 years', 4: '35-54 years', 5: '55-74 years', 6: '75+ years'}
selected_age_group = st.selectbox('Pilih Grup Usia:', list(age_groups.values()), index=0)

# Mendapatkan kode grup usia berdasarkan pilihan pengguna
for code, group_name in age_groups.items():
    if group_name == selected_age_group:
        age_num = code

# Prediksi
if st.button('Prediksi'):
    prediction = model.predict([[year, age_num]])[0]
    st.write(f"Prediksi Tingkat Bunuh Diri per 100,000 penduduk pada tahun {year} untuk grup usia {selected_age_group} adalah {prediction:.2f}")

# Visualisasi data historis (contoh saja, sesuaikan dengan data Anda)
st.subheader('Visualisasi Tingkat Bunuh Diri Historis')

# Load data historis (contoh saja, sesuaikan dengan data Anda)
historical_data = pd.read_csv('historical_data.csv')

# Plot menggunakan seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Suicide Rate', data=historical_data)
plt.title('Tren Tingkat Bunuh Diri di Amerika Serikat')
plt.xlabel('Tahun')
plt.ylabel('Tingkat Bunuh Diri')
st.pyplot()

# Catatan: pastikan file 'historical_data.csv' sudah ada di direktori yang sama dengan notebook Anda.
