import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pyngrok import ngrok

# Function to load data
@st.cache
def load_data():
    # Assuming you have a dataset named 'suicide_rates.csv'
    data = pd.read_csv('suicide_rates.csv')
    return data

# Load data
data = load_data()

# Display the data
st.write("Dataset:")
st.write(data.head())

# Data Visualization
st.write("Data Visualization:")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax, annot=True)
st.pyplot(fig)

# Train/Test Split
X = data.drop(columns=['suicides/100k pop'])
y = data['suicides/100k pop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the Model
joblib.dump(model, 'model.pkl')

# Load the Model
model = joblib.load('model.pkl')

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# User Input
st.write("Input Data for Prediction:")
year = st.number_input("Year", min_value=1985, max_value=2021, value=2000)
sex = st.selectbox("Sex", options=["male", "female"])
age = st.selectbox("Age", options=["5-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"])
country = st.selectbox("Country", options=data['country'].unique())
generation = st.selectbox("Generation", options=data['generation'].unique())

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'year': [year],
    'sex': [sex],
    'age': [age],
    'country': [country],
    'generation': [generation]
})

# Make prediction
prediction = model.predict(input_data)

st.write(f"Predicted Suicides/100k Pop: {prediction[0]}")
