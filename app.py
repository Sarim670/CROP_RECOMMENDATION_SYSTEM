import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and preprocessor
model = joblib.load('notebooks/crop_recommendation_model.pkl')
scaler = joblib.load('notebooks/scaler.pkl')
label_encoder = joblib.load('notebooks/label_encoder.pkl')

st.title("🌾 Crop Recommendation System")
st.markdown("### Enter soil & weather conditions:")

# Input fields for soil and weather conditions
N = st.number_input("Nitrogen (N) in soil (kg/ha):", min_value=0)
P = st.number_input("Phosphorus (P) in soil (kg/ha):", min_value=0)
K = st.number_input("Potassium (K) in soil (kg/ha):", min_value=0)
temperature = st.number_input("Temperature (°C):", format="%.2f")
humidity = st.number_input("Humidity (%):", format="%.2f")
rainfall = st.number_input("Rainfall (mm):", format="%.2f")
ph = st.number_input("Soil pH:", min_value=0.0, max_value=14.0, format="%.2f")

# Predict button
if st.button("Predict Crop"):
    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall,]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Decode the predicted crop label
    crop = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"The recommended crop is: **{crop.upper()}**")