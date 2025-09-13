import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("log_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Liver Disease Prediction App")

# Input fields for user
age = st.number_input("Age", min_value=1, max_value=100, value=38)
gender = st.selectbox("Gender", ["Female", "Male"])
total_bilirubin = st.number_input("Total Bilirubin", value=3.6)
direct_bilirubin = st.number_input("Direct Bilirubin", value=1.5)
alkphos = st.number_input("Alkphos", value=330)
sgpt = st.number_input("SGPT", value=40)
sgot = st.number_input("SGOT", value=50)
albumin = st.number_input("Albumin", value=3.6)
ag_ratio = st.number_input("A/G Ratio", value=0.8)

# Convert Gender to numeric (same as training)
gender_val = 1 if gender == "Male" else 0

# Collect input into numpy array
sample = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                    alkphos, sgpt, sgot, albumin, ag_ratio]])

# Scale input
sample_scaled = scaler.transform(sample)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient is likely to have Liver Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Patient is Healthy (Probability of Disease: {prob:.2f})")
