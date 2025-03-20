import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("asthma_model.pkl")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7f9;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-image: url('https://source.unsplash.com/1600x900/?lungs,health');
            background-size: cover;
        }
        .title {
            color: #2C3E50;
            font-weight: bold;
            text-align: center;
        }
        .prediction-box {
            background: #D4EDDA;
            padding: 10px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
        }
        .warning-box {
            background: #F8D7DA;
            padding: 10px;
            border-radius: 10px;
            font-size: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="title">🫁 Asthma Diagnosis Prediction</h1>', unsafe_allow_html=True)

# Image
st.image("https://source.unsplash.com/800x300/?asthma,healthcare", use_column_width=True)

st.markdown("### **Enter symptoms and get an AI-based prediction!**")

# Feature names (modify based on your dataset)
feature_names = ["Cough", "Shortness of Breath", "Wheezing", "Chest Tightness", "Fatigue"]

# Two-column layout for better UI
col1, col2 = st.columns(2)
user_input = []

for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    value = col.slider(f"{feature} (0-10)", min_value=0, max_value=10, value=5)
    user_input.append(value)

# Convert input to NumPy array
input_array = np.array(user_input).reshape(1, -1)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1] * 100  # Probability score

    if prediction == 1:
        st.markdown(f'<div class="warning-box">🔴 High chance of Asthma ({probability:.2f}%)! Please consult a doctor.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="prediction-box">🟢 Low risk of Asthma ({100 - probability:.2f}%)! Stay healthy!</div>', unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("This AI-powered app predicts the likelihood of asthma based on symptoms.")

st.sidebar.markdown("### 📌 How It Works")
st.sidebar.write("""
1️⃣ Adjust the sliders to enter your symptoms.  
2️⃣ Click "Predict" to get AI-based analysis.  
3️⃣ See the risk level and take action if needed.  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Your Name** | Powered by AI 🚀")
