import streamlit as st
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = "asthma_model.pkl"

@st.cache_resource()  # Cache the model for faster loading
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🫁 Asthma Prediction App")

# Example input fields
cough = st.slider("Cough (0-10)", 0, 10, 5)
breath_shortness = st.slider("Shortness of Breath (0-10)", 0, 10, 5)
wheezing = st.slider("Wheezing (0-10)", 0, 10, 5)
chest_tightness = st.slider("Chest Tightness (0-10)", 0, 10, 5)
fatigue = st.slider("Fatigue (0-10)", 0, 10, 5)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[cough, breath_shortness, wheezing, chest_tightness, fatigue]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("🔴 High chance of Asthma! Please consult a doctor.")
    else:
        st.success("🟢 Low risk of Asthma. Stay healthy!")
