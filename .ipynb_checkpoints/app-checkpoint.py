import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Predicting Credit Risk for Loan Applicants")
st.write("Enter the applicant's financial details to predict risk.")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job Type", options=[0, 1, 2, 3], help="0 = Unskilled, 3 = Highly Skilled")
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (in months)", min_value=1, value=12)

# Prediction
if st.button("Predict Credit Risk"):
    input_data = np.array([[age, job, credit_amount, duration]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("⚠️ Bad Credit Risk: This applicant is likely to default.")
    else:
        st.success("✅ Good Credit Risk: This applicant is likely to repay the loan.")
