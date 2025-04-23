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
    prob = rf_model.predict_proba(input_scaled)[0][1]

    # Custom insights based on input and prediction
    def generate_streamlit_insight(prob, age, job, credit_amount, duration):
        insights = []

        if prob > 0.8:
            insights.append("🔴 Very high risk of default. Consider stricter loan terms.")
        elif prob > 0.6:
            insights.append("🟠 High risk. Consider requiring a guarantor.")
        elif prob > 0.4:
            insights.append("🟡 Moderate risk. Review other financial indicators.")
        else:
            insights.append("🟢 Low risk. Applicant likely to repay the loan.")

        if credit_amount > 10000:
            insights.append("⚠️ High credit amount requested relative to applicant’s profile.")
        if duration > 24:
            insights.append("📅 Long loan duration may increase default risk.")
        if age < 25:
            insights.append("🧒 Younger applicant — consider employment stability.")
        if job == 0:
            insights.append("🔧 Unskilled job — assess job security.")

        return " ".join(insights)

    if prediction == 1:
        st.error("⚠️ Bad Credit Risk: This applicant is likely to default.")
    else:
        st.success("✅ Good Credit Risk: This applicant is likely to repay the loan.")

    # Show insights
    insights = generate_streamlit_insight(prob, age, job, credit_amount, duration)
    st.info(f"📊 **Personalized Insights:** {insights}")
