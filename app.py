# app.py
import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")

st.title("Heart Failure Prediction App")
st.write("Enter patient details to predict risk of death event:")

age = st.number_input("Age", min_value = 0, max_value = 120, value = 50)
anaemia = st.selectbox("Anaemia (0 = No, 1 = Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", value = 100)
diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value = 0, max_value = 100, value = 50)
high_blood_pressure = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
platelets = st.number_input("Platelets count", value = 250000)
serum_creatinine = st.number_input("Serum Creatinine", value = 1.0)
serum_sodium = st.number_input("Serum Sodium", value = 135)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking (0 = No, 1 = Yes)", [0, 1])
time = st.number_input("Follow-up period (days)", value = 100)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium,
        sex, smoking, time
    ]], columns=[
        "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
        "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
        "sex", "smoking", "time"
    ])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High risk of death event. Probability: {prediction_proba:.2f}")
    else:
        st.success(f"Low risk of death event. Probability: {prediction_proba:.2f}")
