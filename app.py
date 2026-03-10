import streamlit as st
import pandas as pd
import pickle

# load model
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "LR.pkl")

model = pickle.load(open(model_path, "rb"))

st.title("Heart Attack Prediction App")

st.write("Input patient data:")

age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

if st.button("Predict"):

    data = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,
                          exang,oldpeak,slope,ca,thal]],
        columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Attack")
    else:
        st.success("Low Risk of Heart Attack")