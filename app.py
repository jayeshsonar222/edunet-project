import numpy as np
import pandas as pd
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu


def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

diabetes_model = load_model("training_models/diabetes_model.pkl")
heart_model = load_model("training_models/heart_model.pkl")
parkinsons_model = load_model("training_models/parkinsons_model.pkl")

with st.sidebar:
    selected = option_menu("Prediction of disease outbreak system",
    ['Diabetes Prediction','Heart Prediction','Parkinsons Prediction'])
    

if selected=='Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies=st.number_input("Number of Pregnancies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood Pressure value")
    
    with col1:
        SkinThickness = st.number_input("Skin Thickness value")
    with col2:
        Insulin = st.number_input("Insulin level")
    with col3:
        BMI = st.number_input("BMI value")

    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function",format="%.3f")
    with col2:
        Age = st.number_input("Age of the person",min_value=0)
    

    if st.button("Predict"):
        input_data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = diabetes_model.predict(input_data)
        result = "The Person is Diabetic" if prediction[0] == 1 else "The Person is not Diabetic"
        st.success(f"Prediction: {result}")

    

if selected=='Heart Prediction':
    st.title('Heart Disease Prediction using ML')
    col1,col2,col3 = st.columns(3)
    with col1:
        age=st.number_input("Age of the person")
    with col2:
        sex = st.selectbox("Gender of the person", options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
    with col3:
        cp = st.number_input("Chest Pain type value")
    
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")
    with col2:
        chol = st.number_input("Cholesterol Level")
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar Value",options=[1,0],format_func=lambda x: 'Yes' if x==1 else 'No')

    with col1:
        restecg = st.number_input("Resting ECG Result Value")
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved")
    with col3:
        exang = st.selectbox("Exercise-Induced Angina",options=[1,0],format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    with col1:
        oldpeak = st.number_input("ST Depression Value")
    with col2:
        slope = st.number_input("Slope of the peak exercise Value")
    with col3:
        ca = st.number_input("Number of Major Vessels")
    
    with col1:
        thal = st.number_input("Thalassemia(0-3) Value")

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs == "Yes" else 0, 
                                restecg, thalach, 1 if exang == "Yes" else 0, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        result = "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have Heart Disease"
        st.success(f"Prediction: {result}")
    

if selected=='Parkinsons Prediction':
    st.title('Parkinsons Prediction using ML')
    col1,col2,col3 = st.columns(3)

    with col1:
        MDVPFo = st.number_input("MDVP:Fo(Hz)",format="%.3f")
    with col2:
        MDVPFhi = st.number_input("MDVP:Fhi(Hz)",format="%.3f")
    with col3:
        MDVPFlo = st.number_input("MDVP:Flo(Hz)",format="%.3f")

    with col1:
        MDVPJitter = st.number_input("MDVP:Jitter(%)",format="%.5f")
    with col2:
        MDVPJitterAbsolute = st.number_input("MDVP:Jitter(Abs)",format="%.5f")
    with col3:
        MDVPRAP = st.number_input("MDVP:RAP",format="%.5f")

    with col1:
        MDVPPPQ = st.number_input("MDVP:PPQ",format="%.5f")
    with col2:
        JitterDDP = st.number_input("Jitter:DDP",format="%.5f")
    with col3:
        MDVPShimmer = st.number_input("MDVP:Shimmer",format="%.5f")

    with col1:
        MDVPShimmerdecibal = st.number_input("MDVP:Shimmer(dB)",format="%.3f")
    with col2:
        ShimmerAPQ3 = st.number_input("Shimmer:APQ3",format="%.5f")
    with col3:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5",format="%.5f")

    with col1:
        MDVPAPQ = st.number_input("MDVP:APQ",format="%.5f")
    with col2:
        ShimmerDDA = st.number_input("Shimmer:DDA",format="%.5f")
    with col3:
        NHR = st.number_input("NHR",format="%.5f")
    
    with col1:
        HNR = st.number_input("HNR",format="%.3f")
    with col2:
        RPDE = st.number_input("RPDE",format="%.6f")
    with col3:
        DFA = st.number_input("DFA",format="%.6f")

    with col1:
        spread1 = st.number_input("Spread 1",format="%.5f")
    with col2:
        spread2 = st.number_input("Spread 2",format="%.6f")
    with col3:
        D2 = st.number_input("D2",format="%.6f")
    
    with col1:
        PPE = st.number_input("PPE",format="%.6f")
    

    if st.button("Predict Parkinson's"):
        input_data = np.array([[MDVPFo,MDVPFhi,MDVPFlo,MDVPJitter,MDVPJitterAbsolute,
                                MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmerdecibal,
                                ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,
                                DFA,spread1,spread2,D2,PPE]])
        prediction = parkinsons_model.predict(input_data)
        result = "The Person has Parkinson's Disease" if prediction[0] == 1 else "The Person does not have Parkinson's Disease"
        st.success(f"Prediction: {result}")
    