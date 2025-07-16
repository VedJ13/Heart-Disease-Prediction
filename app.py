%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('heartsense_model.pkl')

st.title("💓 HeartSense - Heart Disease Risk Predictor")

st.sidebar.header("Enter Your Health Details")

def user_input():
    age = st.sidebar.slider('Age', 20, 80, 45)
    sex = st.sidebar.selectbox('Sex (1 = male, 0 = female)', [1, 0])
    cp = st.sidebar.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting BP', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar >120', [1, 0])
    restecg = st.sidebar.selectbox('Resting ECG (0–2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise-Induced Angina', [1, 0])
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope (0–2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Major Vessels (0–4)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thal (1,2,3)', [1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()
st.subheader("Your Input:")
st.write(input_df)

input_df = pd.get_dummies(input_df)

model_features = model.feature_names_in_  
missing_cols = [col for col in model_features if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0  

input_df = input_df[model_features]

prediction = model.predict(input_df)
risk_score = model.predict_proba(input_df)[0][1] * 100

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("You are likely at risk of heart disease.")
    st.write(f"Risk Score: **{risk_score:.2f}%**")
    st.markdown("- Risk factors: age, cholesterol, chest pain type")
    st.info("Please consult a doctor for confirmation.")
else:
    st.success("You are likely not at risk.")
    st.write(f"Risk Score: **{risk_score:.2f}%**")
