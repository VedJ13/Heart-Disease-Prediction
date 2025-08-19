import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

# User input
input_df = user_input()
st.subheader("Your Input:")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
risk_score = model.predict_proba(input_df)[0][1] * 100

# ---------- Feature 1: Risk Level Categorization ----------
def get_risk_level(score):
    if score < 40:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "High Risk"

risk_level = get_risk_level(risk_score)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ You are likely at risk of heart disease.")
else:
    st.success("✅ You are likely not at risk.")
st.write(f"**Risk Score:** {risk_score:.2f}% ({risk_level})")

# ---------- Feature 2: Important Feature Highlights ----------
st.subheader("Key Risk Factors")
try:
    coef = model.coef_[0]
    feature_importance = pd.Series(coef, index=input_df.columns).sort_values(key=abs, ascending=False)
    st.write("Top contributing features:")
    st.write(feature_importance.head(3))
except:
    st.info("Feature importance is only available for linear models (Logistic Regression).")

# ---------- Feature 3: Visual Risk Dashboard ----------
st.subheader("Your Health Dashboard")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Cholesterol comparison
ax[0].bar(["Your Chol", "Healthy Max"], [input_df['chol'][0], 200], color=['red', 'green'])
ax[0].set_title("Cholesterol (mg/dl)")

# Blood Pressure comparison
ax[1].bar(["Your BP", "Healthy Max"], [input_df['trestbps'][0], 130], color=['red', 'green'])
ax[1].set_title("Resting BP (mmHg)")

st.pyplot(fig)

# ---------- Feature 4: Downloadable Report ----------
st.subheader("Download Your Report")
report = input_df.copy()
report["Risk Score (%)"] = risk_score
report["Prediction"] = "At Risk" if prediction[0] == 1 else "Not at Risk"
st.download_button("📥 Download as CSV", report.to_csv(index=False).encode('utf-8'), "HeartSense_Report.csv", "text/csv")

# ---------- Feature 5: Health Tips ----------
st.subheader("Health Tips")
if risk_level == "Low Risk":
    st.success("👍 Maintain a balanced diet, exercise regularly, and keep monitoring your health.")
elif risk_level == "Medium Risk":
    st.warning("⚠️ Consider regular health check-ups, monitor cholesterol, and maintain a healthy lifestyle.")
else:
    st.error("🚨 Immediate consultation with a doctor is recommended. Focus on reducing cholesterol and BP.")

# ---------- Feature 6: Future Scope ----------
st.subheader("Future Enhancements")
st.markdown("""
- 📊 Integration with wearable devices (Fitbit, Smartwatches) for real-time monitoring  
- 🌐 Connection with hospital EMR systems  
- 🤖 Use of advanced models (Random Forest, Neural Networks) for better accuracy  
- 📱 Mobile app version for easy accessibility  
""")
