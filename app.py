import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

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

# Collect user input
input_df = user_input()
st.subheader("Your Input:")
st.write(input_df)

# Align input features with model
input_df = pd.get_dummies(input_df)
model_features = model.feature_names_in_
missing_cols = [col for col in model_features if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[model_features]

# Predictions
prediction = model.predict(input_df)
risk_score = model.predict_proba(input_df)[0][1] * 100

# Risk Categorization
def categorize_risk(score):
    if score < 30:
        return "Low Risk"
    elif 30 <= score < 60:
        return "Moderate Risk"
    else:
        return "High Risk"

risk_level = categorize_risk(risk_score)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ You are likely at risk of heart disease.")
else:
    st.success("✅ You are likely not at risk.")

st.write(f"Risk Score: **{risk_score:.2f}%**")
st.write(f"Risk Level: **{risk_level}**")



st.subheader("📊 Your Health Dashboard")

# Cholesterol Gauge
cholesterol_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = input_df['chol'][0],
    title = {'text': "Cholesterol (mg/dL)"},
    gauge = {
        'axis': {'range': [0, 400]},
        'bar': {'color': "red" if input_df['chol'][0] > 200 else "green"},
        'steps': [
            {'range': [0, 200], 'color': "lightgreen"},
            {'range': [200, 400], 'color': "lightcoral"}
        ]
    }
))

# Blood Pressure Gauge
bp_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = input_df['trestbps'][0],
    title = {'text': "Resting BP (mmHg)"},
    gauge = {
        'axis': {'range': [0, 200]},
        'bar': {'color': "red" if input_df['trestbps'][0] > 130 else "green"},
        'steps': [
            {'range': [0, 130], 'color': "lightgreen"},
            {'range': [130, 200], 'color': "lightcoral"}
        ]
    }
))

# Show in Streamlit
st.plotly_chart(cholesterol_gauge, use_container_width=True)
st.plotly_chart(bp_gauge, use_container_width=True)





# --- Healthy Ranges Reference ---
healthy_ranges = {
    "age": (20, 50),              # ideal working-age range (example, not medical advice)
    "trestbps": (90, 120),        # normal resting blood pressure (mm Hg)
    "chol": (125, 200),           # desirable cholesterol (mg/dL)
    "thalach": (100, 170),        # normal max heart rate range (varies with age)
    "oldpeak": (0.0, 1.0)         # normal ST depression
}

# --- Comparison Chart ---
st.subheader("📉 Your Values vs Healthy Ranges")

# Prepare data
compare_data = []
for feature, (low, high) in healthy_ranges.items():
    user_value = input_df.iloc[0][feature] if feature in input_df.columns else None
    compare_data.append({
        "Feature": feature,
        "User Value": user_value,
        "Healthy Min": low,
        "Healthy Max": high
    })

compare_df = pd.DataFrame(compare_data)

# Plot
fig2, ax2 = plt.subplots(figsize=(6,4))
for i, row in compare_df.iterrows():
    ax2.plot([row["Healthy Min"], row["Healthy Max"]], [i, i], color="green", linewidth=5, label="Healthy Range" if i==0 else "")
    ax2.scatter(row["User Value"], i, color="red", zorder=5, label="Your Value" if i==0 else "")

ax2.set_yticks(range(len(compare_df)))
ax2.set_yticklabels(compare_df["Feature"])
ax2.set_xlabel("Value")
ax2.set_title("Comparison of User Inputs with Healthy Ranges")
ax2.legend()
st.pyplot(fig2)

st.dataframe(compare_df)  # Optional table for clarity

# Visual Risk Dashboard
st.subheader("📊 Visual Risk Dashboard")
fig, ax = plt.subplots()
ax.bar(["Risk Probability"], [risk_score], color="red" if risk_score >= 50 else "green")
ax.set_ylim([0, 100])
ax.set_ylabel("Risk %")
st.pyplot(fig)

# Health Tips
st.subheader("💡 Health Tips")
if risk_level == "Low Risk":
    st.info("Maintain a healthy lifestyle with regular exercise, balanced diet, and routine checkups.")
elif risk_level == "Moderate Risk":
    st.warning("Consider improving diet, managing stress, and consulting a doctor for preventive measures.")
else:
    st.error("Seek medical advice immediately. Lifestyle changes and regular monitoring are strongly recommended.")

# Downloadable Report
st.subheader("📥 Download Your Report")
report = f"""
HeartSense - Heart Disease Prediction Report

Risk Score: {risk_score:.2f}%
Risk Level: {risk_level}

Health Tips:
- {('Maintain a healthy lifestyle.' if risk_level=='Low Risk' else 'Consider preventive care and consult a doctor.' if risk_level=='Moderate Risk' else 'Seek medical help immediately.')}
"""

buffer = BytesIO()
buffer.write(report.encode())
st.download_button(
    label="Download Report",
    data=buffer,
    file_name="HeartSense_Report.txt",
    mime="text/plain"
)





















# import streamlit as st
# import pandas as pd
# import joblib

# # Load model
# model = joblib.load('heartsense_model.pkl')

# st.title("💓 HeartSense - Heart Disease Risk Predictor")

# st.sidebar.header("Enter Your Health Details")

# def user_input():
#     age = st.sidebar.slider('Age', 20, 80, 45)
#     sex = st.sidebar.selectbox('Sex (1 = male, 0 = female)', [1, 0])
#     cp = st.sidebar.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
#     trestbps = st.sidebar.slider('Resting BP', 80, 200, 120)
#     chol = st.sidebar.slider('Cholesterol', 100, 400, 200)
#     fbs = st.sidebar.selectbox('Fasting Blood Sugar >120', [1, 0])
#     restecg = st.sidebar.selectbox('Resting ECG (0–2)', [0, 1, 2])
#     thalach = st.sidebar.slider('Max Heart Rate', 70, 210, 150)
#     exang = st.sidebar.selectbox('Exercise-Induced Angina', [1, 0])
#     oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0)
#     slope = st.sidebar.selectbox('Slope (0–2)', [0, 1, 2])
#     ca = st.sidebar.selectbox('Major Vessels (0–4)', [0, 1, 2, 3, 4])
#     thal = st.sidebar.selectbox('Thal (1,2,3)', [1, 2, 3])

#     data = {
#         'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
#         'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
#         'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
#     }

#     return pd.DataFrame(data, index=[0])

# input_df = user_input()
# st.subheader("Your Input:")
# st.write(input_df)

# input_df = pd.get_dummies(input_df)

# model_features = model.feature_names_in_  
# missing_cols = [col for col in model_features if col not in input_df.columns]
# for col in missing_cols:
#     input_df[col] = 0  

# input_df = input_df[model_features]

# prediction = model.predict(input_df)
# risk_score = model.predict_proba(input_df)[0][1] * 100

# st.subheader("Prediction Result")
# if prediction[0] == 1:
#     st.error("You are likely at risk of heart disease.")
#     st.write(f"Risk Score: **{risk_score:.2f}%**")
#     st.markdown("- Risk factors: age, cholesterol, chest pain type")
#     st.info("Please consult a doctor for confirmation.")
# else:
#     st.success("You are likely not at risk.")
#     st.write(f"Risk Score: **{risk_score:.2f}%**")


# ---------- Feature 1: Risk Level Categorization ----------
# def get_risk_level(score):
#     if score < 40:
#         return "Low Risk"
#     elif score < 70:
#         return "Medium Risk"
#     else:
#         return "High Risk"

# risk_level = get_risk_level(risk_score)

# st.subheader("Prediction Result")
# if prediction[0] == 1:
#     st.error("⚠️ You are likely at risk of heart disease.")
# else:
#     st.success("✅ You are likely not at risk.")
# st.write(f"**Risk Score:** {risk_score:.2f}% ({risk_level})")

# # ---------- Feature 2: Important Feature Highlights ----------
# st.subheader("Key Risk Factors")
# try:
#     coef = model.coef_[0]
#     feature_importance = pd.Series(coef, index=input_df.columns).sort_values(key=abs, ascending=False)
#     st.write("Top contributing features:")
#     st.write(feature_importance.head(3))
# except:
#     st.info("Feature importance is only available for linear models (Logistic Regression).")

# # ---------- Feature 3: Visual Risk Dashboard ----------
# st.subheader("Your Health Dashboard")
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# # Cholesterol comparison
# ax[0].bar(["Your Chol", "Healthy Max"], [input_df['chol'][0], 200], color=['red', 'green'])
# ax[0].set_title("Cholesterol (mg/dl)")

# # Blood Pressure comparison
# ax[1].bar(["Your BP", "Healthy Max"], [input_df['trestbps'][0], 130], color=['red', 'green'])
# ax[1].set_title("Resting BP (mmHg)")

# st.pyplot(fig)

# # ---------- Feature 4: Downloadable Report ----------
# st.subheader("Download Your Report")
# report = input_df.copy()
# report["Risk Score (%)"] = risk_score
# report["Prediction"] = "At Risk" if prediction[0] == 1 else "Not at Risk"
# st.download_button("📥 Download as CSV", report.to_csv(index=False).encode('utf-8'), "HeartSense_Report.csv", "text/csv")

# # ---------- Feature 5: Health Tips ----------
# st.subheader("Health Tips")
# if risk_level == "Low Risk":
#     st.success("👍 Maintain a balanced diet, exercise regularly, and keep monitoring your health.")
# elif risk_level == "Medium Risk":
#     st.warning("⚠️ Consider regular health check-ups, monitor cholesterol, and maintain a healthy lifestyle.")
# else:
#     st.error("🚨 Immediate consultation with a doctor is recommended. Focus on reducing cholesterol and BP.")
# 
