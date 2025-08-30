import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO


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


def categorize_risk(score):
    if score > 30:
        return "Low Risk"
    elif 30 >= score > 60:
        return "Moderate Risk"
    else:
        return "High Risk"

risk_level = categorize_risk(risk_score)

st.subheader("Prediction Result")

if risk_level == "Low Risk":
    st.success("✅ You are at Low Risk of heart disease.")
    st.info("Keep maintaining a healthy lifestyle with regular exercise and balanced diet.")
elif risk_level == "Moderate Risk":
    st.warning("⚠️ You are at Moderate Risk of heart disease.")
    st.info("Consider consulting a doctor and adopting preventive measures.")
else:  # High Risk
    st.error("🚨 You are at High Risk of heart disease.")
    st.warning("It is strongly advised to consult a healthcare professional immediately.")

# st.write(f"Risk Score: **{risk_score:.2f}%**")
st.write(f"Risk Level: **{risk_level}**")


st.subheader("📊 Your Health Dashboard")


healthy_ranges = {
    'age': (20, 50),
    'trestbps': (80, 120),
    'chol': (150, 240),
    'thalach': (120, 180),
    'oldpeak': (0.0, 2.0)
}


user_values = {
    'age': int(input_df['age']),
    'trestbps': int(input_df['trestbps']),
    'chol': int(input_df['chol']),
    'thalach': int(input_df['thalach']),
    'oldpeak': float(input_df['oldpeak'])
}

fig = go.Figure()


for feature, (low, high) in healthy_ranges.items():
    fig.add_trace(go.Bar(
        x=[feature],
        y=[high - low],
        base=low,
        name=f"Healthy Range",
        marker_color='lightgreen',
        opacity=0.6
    ))


fig.add_trace(go.Scatter(
    x=list(user_values.keys()),
    y=list(user_values.values()),
    mode='markers+lines',
    name="Your Value",
    marker=dict(size=12, color='red', symbol='diamond')
))


fig.update_layout(
    title="🔍 Your Health Parameters vs Healthy Ranges",
    yaxis_title="Value",
    barmode='overlay',
    template="plotly_white",
    height=500
)

st.plotly_chart(fig)




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


st.plotly_chart(cholesterol_gauge, use_container_width=True)
st.plotly_chart(bp_gauge, use_container_width=True)




healthy_ranges = {
    "age": (20, 50),             
    "trestbps": (90, 120),        
    "chol": (125, 200),           
    "thalach": (100, 170),        
    "oldpeak": (0.0, 1.0)         
}


st.subheader("📉 Your Values vs Healthy Ranges")


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


fig2, ax2 = plt.subplots(figsize=(6,4))
for i, row in compare_df.iterrows():
    ax2.plot([row["Healthy Min"], row["Healthy Max"]], [i, i], color="green", linewidth=5, label="Healthy Range" if i==0 else "")
    ax2.scatter(row["User Value"], i, color="red", zorder=5, label="Your Value" if i==0 else "")


st.dataframe(compare_df)  


st.subheader("📊 Visual Risk Dashboard")
fig, ax = plt.subplots()
ax.bar(["Risk Probability"], [risk_score], color="red" if risk_score >= 50 else "green")
ax.set_ylim([0, 100])
ax.set_ylabel("Risk %")
st.pyplot(fig)


st.subheader("💡 Health Tips")
if risk_level == "Low Risk":
    st.info("Maintain a healthy lifestyle with regular exercise, balanced diet, and routine checkups.")
elif risk_level == "Moderate Risk":
    st.warning("Consider improving diet, managing stress, and consulting a doctor for preventive measures.")
else:
    st.error("Seek medical advice immediately. Lifestyle changes and regular monitoring are strongly recommended.")


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


