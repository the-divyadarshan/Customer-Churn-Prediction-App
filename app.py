import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Fill the customer details below and click Predict to know if the customer will churn.")

st.sidebar.header("Enter Customer Details")

# Sidebar Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)

# TotalCharges calculation
TotalCharges = MonthlyCharges * tenure

# Create input dataframe
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

st.subheader("📌 Customer Input Summary")
st.dataframe(input_data)

# Convert input data into dummy variables
input_encoded = pd.get_dummies(input_data)

# Add missing columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure same column order
input_encoded = input_encoded[model_columns]

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠ Customer is likely to CHURN (Leave the company)")
        st.write(f"📌 Churn Probability: **{probability*100:.2f}%**")
        st.warning("💡 Suggestion: Give discount / better plan / customer support to retain.")
    else:
        st.success("✅ Customer is likely to STAY with the company")
        st.write(f"📌 Churn Probability: **{probability*100:.2f}%**")
        st.info("💡 Suggestion: Maintain service quality and offer loyalty rewards.")