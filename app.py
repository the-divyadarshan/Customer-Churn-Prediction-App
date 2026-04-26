import streamlit as st
import pandas as pd
import joblib

# Load model and model columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Enter a few customer details and predict whether the customer is likely to churn or stay.")

st.markdown("---")

# =========================
# BASIC USER FRIENDLY INPUTS
# =========================
st.subheader("🧾 Basic Customer Details")

tenure = st.slider("Tenure (Months)", 0, 72, 12)

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

MonthlyCharges = st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=200.0, value=70.0)

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

TotalCharges = MonthlyCharges * tenure

st.markdown("---")

# =========================
# ADVANCED OPTIONS (HIDDEN)
# =========================
show_advanced = st.checkbox("⚙ Show Advanced Options (Optional)")

if show_advanced:
    st.subheader("⚙ Advanced Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

else:
    # Default values if advanced options not selected
    gender = "Male"
    SeniorCitizen = 0
    Partner = "No"
    Dependents = "No"

    PhoneService = "Yes"
    MultipleLines = "No"

    OnlineBackup = "No"
    DeviceProtection = "No"

    StreamingTV = "No"
    StreamingMovies = "No"

# =========================
# CREATE INPUT DATAFRAME
# =========================
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

# =========================
# ENCODING INPUT DATA
# =========================
input_encoded = pd.get_dummies(input_data)

# Add missing columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[model_columns]

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    churn_percentage = probability * 100

    # Risk Category
    if churn_percentage < 35:
        risk_level = "LOW RISK 🟢"
    elif churn_percentage < 70:
        risk_level = "MEDIUM RISK 🟠"
    else:
        risk_level = "HIGH RISK 🔴"

    st.write(f"📌 **Churn Probability:** {churn_percentage:.2f}%")
    st.write(f"⚠ **Risk Level:** {risk_level}")

    st.progress(probability)

    if prediction == 1:
        st.error("⚠ Customer is likely to CHURN (Leave the company)")
        st.warning("💡 Suggestion: Offer discounts, better plan, or customer support to retain this customer.")
    else:
        st.success("✅ Customer is likely to STAY with the company")
        st.info("💡 Suggestion: Maintain good service quality and offer loyalty rewards.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Customer Churn Prediction ML App")
