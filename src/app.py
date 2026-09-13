import streamlit as st
import pandas as pd
import sys

sys.path.append("churn_project/src")

from model_utils import predict_customer_churn


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Enterprise Customer Churn Prediction")
st.markdown(
    "### AI-powered customer churn risk assessment"
)

st.divider()


st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)


avg_monthly_spend = total_charges / (tenure + 1)

service_columns = [
    online_security,
    online_backup,
    device_protection,
    tech_support
]

total_services_active = sum(
    value == "Yes" for value in service_columns
)


customer_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "AvgMonthlySpend": avg_monthly_spend,
    "TotalServicesActive": total_services_active
}])


st.subheader("Customer Prediction")

if st.button("🔍 Predict Churn Risk", use_container_width=True):

    result = predict_customer_churn(customer_data)

    probability = result["churn_probability"] * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

    with col2:
        st.metric(
            "Prediction",
            result["churn_prediction"]
        )

    with col3:
        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    st.divider()

    st.subheader("Recommended Action")

    if result["risk_level"] == "High Risk":
        st.error(result["recommended_action"])
    elif result["risk_level"] == "Medium Risk":
        st.warning(result["recommended_action"])
    else:
        st.success(result["recommended_action"])

    with st.expander("View Customer Input"):
        st.dataframe(
            customer_data,
            use_container_width=True
        )


st.divider()

st.caption(
    "Enterprise Customer Churn Prediction | "
    "Machine Learning Portfolio Project"
)
