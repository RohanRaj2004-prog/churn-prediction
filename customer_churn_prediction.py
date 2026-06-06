import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📱 Telecom Customer Churn Prediction")
st.write("Enter customer details to predict if they will leave the service.")

# 2. Models aur Encoders load karo
@st.cache_resource
def load_assets():
    with open("customer_churn.pkl","rb") as m_file:
        loaded_data = pickle.load(m_file)
    if isinstance(loaded_data, dict):
        model = loaded_data.get('model',loaded_data)    
    else:
        model = loaded_data
    with open('encoder.pkl','rb') as e_file:
        encoders = pickle.load(e_file)

    return model,encoders        
try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading model/encoders: {e}")

# 3. User Input Form (UI Elements)
st.header("Customer Demographics & Services")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("SeniorCitizen (0=No, 1=Yes)", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.header("Contract & Billing")
col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col4:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

# 4. Data Processing & Prediction

   # # 4. Data Processing & Prediction
if st.button("Predict Churn Status", type="primary"):
    # Input dictionary banao (Variables aur keys dono sahi hain ab)
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,   # Pehle yahan galat variable tha
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,     # Pehle yahan space ya caps ka issue tha
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Dataframe banao
    input_df = pd.DataFrame([input_data])

    # Encoders apply karo (Tumhara fixed safe logic)
    for column, encoder in encoders.items():
        if column in input_df.columns and column != 'SeniorCitizen':  # SeniorCitizen ko encode nahi karna kyunki wo 0/1 hai
            try:
                val = str(input_df.loc[0, column]).strip()
                input_df[column] = encoder.transform([val])[0]
            except Exception as e:
                pass

    # Feature names check karo model ke hisab se
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]

    # Final Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **Warning:** This customer is likely to **CHURN**! (Probability: {probability:.2f})")
    else:
        st.success(f"🎉 **Good News:** This customer is likely to **STAY**! (Probability of Churn: {probability:.2f})")