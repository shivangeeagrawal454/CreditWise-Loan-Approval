import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Load the 5 saved artifacts
model = joblib.load("creditwise_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("onehot_encoder.pkl")
edu_le = joblib.load("edu_encoder.pkl")
model_columns = joblib.load("feature_columns.pkl")

ohe_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

# 2. Page Setup
st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦")
st.title("🏦 CreditWise Intelligent Loan Approval")
st.write(
    "Enter applicant details below to predict loan eligibility using our"
    " trained Naive Bayes model."
)

# 3. Input Fields (Categories pulled directly from your fitted encoders!)
col1, col2 = st.columns(2)

with col1:
  applicant_income = st.number_input(
      "Applicant Monthly Income", min_value=0.0, value=10500.0, step=500.0
  )
  coapplicant_income = st.number_input(
      "Co-applicant Monthly Income", min_value=0.0, value=5000.0, step=500.0
  )
  age = st.slider("Age", min_value=18, max_value=75, value=35)
  dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
  credit_score = st.number_input(
      "Credit Score", min_value=300, max_value=900, value=680
  )
  existing_loans = st.number_input(
      "Existing Running Loans", min_value=0, max_value=10, value=1
  )
  dti_ratio = st.slider(
      "Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.34
  )
  savings = st.number_input(
      "Savings Balance", min_value=0.0, value=10000.0, step=500.0
  )

with col2:
  collateral_value = st.number_input(
      "Collateral Value", min_value=0.0, value=25000.0, step=1000.0
  )
  loan_amount = st.number_input(
      "Loan Amount Requested", min_value=500.0, value=20000.0, step=1000.0
  )
  loan_term = st.selectbox(
      "Loan Duration (Months)", [12, 24, 36, 48, 60, 72, 84], index=3
  )

  # Exact categories extracted from your fitted OneHotEncoder & LabelEncoder
  employment_status = st.selectbox("Employment Status", ohe.categories_[0])
  marital_status = st.selectbox("Marital Status", ohe.categories_[1])
  loan_purpose = st.selectbox("Loan Purpose", ohe.categories_[2])
  property_area = st.selectbox("Property Area", ohe.categories_[3])
  gender = st.selectbox("Gender", ohe.categories_[4])
  employer_category = st.selectbox("Employer Category", ohe.categories_[5])
  education_level = st.selectbox("Education Level", edu_le.classes_)

# 4. Predict Button
if st.button("Check Loan Eligibility", type="primary"):
  # Build single-row DataFrame from user inputs
  input_data = pd.DataFrame([{
      "Applicant_Income": applicant_income,
      "Coapplicant_Income": coapplicant_income,
      "Employment_Status": employment_status,
      "Age": float(age),
      "Marital_Status": marital_status,
      "Dependents": float(dependents),
      "Credit_Score": float(credit_score),
      "Existing_Loans": float(existing_loans),
      "DTI_Ratio": float(dti_ratio),
      "Savings": savings,
      "Collateral_Value": collateral_value,
      "Loan_Amount": loan_amount,
      "Loan_Term": float(loan_term),
      "Loan_Purpose": loan_purpose,
      "Property_Area": property_area,
      "Education_Level": education_level,
      "Gender": gender,
      "Employer_Category": employer_category,
  }])

  # --- STEP A: Label Encode Education_Level (Cell [57]) ---
  input_data["Education_Level"] = edu_le.transform(
      input_data["Education_Level"]
  )

  # --- STEP B: One-Hot Encode the 6 categorical columns (Cell [60]) ---
  encoded = ohe.transform(input_data[ohe_cols])
  encoded_df = pd.DataFrame(
      encoded, columns=ohe.get_feature_names_out(ohe_cols), index=input_data.index
  )
  input_data = pd.concat([input_data.drop(columns=ohe_cols), encoded_df], axis=1)

  # --- STEP C: Feature Engineering & Dropping Original Columns (Cell [94]) ---
  input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"] ** 2
  input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2
  input_data = input_data.drop(columns=["Credit_Score", "DTI_Ratio"])

  # --- STEP D: Enforce Exact Column Order & Scale ---
  input_aligned = input_data[model_columns]
  input_scaled = scaler.transform(input_aligned)

  # --- STEP E: Predict ---
  prediction = model.predict(input_scaled)[0]
  probability = model.predict_proba(input_scaled)[0][1]

  st.divider()
  if prediction == 1:
    st.success(f"✅ **Loan Approved!** (Approval Probability: {probability:.1%})")
  else:
    st.error(f"❌ **Loan Rejected.** (Approval Probability: {probability:.1%})")