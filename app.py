import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("final_churn_model.pkl", "rb"))

st.title("Customer Churn Prediction")
st.subheader("Customer Details")

# numeric inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 50.0)
total = st.slider("Total Charges", 0.0, 10000.0, 500.0)

# categorical inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phoneservice = st.selectbox("Phone Service", ["No", "Yes"])

if st.button("Predict"):

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}
    yes_no = {"No": 0, "Yes": 1}

    data = np.zeros((1, model.n_features_in_))

    data[0][4] = tenure
    data[0][-2] = monthly
    data[0][-1] = total

    data[0][14] = contract_map[contract]
    data[0][7] = internet_map[internet]
    data[0][15] = yes_no[paperless]
    data[0][1] = yes_no[senior]
    data[0][2] = yes_no[partner]
    data[0][3] = yes_no[dependents]
    data[0][5] = yes_no[phoneservice]
    data[0][16] = payment_map[payment]

    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("Customer Likely to Churn")
    else:
        st.success("Customer Likely to Stay")

    st.metric("Churn Probability", f"{prob:.2f}")

    # Risk level
    if prob > 0.7:
        st.write("Risk Level: High")
    elif prob > 0.4:
        st.write("Risk Level: Medium")
    else:
        st.write("Risk Level: Low")

    # -------- REASONS --------
    st.subheader("Why this prediction?")

    reasons = []

    if tenure < 12:
        reasons.append("Customer has low tenure (new customer)")

    if monthly > 80:
        reasons.append("High monthly charges")

    if contract == "Month-to-month":
        reasons.append("Month-to-month contract increases churn risk")

    if internet == "Fiber optic":
        reasons.append("Fiber optic users tend to churn more")

    if paperless == "Yes":
        reasons.append("Paperless billing associated with higher churn")

    if senior == "Yes":
        reasons.append("Senior citizens show slightly higher churn")

    if payment == "Electronic check":
        reasons.append("Electronic check payment linked with churn")

    if len(reasons) == 0:
        st.write("Customer shows stable retention indicators")
    else:
        for r in reasons:
            st.write("•", r)