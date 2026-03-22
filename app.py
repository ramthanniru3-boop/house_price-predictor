import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# LOAD FILES
# ==========================
base_path = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_path, "model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
columns = joblib.load(os.path.join(base_path, "columns.pkl"))

# ==========================
# UI
# ==========================
st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Prediction + Explainable AI")

# ==========================
# INPUTS
# ==========================
GrLivArea = st.number_input("Living Area", value=1000)
BedroomAbvGr = st.number_input("Bedrooms", value=3)
FullBath = st.number_input("Bathrooms", value=2)
YearBuilt = st.number_input("Year Built", value=2000)
GarageCars = st.number_input("Garage Capacity", value=2)

# ==========================
# CREATE INPUT DATA
# ==========================
input_dict = {
    "GrLivArea": GrLivArea,
    "BedroomAbvGr": BedroomAbvGr,
    "FullBath": FullBath,
    "YearBuilt": YearBuilt,
    "GarageCars": GarageCars
}

input_df = pd.DataFrame([input_dict])

# Add missing columns
for col in columns:
    if col not in input_df:
        input_df[col] = 0

# Arrange columns
input_df = input_df[columns]

# Scale
input_scaled = scaler.transform(input_df)

# ==========================
# PREDICTION
# ==========================
if st.button("Predict Price"):

    prediction = model.predict(input_scaled)
    price = np.exp(prediction[0])

    st.success(f"💰 Predicted Price: ₹ {price:,.2f}")

    # ==========================
    # SHAP (SAFE MODE)
    # ==========================
    st.subheader("🔍 Why this prediction? (SHAP)")

    try:
        import shap
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge, LinearRegression, Lasso

        # Select correct explainer
        if isinstance(model, (Ridge, LinearRegression, Lasso)):
            explainer = shap.LinearExplainer(model, input_scaled)
            shap_values = explainer(input_scaled)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_scaled)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception:
        st.warning("⚠️ SHAP not available in cloud environment")