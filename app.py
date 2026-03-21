import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Prediction (Advanced)")

# ==========================
# INPUTS (IMPORTANT FEATURES)
# ==========================

GrLivArea = st.number_input("Living Area", value=1000)
BedroomAbvGr = st.number_input("Bedrooms", value=3)
FullBath = st.number_input("Full Bathrooms", value=2)
YearBuilt = st.number_input("Year Built", value=2000)
GarageCars = st.number_input("Garage Capacity", value=2)

# ==========================
# CREATE INPUT DATAFRAME
# ==========================
input_dict = {
    "GrLivArea": GrLivArea,
    "BedroomAbvGr": BedroomAbvGr,
    "FullBath": FullBath,
    "YearBuilt": YearBuilt,
    "GarageCars": GarageCars
}

input_df = pd.DataFrame([input_dict])

# Fill missing columns
for col in columns:
    if col not in input_df:
        input_df[col] = 0

# Arrange columns correctly
input_df = input_df[columns]

# Scale
input_scaled = scaler.transform(input_df)

# ==========================
# PREDICT
# ==========================
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    price = np.exp(prediction[0])
    
    st.success(f"💰 Predicted Price: ₹ {price:,.2f}")