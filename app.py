import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load files safely
base_path = os.path.dirname(__file__)

with open(os.path.join(base_path, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(base_path, "columns.pkl"), "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Prediction")

# Inputs
GrLivArea = st.number_input("Living Area", value=1000)
BedroomAbvGr = st.number_input("Bedrooms", value=3)
FullBath = st.number_input("Bathrooms", value=2)
YearBuilt = st.number_input("Year Built", value=2000)
GarageCars = st.number_input("Garage Capacity", value=2)

input_dict = {
    "GrLivArea": GrLivArea,
    "BedroomAbvGr": BedroomAbvGr,
    "FullBath": FullBath,
    "YearBuilt": YearBuilt,
    "GarageCars": GarageCars
}

input_df = pd.DataFrame([input_dict])

for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]
input_scaled = scaler.transform(input_df)

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    price = np.exp(prediction[0])
    st.success(f"💰 Price: ₹ {price:,.2f}")