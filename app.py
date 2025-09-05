from operator import is_
import numpy as np
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf


# Load the pre-trained model
model = tf.keras.models.load_model('salary_regression_model.h5')

# Load the scaler and encoders
with open("onehot_encoder.pkl", "rb") as f:
    ohe_geography = pickle.load(f)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
# Start streamlit app
# center the heading

st.title("Customer Estimated Salary Prediction")

# Get user input
geography = st.selectbox("Geography", ohe_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure", min_value=0, max_value=10, value=1)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Exited", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode Geography
geography_encoded = ohe_geography.transform([[geography]])
geography_df = pd.DataFrame(geography_encoded, columns=ohe_geography.get_feature_names_out(['Geography']))


# combine all input features
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

# scale numerical features
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
pradiction_probability = prediction[0][0]


# result
st.write("### Prediction Result")
st.write(f"The estimated salary for the customer is: ${pradiction_probability:.2f}")