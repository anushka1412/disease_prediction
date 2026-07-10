import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("disease_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load dataset to get symptom names
train = pd.read_csv("Training.csv")
train = train.loc[:, ~train.columns.str.contains("^Unnamed")]

symptoms = train.drop("prognosis", axis=1).columns

# Title
st.title("Disease Prediction System")

st.write("Select the symptoms you are experiencing.")

# Symptom selection
selected_symptoms = st.multiselect(
    "Symptoms",
    symptoms
)

# Create input vector
input_data = pd.DataFrame(
    [[0]*len(symptoms)],
    columns=symptoms
)

# Set selected symptoms to 1
for symptom in selected_symptoms:
    input_data[symptom] = 1

# Predict
if st.button("Predict Disease"):

    prediction = model.predict(input_data)

    disease = encoder.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease[0]}")
