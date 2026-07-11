import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

.block-container{
padding-top:2rem;
}

.main-card{
background:#1e293b;
padding:30px;
border-radius:20px;
box-shadow:0px 0px 25px rgba(0,255,255,0.2);
}

.prediction-box{
background:linear-gradient(90deg,#00c853,#00e676);
padding:20px;
border-radius:15px;
font-size:28px;
font-weight:bold;
text-align:center;
color:white;
margin-top:20px;
}

.stButton>button{
width:100%;
height:55px;
font-size:20px;
font-weight:bold;
background:linear-gradient(90deg,#ff9800,#ff5722);
color:white;
border:none;
border-radius:10px;
}

.stButton>button:hover{
background:linear-gradient(90deg,#ff5722,#ff9800);
transform:scale(1.02);
}

div[data-baseweb="select"]{
background:#ffffff;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("disease_prediction_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load dataset to get symptom names
train = pd.read_csv("Training.csv")
train = train.loc[:, ~train.columns.str.contains("^Unnamed")]

symptoms = train.drop("prognosis", axis=1).columns

# Title
st.markdown("""
<h1 style='text-align:center;
color:#00E5FF;
font-size:48px;'>
🩺 AI Disease Prediction System
</h1>

<p style='text-align:center;
font-size:20px;
color:#DDDDDD;'>
Predict possible diseases instantly using Machine Learning
</p>
""", unsafe_allow_html=True)


st.markdown("<div class='main-card'>", unsafe_allow_html=True)


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


st.markdown("</div>", unsafe_allow_html=True)
