import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Disease Predictor",
    layout="wide"
)


with st.sidebar:

    st.image(
        "https://img.icons8.com/color/240/stethoscope.png",
        width=120
    )

    st.title("Disease Predictor")

    st.write("""
This application predicts possible diseases based on selected symptoms.

**Steps**

✔ Select symptoms

✔ Click Predict

✔ View result

⚠ This is an educational project only.
""")
    
st.set_page_config(
    page_title="AI Disease Predictor",
    layout="wide"
)

st.markdown("""
<style>

/* Main Background */
.stApp{
    background:#F4F8FB;
}

/* Main Container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#0F4C81;
    margin-bottom:10px;
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:#5B6770;
    font-size:20px;
    margin-bottom:30px;
}

/* Card */
.card{
    background:white;
    padding:30px;
    border-radius:15px;
    box-shadow:0 8px 25px rgba(0,0,0,0.08);
}

/* Predict Button */
.stButton>button{
    width:100%;
    height:55px;
    background:#0F4C81;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1565C0;
}

/* Success Box */
.prediction-box{
    background:#E8F5E9;
    border-left:8px solid #2E7D32;
    padding:20px;
    border-radius:10px;
    font-size:28px;
    font-weight:bold;
    color:#1B5E20;
    margin-top:20px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0F4C81;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Multiselect */
div[data-baseweb="select"]{
    border-radius:10px;
}

/* Footer */
.footer{
    text-align:center;
    color:#607D8B;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

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

# -------------------------------
# Disease Recommendations
# -------------------------------

recommendations = {

    "Fungal infection":[
        "🧼 Keep the affected area clean and dry.",
        "💊 Apply antifungal cream as prescribed.",
        "👕 Wear clean cotton clothes.",
        "👨‍⚕️ Consult a dermatologist if symptoms persist."
    ],

    "Allergy":[
        "🌸 Avoid dust, pollen and allergens.",
        "💧 Drink plenty of water.",
        "💊 Take antihistamines if prescribed.",
        "👨‍⚕️ Visit a doctor if symptoms worsen."
    ],

    "Common Cold":[
        "☕ Drink warm fluids.",
        "😴 Get enough rest.",
        "🍊 Increase Vitamin C intake.",
        "💊 Take medicines only if prescribed."
    ],

    "Malaria":[
        "🛏 Take complete rest.",
        "💧 Drink plenty of fluids.",
        "💊 Complete the prescribed medication.",
        "🏥 Visit a hospital immediately."
    ],

    "Typhoid":[
        "🥣 Eat soft and nutritious food.",
        "💧 Drink boiled water.",
        "💊 Finish the antibiotic course.",
        "👨‍⚕️ Consult a doctor."
    ]
}

# Title
st.markdown("""
<h1 style='text-align:center;
color:#00E5FF;
font-size:48px;'>
 Disease Prediction System
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
    " Select Symptoms",
    symptoms,
    placeholder="Choose one or more symptoms..."
)

progress = len(selected_symptoms) / len(symptoms)

st.progress(progress)

st.caption(f"Selected {len(selected_symptoms)} symptoms")

# Create input vector
input_data = pd.DataFrame(
    [[0]*len(symptoms)],
    columns=symptoms
)

# Set selected symptoms to 1
for symptom in selected_symptoms:
    input_data[symptom] = 1

# Predict
if st.button(" Predict Disease"):

    with st.spinner("Analyzing symptoms..."):

        prediction = model.predict(input_data)

        disease = encoder.inverse_transform(prediction)

    st.markdown(f"""
<div style="
background:rgba(255,255,255,0.08);
backdrop-filter:blur(12px);
border:1px solid rgba(255,255,255,0.15);
padding:30px;
border-radius:20px;
margin-top:25px;
box-shadow:0 8px 25px rgba(0,0,0,0.35);
">

<h2 style="color:#4FC3F7;">
🩺 Prediction Result
</h2>

<hr style="border:1px solid #3B82F6;">

<h1 style="color:#4CAF50;">
{disease[0]}
</h1>

<p style="font-size:18px;color:#D6E4F0;">
The AI model predicts that your symptoms most closely match
<b>{disease[0]}</b>.
</p>

</div>
""", unsafe_allow_html=True)







st.markdown("---")

st.markdown("""
<center>

Made with ❤️ using Streamlit & Scikit-Learn

</center>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
