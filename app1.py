import pandas as pd

startup = pd.read_csv(r"C:\Users\areeb\Desktop\S5\ML\datasets\50_Startups (1).csv")
startup = pd.get_dummies(startup, columns=['State'], drop_first=True)  # as state is a categprical variable and the system dont read tha 


print(startup.isnull().sum())




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Features and target
X = startup.drop('Profit', axis=1)
y = startup['Profit']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "profit_model.pkl")



import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("profit_model.pkl")  # make sure this file is in the same folder

# Page settings
st.set_page_config(page_title="Profit Prediction App", layout="wide")

# Custom CSS for colors & buttons
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e2f;  /* dark purple/blue background */
        color: white;
    }
    h1 {
        color: #f1c40f;  /* bright yellow title */
        font-family: 'Comic Sans MS', cursive;
    }
    p {
        color: #ecf0f1;  /* light gray for paragraph */
        font-size: 18px;
    }
    .stButton>button {
        background-color: #ff6f61;  /* coral red button */
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1>💼 Startup Profit Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter your company details in the sidebar to get an instant profit prediction 📈</p>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Company Details")
rd = st.sidebar.number_input("💡 R&D Spend", min_value=0.0, step=1000.0)
admin = st.sidebar.number_input("🗂️ Administration Spend", min_value=0.0, step=1000.0)
marketing = st.sidebar.number_input("📢 Marketing Spend", min_value=0.0, step=1000.0)
state = st.sidebar.selectbox("🌍 State", ["New York", "California", "Florida"])

# One-hot encode state
state_dict = {"New York": [0,0], "California": [1,0], "Florida": [0,1]}
state_encoded = state_dict[state]

# Combine features into array
features = [rd, admin, marketing] + state_encoded
features = np.array([features])

# Predict button
if st.button("🚀 Predict Profit"):
    prediction = model.predict(features)
    st.success(f"✅ Predicted Profit: **${prediction[0]:,.2f}**")
    st.balloons()
    st.info("“Grow your wealth smartly for greater returns!”!")
