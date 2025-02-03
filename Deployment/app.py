import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def Load_Model():
    return load_model('my_model.keras')


@st.cache_resource
def load_encoder():
    with open('encoder.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# --- MAIN TITLE ---
st.markdown(
    "<h1 style='text-align: center; color: #3366cc;'>💻 Laptop Price Prediction 💻</h1>", 
    unsafe_allow_html=True
)
st.write("🚀 **Enter the laptop specifications to estimate its price.**")


# --- SIDEBAR INPUTS ---
st.sidebar.header("🔧 Laptop Features")

company = st.sidebar.selectbox("🏢 Brand", ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
                                            'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Mediacom',
                                            'Samsung', 'Google', 'Fujitsu', 'Razer', 'LG'])
type_name = st.sidebar.selectbox("💻 Type", ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
cpu_brand = st.sidebar.selectbox("🖥️ CPU", ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor'])
gpu_brand = st.sidebar.selectbox("🎮 GPU", ['Intel', 'AMD', 'Nvidia'])
os = st.sidebar.selectbox("🖥️ Operating System", ['Mac', 'Others/No OS/Linux', 'Windows'])

ram = st.sidebar.slider("📏 RAM (GB)", 2, 64, step=1)
weight = st.sidebar.number_input("⚖️ Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
ppi = st.sidebar.number_input("🖥️ PPI (Pixels per Inch)", min_value=80.0, max_value=400.0, value=120.0, step=10.0)

touchscreen = st.sidebar.radio("📱 Touchscreen", ["No", "Yes"])
ips = st.sidebar.radio("🎨 IPS Display", ["No", "Yes"])
hdd = st.sidebar.slider("🛠 HDD (GB)", min_value=0, max_value=5000, value=512, step=256)
ssd = st.sidebar.slider("⚡ SSD (GB)", min_value=0, max_value=5000, value=512, step=256)

# Convert categorical data to numeric
input_data = {
    "Company": company,
    "TypeName": type_name,
    "RAM": ram,
    "Weight": weight,
    "TouchScreen": 1 if touchscreen == "Yes" else 0,
    "IPS": 1 if ips == "Yes" else 0,
    "PPI": ppi,
    "CPU Brand": cpu_brand,
    "HDD": hdd,
    "SSD": ssd,
    "GPU Brand": gpu_brand,
    "OS": os,
}

# --- DISPLAY INPUT DATA ---
st.subheader("📝 Selected Features")
st.write("Here are the details of the laptop you have selected:")
df = pd.DataFrame([input_data])
st.table(df)

# --- PREDICTION ---
if st.button("🔮 Predict Price"):
    try:
        with st.spinner("⏳ Predicting the price... Please wait!"):
            # Load necessary components
            loaded_encoder = load_encoder()
            loaded_scaler = load_scaler()

            # Extract categorical features to be encoded
            categorical_features = np.array([
                input_data["Company"], 
                input_data["TypeName"], 
                input_data["CPU Brand"], 
                input_data["GPU Brand"], 
                input_data["OS"]
            ]).reshape(1, -1)

            # Apply encoding
            encoded_data = loaded_encoder.transform(categorical_features)

            # Extract numerical features
            numerical_features = np.array([
                input_data["RAM"], 
                input_data["Weight"], 
                input_data["TouchScreen"], 
                input_data["IPS"], 
                input_data["PPI"], 
                input_data["HDD"], 
                input_data["SSD"]
            ]).reshape(1, -1).astype(float)

            # Combine encoded and numerical features
            X = np.hstack([encoded_data, numerical_features])

            # Apply scaling
            scaled_data = loaded_scaler.transform(X)

            # Load model and make prediction
            model = Load_Model()
            prediction = np.exp(model.predict(scaled_data))  # Reverse log transformation

        st.success(f"💰 Predicted Laptop Price:   **${prediction[0][0]:,.2f}**")

    except Exception as e:
        st.error("🚨 An error occurred during prediction.")
        st.text(f"🔍 Debugging Info: {type(e).__name__} - {str(e)}")

# --- FOOTER ---
st.markdown("""
    <br><hr>
    <p style="text-align:center; color:gray;">
        Made with ❤️ by <b>AcWoc</b> | Powered by <b>Streamlit & TensorFlow</b>
    </p>
""", unsafe_allow_html=True)
