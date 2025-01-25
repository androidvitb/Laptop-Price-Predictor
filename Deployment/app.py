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

# Page Title
st.title("💻Laptop Price Prediction 💻")
st.sidebar.header("Input Features")

company = st.sidebar.selectbox("Company", ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
                                            'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Mediacom',
                                            'Samsung', 'Google', 'Fujitsu', 'Razer', 'LG'])
type_name = st.sidebar.selectbox("Type", ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible','Workstation'])
cpu_brand = st.sidebar.selectbox("CPU Brand", ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3','Other Intel Processor'])
gpu_brand = st.sidebar.selectbox("GPU Brand", ['Intel', 'AMD', 'Nvidia'])
os = st.sidebar.selectbox("Operating System", ['Mac', 'Others/No OS/Linux', 'Windows'])

ram = st.sidebar.slider("RAM (GB)", 2, 64, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
ppi = st.sidebar.number_input("PPI (Pixels per Inch)", min_value=80.0, max_value=400.0, value=120.0, step=10.0)

touchscreen = st.sidebar.selectbox("Touchscreen", ["No", "Yes"])
ips = st.sidebar.selectbox("IPS Display", ["No", "Yes"])
hdd = st.sidebar.number_input("HDD", min_value=0, max_value=5000, value=512, step=1)
ssd = st.sidebar.number_input("SSD", min_value=0, max_value=5000, value=512, step=1)


# Convert categorical data to numeric
input_data = {
    "Company": company,
    "TypeName": type_name,
    "CPU Brand": cpu_brand,
    "GPU Brand": gpu_brand,
    "OS": os,
    "RAM": ram,
    "Weight": weight,
    "PPI": ppi,
    "TouchScreen": 1 if touchscreen == "Yes" else 0,
    "IPS": 1 if ips == "Yes" else 0,
    "HDD": hdd,
    "SSD": ssd,
}

# Prepare data for prediction
input_array = np.array([
    input_data["Company"],
    input_data["TypeName"],
    input_data["RAM"],
    input_data["Weight"],
    input_data["TouchScreen"],
    input_data["IPS"],
    input_data["PPI"],
    input_data["CPU Brand"],
    input_data["HDD"],
    input_data["SSD"],
    input_data["GPU Brand"],
    input_data["OS"],
    ])

st.divider()
df = pd.DataFrame([input_data])
st.table(df)
st.divider()


#Predict button
if st.button("Predict Price"):
    try:
        loaded_encoder=load_encoder()
        loaded_scaler=load_scaler()
        encoded_data = loaded_encoder.transform(input_array[[0,1,7,10,11]].reshape(1, -1))
        num=input_array[[2,3,4,5,6,8,9]]
        X=np.hstack([encoded_data[0],num])
        scaled_data = loaded_scaler.transform([X])
        model=Load_Model()
        prediction=np.e**(model.predict(scaled_data))
        st.success(f"Predicted Laptop Price:   {prediction[0][0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
