import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
import traceback
import streamlit as st

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(str(e))
    st.code(traceback.format_exc())
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main-title {
    font-size:42px;
    font-weight:800;
    color:#1f4e79;
    text-align:center;
}

.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.card {
    padding:20px;
    border-radius:15px;
    background-color:#f7f9fc;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
}

.result-box {
    padding:25px;
    background:linear-gradient(90deg,#1f4e79,#3b82f6);
    color:white;
    text-align:center;
    border-radius:15px;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Get instant AI-powered laptop price estimation</div>', unsafe_allow_html=True)

st.divider()

# ---------------- LAYOUT ----------------
left, right = st.columns([2, 1])

# ================= LEFT SIDE (INPUTS) =================
with left:

    st.markdown("### 🧾 Laptop Specifications")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox("Brand", ["Dell","HP","Lenovo","Asus","Acer","Apple","MSI"])
        typename = st.selectbox("Laptop Type", ["Notebook","Gaming","Ultrabook","2 in 1 Convertible","Workstation"])
        cpu = st.selectbox("Processor", ["Intel Core i3","Intel Core i5","Intel Core i7","AMD Ryzen 5","AMD Ryzen 7"])

    with col2:
        ram = st.selectbox("RAM (GB)", [4,8,16,32])
        weight = st.slider("Weight (Kg)", 0.5, 4.5, 2.0)
        opsys = st.selectbox("Operating System", ["Windows","Mac","Linux","No OS"])

    st.markdown("### 🖥 Display Options")

    col3, col4 = st.columns(2)

    with col3:
        touchscreen = st.selectbox("Touchscreen", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

    with col4:
        ips = st.selectbox("IPS Display", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

    screen = st.selectbox(
        "Screen Resolution",
        ["1920x1080","1366x768","2560x1440","3840x2160"]
    )

    st.markdown("### 💾 Storage & Graphics")

    col5, col6 = st.columns(2)

    with col5:
        memory = st.selectbox("Storage", [
            "256GB SSD",
            "512GB SSD",
            "1TB SSD",
            "1TB HDD",
            "Hybrid"
        ])

    with col6:
        gpu = st.selectbox("GPU", [
            "Intel UHD Graphics",
            "NVIDIA GTX 1650",
            "NVIDIA RTX 3050",
            "NVIDIA RTX 3060",
            "AMD Radeon"
        ])

# ================= RIGHT SIDE (RESULT) =================
with right:

    st.markdown("### 📊 Prediction Panel")

    st.info("Click below to estimate laptop price")

    predict_btn = st.button("🚀 Predict Price", use_container_width=True)

# ---------------- FEATURE ENGINEERING ----------------
resolution = screen
x_res = int(resolution.split("x")[0])
y_res = int(resolution.split("x")[1])

ppi = ((x_res**2 + y_res**2) ** 0.5) / 15.6  # default scaling

# ---------------- PREDICTION ----------------
if predict_btn:

    data = pd.DataFrame([[
        company,
        typename,
        cpu,
        ram,
        memory,
        gpu,
        opsys,
        weight,
        touchscreen,
        ips,
        ppi
    ]], columns=[
        "Company","TypeName","Cpu",
        "Ram","Memory","Gpu",
        "OpSys","Weight",
        "Touchscreen","IPS","PPI"
    ])

    prediction = model.predict(data)[0]

    st.markdown("---")

    st.markdown(f"""
    <div class="result-box">
        💰 Estimated Laptop Price: ₹ {int(prediction):,}
    </div>
    """, unsafe_allow_html=True)

    st.success("Prediction completed successfully!")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption("🔹 Built with Machine Learning (Gradient Boosting Regressor)")
