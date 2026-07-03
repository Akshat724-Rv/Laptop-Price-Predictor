import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os 

if "similar_display" not in st.session_state:
    st.session_state.similar_display = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# ---------- LOAD DATA & MODEL ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pipe = pickle.load(open(os.path.join(BASE_DIR, "pipe.pkl"), "rb"))
df = pickle.load(open(os.path.join(BASE_DIR, "df.pkl"), "rb"))


# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #f1f5f9;
}

/* HEADERS */
.header {
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    font-size: 42px;
    color: #0f172a;
    font-weight: 800;
}
.header p {
    color: #475569;
    font-size: 17px;
}

/* INPUT BOX STYLING */
.box {
    background: #ffffff;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(148, 163, 184, 0.2);
    margin-bottom: 25px;
}

/* SECTION TITLES */
.box-title {
    font-weight: 600;
    font-size: 18px;
    color: #1e293b;
    margin-bottom: 10px;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1e3a8a, #1e40af);
    transition: 0.3s;
}

/* RESULT BOX */
.result {
    background: #ecfdf5;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #065f46;
    margin-top: 25px;
    border: 2px solid #34d399;
}

/* FOOTER */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>💻 Laptop Price Predictor</h1>
    <p>AI-powered Smart Pricing System 🚀</p>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT SECTIONS ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="box-title">💡 Basic Details</div>', unsafe_allow_html=True)
    company = st.selectbox('Brand', sorted(df['Company'].unique()))
    type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
    ram = st.selectbox('RAM (GB)', [4, 8, 16, 32, 64])
    weight = st.number_input('Weight (kg)', value=1.5, step=0.1)

with col2:
    st.markdown('<div class="box-title">🖥️ Display Features</div>', unsafe_allow_html=True)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Panel', ['No', 'Yes'])
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 15.6)
    resolution = st.selectbox('Resolution', ['1366x768', '1600x900', '1920x1080', '2560x1440', '3840x2160'])

with col3:
    st.markdown('<div class="box-title">⚙️ Hardware</div>', unsafe_allow_html=True)
    cpu = st.selectbox('CPU Brand', sorted(df['Cpu brand'].unique()))
    gpu = st.selectbox('GPU Brand', sorted(df['Gpu brand'].unique()))
    hdd = st.selectbox('HDD (GB)', [0, 256, 512, 1024])
    ssd = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024])

    os_list = list(df['OpSys'].unique())
    os_list += ["Windows 11", "Windows 12"]
    os_list = list(set(os_list))
    os = st.selectbox('Operating System', sorted(os_list))


# ---------- SAFE CATEGORY FIX ----------
def safe_value(val, allowed):
    return val if val in allowed else allowed[0]

company = safe_value(company, df['Company'].unique())
type_name = safe_value(type_name, df['TypeName'].unique())
cpu = safe_value(cpu, df['Cpu brand'].unique())
gpu = safe_value(gpu, df['Gpu brand'].unique())


# ---------- PREDICTION ----------
if st.button("🔮 Predict Laptop Price"):

    # 🔥 OS NORMALIZATION (FINAL FIX)
    os_clean = os.lower()
    if "windows" in os_clean:
        os_model = "Windows"
    elif "mac" in os_clean:
        os_model = "Mac"
    elif "linux" in os_clean:
        os_model = "Linux"
    else:
        os_model = "No OS"

    os_model = safe_value(os_model, df['OpSys'].unique())

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': int(ram),
        'Weight': float(weight),
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': float(ppi),
        'Cpu brand': cpu,
        'HDD': int(hdd),
        'SSD': int(ssd),
        'Gpu brand': gpu,
        'OpSys': os_model
    }]).astype(object)

    pred = pipe.predict(query)[0]

    # 🔥 SAFE PRICE LOGIC
    if np.isnan(pred) or np.isinf(pred):
        st.error("⚠️ Prediction failed. Try different inputs.")
    else:
        # auto detect log scale
        if 0 <= pred <= 20:
            price = np.exp(pred)
        else:
            price = pred

        # clamp realistic range
        price = max(10000, min(price, 300000))
        st.session_state.prediction = int(price)
        
        # ---------- SIMILAR LAPTOPS ----------

        price_col = 'Price' if 'Price' in df.columns else 'price'

        similar = df[
            (df[price_col] >= st.session_state.prediction * 0.6) &
            (df[price_col] <= st.session_state.prediction * 1.4)
        ]

        if similar.shape[0] > 0:
            similar_display = similar[['Company','TypeName','Ram','Cpu brand','Gpu brand', price_col]].copy()

            # price round + int convert
            similar_display[price_col] = similar_display[price_col].astype(float).round(0).astype(int)
               
           # sort first, then show
            similar_display = similar_display.sort_values(by=price_col)

            st.session_state.similar_display = similar_display.head(5)
        
        else:
            st.info("No similar laptops found.")


if st.session_state.prediction is not None:
    st.markdown(
        f'<div class="result">💰 Estimated Price: ₹ {st.session_state.prediction:,}</div>',
        unsafe_allow_html=True
    )

if st.session_state.similar_display is not None:
    st.markdown("### 💻 Similar Laptops")
    st.table(st.session_state.similar_display)

# ---------- COMPARE ----------
st.divider()
st.markdown("## ⚔️ Compare Two Laptops")

colA, colB = st.columns(2)

with colA:
    st.markdown("### Laptop A")
    company_A = st.selectbox('Brand A', df['Company'].unique(), key="A1")
    ram_A = st.selectbox('RAM A', [4,8,16,32,64], key="A2")
    gpu_A = st.selectbox('GPU A', df['Gpu brand'].unique(), key="A4")
    cpu_A = st.selectbox('CPU A', df['Cpu brand'].unique(), key="A3")
    ssd_A = st.selectbox('SSD A (GB)', [0, 128, 256, 512, 1024], key="A5")
with colB:
    st.markdown("### Laptop B")
    company_B = st.selectbox('Brand B', df['Company'].unique(), key="B1")
    ram_B = st.selectbox('RAM B', [4,8,16,32,64], key="B2")
    gpu_B = st.selectbox('GPU B', df['Gpu brand'].unique(), key="B4")
    cpu_B = st.selectbox('CPU B', df['Cpu brand'].unique(), key="B3")
    ssd_B = st.selectbox('SSD B (GB)', [0, 128, 256, 512, 1024], key="B5")

def make_query(company, ram, cpu,gpu, ssd):

    # use same resolution + screen size as main input
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    os_clean = os.lower()
    if "windows" in os_clean:
        os_model = "Windows"
    elif "mac" in os_clean:
        os_model = "Mac"
    elif "linux" in os_clean:
        os_model = "Linux"
    else:
        os_model = "No OS"

    os_model = safe_value(os_model, df['OpSys'].unique()) 

    return pd.DataFrame([{
            'Company': company,
            'TypeName': 'Notebook',
            'Ram': int(ram),
            'Weight': 2.0,
            'Touchscreen': 1 if touchscreen == 'Yes' else 0,
            'Ips': 1 if ips == 'Yes' else 0,
            'ppi': float(ppi),
            'Cpu brand': cpu,
            'HDD': int(hdd),
            'SSD': int(ssd),
            'Gpu brand': gpu,
            'OpSys': os_model
        }]).astype(object)

    # ---------- SAFE COMPARE PREDICTION ----------
def safe_price(pred):
    if np.isnan(pred) or np.isinf(pred):
         return None
    
    if 0 <= pred <= 20:
         price = np.exp(pred)
    else:
         price = pred
    price = max(10000, min(price, 300000))     
    return int(price)

if st.button("⚔️ Compare"):
    predA_raw = pipe.predict(make_query(company_A, ram_A, cpu_A,gpu_A, ssd_A))[0]
    predB_raw = pipe.predict(make_query(company_B, ram_B, cpu_B,gpu_B, ssd_B))[0]

    pred_A = safe_price(predA_raw)
    pred_B = safe_price(predB_raw)

    st.markdown("### 📊 Comparison Result")

    if pred_A is None or pred_B is None:
       st.error("⚠️ Comparison failed. Try different inputs.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"💻 Laptop A: ₹{pred_A:,}")

        with col2:
            st.success(f"💻 Laptop B: ₹{pred_B:,}")

        if pred_A > pred_B:
           st.info("👉 Laptop A is more expensive")
        elif pred_B > pred_A:
            st.success("Laptop B is Premium Choice")
        else:
            st.warning("Both laptops are similar in price")
  # cd "C:\Users\aksha\OneDrive\Desktop\Sem 6\laptop-price-predictor-regression-project-main (1)\laptop-price-predictor-regression-project-main"
  # python train_model.py 
  # streamlit run app.py