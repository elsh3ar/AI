import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go

# 1. إعدادات الصفحة بقوة (Custom CSS for Bold & Powerful Look)
st.set_page_config(
    page_title="Liver Health AI - Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design and animation effect
st.markdown("""
<style>
    /* Gradient Background (Deep Blue to Teal) */
    .stApp {
        background: linear-gradient(135deg, #022533 0%, #004d40 100%);
        color: white !important;
    }
    
    /* Main Title Styling (White/Cyan) */
    .main-title {
        text-align: center;
        color: #00bcd4;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        padding-bottom: 0px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Subtitle Styling (Light Cyan) */
    .subtitle {
        text-align: center;
        color: #b2ebf2;
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 1.4rem;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    /* Container Styling (Dark Cyan Card) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        background-color: #006064;
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 2px solid #00bcd4;
    }
    
    /* Input Label Styling (White) */
    .stNumberInput label, .stSelectbox label {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem;
    }

    /* Button Styling (Bold Gradient) */
    div.stButton > button {
        background: linear-gradient(135deg, #ff1744 0%, #d50000 100%) !important;
        color: white !important;
        border-radius: 40px !important;
        padding: 20px 40px !important;
        border: none !important;
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 30px rgba(0,0,0,0.6);
        background: linear-gradient(135deg, #d50000 0%, #ff1744 100%) !important;
    }

    /* Input Fields Styling (Background) */
    .stNumberInput input, .stSelectbox select {
        background-color: #e0f7fa !important;
        border-radius: 10px !important;
        color: #004d40 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. وظيفة تحميل الانيميشن (Lottie)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# روابط الانيميشن (يمكنك تغييرها لاحقاً)
lottie_main = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5njpXv.json") # Scanning

# 3. تحميل العقل اللي صنعناه (Model & Scaler)
# تأكد أن الملفات دي في نفس الفولدر
model = joblib.load('liver_model.pkl')
scaler = joblib.load('scaler.pkl')

# 4. تصميم الهيدر (Header with Title and Animation)
st.markdown('<div class="main-title">Liver Health AI - Diagnosis Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-driven analysis of patient biochemical markers.</div>', unsafe_allow_html=True)

# عرض انيميشن "الاسكان" فوق المدخلات
if lottie_main:
    st_lottie(lottie_main, height=180, key="scanning")

st.write("---")

# 5. تصميم جسم الواجهة (Two Columns in a styled container)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧬 **Demographics & Basic Vital Signs**", unsafe_allow_html=True)
    age = st.number_input("Patient Age (1-100)", 1, 100, 25, help="Age in years.")
    gender = st.selectbox("Patient Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin (TB)", 0.1, 50.0, 1.0)
    db = st.number_input("Direct Bilirubin (DB)", 0.1, 20.0, 0.5)
    ap = st.number_input("Alkaline Phosphotase (AP)", 10, 2000, 200)

with col2:
    st.markdown("### 🧫 **Enzyme & Protein Panel**", unsafe_allow_html=True)
    sgpt = st.number_input("Alamine Aminotransferase (SGPT)", 10, 2000, 40)
    sgot = st.number_input("Aspartate Aminotransferase (SGOT)", 10, 2000, 40)
    tp = st.number_input("Total Proteins (TP)", 1.0, 10.0, 7.0)
    alb = st.number_input("Albumin (ALB)", 1.0, 10.0, 3.5)
    ag_ratio = st.number_input("Albumin & Globulin Ratio (A/G)", 0.1, 3.0, 1.0)

# 6. زرار التشخيص بتأثير حركي (Diagnostic Button with hover effect)
st.write("---")
left_spacer, center_btn, right_spacer = st.columns([1.5, 2, 1.5])
with center_btn:
    analyze_btn = st.button("RUN ADVANCED DIAGNOSTIC ANALYSIS")

# 7. معالجة النتائج وعرض الجراف (Processing & Personalized Graph)
if analyze_btn:
    with st.spinner('Analyzing patterns and calculating risk probabilities...'):
        # تحويل الجنس لرقم
        gender_num = 1 if gender == "Male" else 0
        
        # تجهيز البيانات
        features = np.array([[age, gender_num, tb, db, ap, sgpt, sgot, tp, alb, ag_ratio]])
        features_scaled = scaler.transform(features)
        
        # التوقع والنسبة
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)[0][1] # نسبة المرض
        
        # تحويل النسبة المئوية
        risk_percentage = prediction_proba * 100
        
        # عرض النتائج في صف جديد
        st.write("---")
        result_title_col1, result_title_col2 = st.columns([2, 1])
        with result_title_col1:
            st.markdown("## **Analysis Complete! - Risk Assessment**")
        
        # إنشاء الجراف (Gauge Chart) لمدى الخطورة
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disease Risk %", 'font': {'size': 24, 'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
                'bar': {'color': "#00bcd4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#00c853"}, # Green (Low)
                    {'range': [40, 70], 'color': "#ffd600"}, # Yellow (Medium)
                    {'range': [70, 100], 'color': "#ff1744"}], # Red (High)
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage}
            }
        ))
        
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"})
        
        # عرض الجراف والنتيجة النصية
        result_col1, result_col2 = st.columns([2, 1])
        with result_col1:
            st.plotly_chart(fig, use_container_width=True)
            
        with result_col2:
            if prediction[0] == 1:
                st.error("### 🚨 **Diagnosis: Liver Disease Detected**")
                st.markdown(f"**Calculated Risk:** `{risk_percentage:.2f}%` (HIGH)")
                st.markdown("""
                > **Warning:** The patient's biochemical profile strongly matches patterns associated with Liver Disease. 
                > **Recommendation:** Immediate follow-up and further diagnostic testing are advised.
                """)
            else:
                st.success("### ✅ **Diagnosis: No Liver Disease Detected**")
                st.markdown(f"**Calculated Risk:** `{risk_percentage:.2f}%` (LOW)")
                st.markdown("""
                > **Result:** The patient's biochemical profile is within the normal range as per this analysis. 
                > **Recommendation:** Annual check-ups are still recommended.
                """)