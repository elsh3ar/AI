import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Liver Health AI – Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  AUTO-TRAIN if pkl files are missing (needed for Streamlit Cloud)
# ══════════════════════════════════════════════════════════════
if not os.path.exists("liver_model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("⚙️ First run — training AI model, please wait (~15 sec)..."):
        from train import train_and_save
        acc = train_and_save()
    st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root {
    --bg:      #f0f4f8;
    --card:    #ffffff;
    --primary: #0077b6;
    --accent:  #00b4d8;
    --danger:  #e63946;
    --success: #2a9d8f;
    --text:    #1a202c;
    --muted:   #64748b;
    --border:  #e2e8f0;
    --shadow:  0 4px 24px rgba(0,119,182,0.10);
}
.stApp {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.hero {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
    border-radius: 20px; padding: 36px 48px;
    margin-bottom: 28px; box-shadow: 0 8px 32px rgba(0,119,182,0.25);
}
.hero h1 { color:#fff; font-size:2.4rem; font-weight:800; margin:0 0 6px 0; }
.hero p  { color:rgba(255,255,255,0.85); font-size:1.05rem; margin:0; }
.badge {
    display:inline-block; background:rgba(255,255,255,0.2); color:#fff;
    padding:4px 14px; border-radius:50px; font-size:0.78rem;
    font-weight:700; letter-spacing:1px; margin-bottom:14px;
}
.card {
    background: var(--card); border-radius:16px; padding:28px;
    box-shadow: var(--shadow); border:1px solid var(--border); margin-bottom:20px;
}
.card-title {
    font-size:0.95rem; font-weight:700; color:var(--primary);
    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:18px;
}
.stNumberInput label, .stSelectbox label {
    color: var(--text) !important; font-weight:600 !important; font-size:0.9rem !important;
}
.stNumberInput input {
    background:#f8fafc !important; border:1.5px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important; font-weight:600 !important;
}
div.stButton > button {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%) !important;
    color:#fff !important; border-radius:50px !important; padding:16px 48px !important;
    border:none !important; font-size:1.05rem !important; font-weight:800 !important;
    letter-spacing:1.5px !important; box-shadow:0 6px 20px rgba(0,119,182,0.35) !important;
    transition:all 0.25s ease !important; width:100% !important;
}
div.stButton > button:hover {
    transform:translateY(-2px) !important; box-shadow:0 10px 28px rgba(0,119,182,0.45) !important;
}
hr { border-color:var(--border) !important; margin:24px 0 !important; }
.metric-box {
    background:var(--card); border-radius:14px; padding:20px 24px;
    box-shadow:var(--shadow); border:1px solid var(--border); text-align:center;
}
.metric-box .val { font-size:2rem; font-weight:800; line-height:1; margin-bottom:6px; }
.metric-box .lbl { font-size:0.8rem; font-weight:600; color:var(--muted);
                   text-transform:uppercase; letter-spacing:0.7px; }
.result-danger {
    background:linear-gradient(135deg,#ffeef0,#ffe0e3);
    border:2px solid #e63946; border-radius:16px; padding:24px 28px; color:#9b2226;
}
.result-safe {
    background:linear-gradient(135deg,#e8f8f5,#d4efeb);
    border:2px solid #2a9d8f; border-radius:16px; padding:24px 28px; color:#1a5c55;
}
.result-danger h2, .result-safe h2 { margin:0 0 8px 0; font-size:1.3rem; }
.result-danger p,  .result-safe p  { margin:0; font-size:0.92rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  NORMAL REFERENCE RANGES
# ══════════════════════════════════════════════════════════════
NORMAL_RANGES = {
    "Total Bilirubin":  (0.1, 1.2),
    "Direct Bilirubin": (0.0, 0.3),
    "Alk. Phosphatase": (44,  147),
    "SGPT":             (7,   56),
    "SGOT":             (10,  40),
    "Total Proteins":   (6.0, 8.3),
    "Albumin":          (3.5, 5.0),
    "A/G Ratio":        (0.8, 2.0),
}

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('liver_model.pkl'), joblib.load('scaler.pkl')

model, scaler = load_model()

# ══════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">🩺 AI-POWERED DIAGNOSTICS</div>
    <h1>Liver Health AI — Diagnosis Pro</h1>
    <p>Enter the patient's biochemical markers and receive an instant AI-powered risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">🧬 Demographics & Bilirubin</div>', unsafe_allow_html=True)
    age    = st.number_input("Patient Age (years)",         1,   100,  45)
    gender = st.selectbox(   "Patient Gender",             ["Male", "Female"])
    tb     = st.number_input("Total Bilirubin (mg/dL)",    0.1, 75.0,  0.9, step=0.1)
    db     = st.number_input("Direct Bilirubin (mg/dL)",   0.0, 20.0,  0.2, step=0.1)
    ap     = st.number_input("Alkaline Phosphatase (U/L)", 10,  2500, 180)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🧫 Enzymes & Protein Panel</div>', unsafe_allow_html=True)
    sgpt     = st.number_input("SGPT / ALT (U/L)",         5,   2500, 35)
    sgot     = st.number_input("SGOT / AST (U/L)",         5,   2500, 30)
    tp       = st.number_input("Total Proteins (g/dL)",    1.0, 10.0,  7.0, step=0.1)
    alb      = st.number_input("Albumin (g/dL)",           0.5, 10.0,  3.8, step=0.1)
    ag_ratio = st.number_input("Albumin / Globulin Ratio", 0.1,  3.0,  1.1, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  RUN BUTTON
# ══════════════════════════════════════════════════════════════
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyse = st.button("🔬  RUN ADVANCED DIAGNOSTIC ANALYSIS")

# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════
if analyse:
    with st.spinner("Analysing patient data…"):
        gender_num      = 1 if gender == "Male" else 0
        features        = np.array([[age, gender_num, tb, db, ap, sgpt, sgot, tp, alb, ag_ratio]])
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        proba           = model.predict_proba(features_scaled)[0]
        risk_pct        = proba[1] * 100
        safe_pct        = proba[0] * 100

    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    risk_color   = "#e63946" if risk_pct >= 70 else ("#e9c46a" if risk_pct >= 40 else "#2a9d8f")
    status_label = "HIGH RISK" if risk_pct >= 70 else ("MODERATE" if risk_pct >= 40 else "LOW RISK")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:{risk_color}">{risk_pct:.1f}%</div><div class="lbl">Disease Risk</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:#0077b6">{safe_pct:.1f}%</div><div class="lbl">Healthy Confidence</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:{risk_color}">{status_label}</div><div class="lbl">Risk Level</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:#64748b">{age}y / {gender[0]}</div><div class="lbl">Patient Profile</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=risk_pct,
            delta={'reference': 50, 'valueformat': '.1f',
                   'increasing': {'color': '#e63946'}, 'decreasing': {'color': '#2a9d8f'}},
            number={'suffix': '%', 'font': {'size': 52, 'color': '#1a202c'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Liver Disease Risk Score", 'font': {'size': 15, 'color': '#64748b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#cbd5e1'},
                'bar':  {'color': risk_color, 'thickness': 0.28},
                'bgcolor': 'white', 'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': '#d4efeb'},
                    {'range': [40, 70], 'color': '#fef9e7'},
                    {'range': [70,100], 'color': '#fde8ea'},
                ],
                'threshold': {'line': {'color': risk_color, 'width': 3}, 'thickness': 0.8, 'value': risk_pct}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=50, b=20, l=30, r=30), height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with ch2:
        labels       = list(NORMAL_RANGES.keys())
        patient_vals = [tb, db, ap, sgpt, sgot, tp, alb, ag_ratio]
        def normalise(val, lo, hi):
            return (val - lo) / (hi - lo) * 100 if hi != lo else 100
        norm_patient = [normalise(v, NORMAL_RANGES[k][0], NORMAL_RANGES[k][1]) for v, k in zip(patient_vals, labels)]
        bar_colors   = ["#e63946" if v < 0 or v > 100 else "#0077b6" for v in norm_patient]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=labels, y=norm_patient, marker_color=bar_colors,
                                  marker_line_width=0,
                                  hovertemplate="<b>%{x}</b><br>Patient: %{customdata}<extra></extra>",
                                  customdata=[f"{v:.2f}" for v in patient_vals]))
        fig_bar.add_hrect(y0=0, y1=100, fillcolor="rgba(42,157,143,0.07)", line_width=0)
        fig_bar.add_shape(type="line", x0=-0.5, x1=len(labels)-0.5, y0=0, y1=0,
                          line=dict(color="#2a9d8f", width=1.5, dash="dot"))
        fig_bar.add_shape(type="line", x0=-0.5, x1=len(labels)-0.5, y0=100, y1=100,
                          line=dict(color="#2a9d8f", width=1.5, dash="dot"))
        fig_bar.update_layout(
            title={'text': "Biomarkers vs Normal Range", 'font': {'size': 14, 'color': '#64748b'}},
            yaxis_title="% of Normal Range", xaxis_tickangle=-35,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#e2e8f0'), xaxis=dict(linecolor='#e2e8f0'),
            margin=dict(t=50, b=80, l=40, r=20), height=320,
            font=dict(color='#1a202c'), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 🕸️ Biomarker Radar — Patient vs Reference")
    radar_labels  = labels + [labels[0]]
    patient_radar = [max(0, min(200, v)) for v in norm_patient] + [max(0, min(200, norm_patient[0]))]
    normal_radar  = [100]*len(labels) + [100]
    r2, g2, b2 = int(risk_color[1:3],16), int(risk_color[3:5],16), int(risk_color[5:7],16)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normal_radar, theta=radar_labels, fill='toself', name='Normal',
        fillcolor='rgba(42,157,143,0.12)', line=dict(color='#2a9d8f', width=2, dash='dash')))
    fig_radar.add_trace(go.Scatterpolar(
        r=patient_radar, theta=radar_labels, fill='toself', name='Patient',
        fillcolor=f'rgba({r2},{g2},{b2},0.18)', line=dict(color=risk_color, width=2.5)))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, range=[0,200],
                                   tickfont=dict(size=9, color='#64748b'), gridcolor='#e2e8f0'),
                   angularaxis=dict(tickfont=dict(size=11, color='#1a202c'), linecolor='#e2e8f0')),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        margin=dict(t=40, b=40, l=60, r=60), height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="result-danger">
            <h2>🚨 Diagnosis: Liver Disease Indicators Detected</h2>
            <p><b>Calculated risk: {risk_pct:.1f}%</b> — The patient's biochemical profile shows patterns
            associated with liver disease. Immediate clinical follow-up and further diagnostic testing are strongly advised.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <h2>✅ Diagnosis: No Liver Disease Indicators</h2>
            <p><b>Calculated risk: {risk_pct:.1f}%</b> — The patient's biochemical markers are within
            the expected healthy range. Routine annual check-ups are still recommended.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ This tool is for research and decision-support only. Not a substitute for professional medical diagnosis.")
