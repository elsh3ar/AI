import streamlit as st
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
#  AUTO-TRAIN on first run (Streamlit Cloud — no .pkl files yet)
# ══════════════════════════════════════════════════════════════
if not os.path.exists("liver_model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("⚙️ First run — training AI model, please wait (~15 sec)..."):
        from app import train_and_save
        acc = train_and_save()
    st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

:root {
    --bg:      #f0f4f8;
    --card:    #ffffff;
    --primary: #0077b6;
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
.hero p  { color:rgba(255,255,255,0.88); font-size:1.05rem; margin:0; }
.badge {
    display:inline-block; background:rgba(255,255,255,0.22); color:#fff;
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
    color: #1a202c !important; font-weight:600 !important; font-size:0.9rem !important;
}
.stNumberInput input {
    background:#f8fafc !important; border:1.5px solid var(--border) !important;
    border-radius:10px !important; color:#1a202c !important; font-weight:600 !important;
}
div.stButton > button {
    background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%) !important;
    color:#fff !important; border-radius:50px !important; padding:16px 48px !important;
    border:none !important; font-size:1.05rem !important; font-weight:800 !important;
    letter-spacing:1.5px !important; box-shadow:0 6px 20px rgba(0,119,182,0.35) !important;
    transition:all 0.25s ease !important; width:100% !important;
}
div.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 28px rgba(0,119,182,0.45) !important;
}
hr { border-color:var(--border) !important; margin:24px 0 !important; }
.metric-box {
    background:var(--card); border-radius:14px; padding:20px 16px;
    box-shadow:var(--shadow); border:1px solid var(--border); text-align:center;
}
.metric-box .val { font-size:1.9rem; font-weight:800; line-height:1; margin-bottom:6px; }
.metric-box .lbl { font-size:0.78rem; font-weight:600; color:#64748b;
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
#  NORMAL RANGES
# ══════════════════════════════════════════════════════════════
NORMAL_RANGES = {
    "Total_Bilirubin":            (0.1,  1.2),
    "Direct_Bilirubin":           (0.0,  0.3),
    "Alkaline_Phosphotase":       (44,   147),
    "Alamine_Aminotransferase":   (7,    56),
    "Aspartate_Aminotransferase": (10,   40),
    "Total_Protiens":             (6.0,  8.3),
    "Albumin":                    (3.5,  5.0),
    "Albumin_and_Globulin_Ratio": (0.8,  2.0),
}
DISPLAY_LABELS = {
    "Total_Bilirubin":            "Total Bilirubin",
    "Direct_Bilirubin":           "Direct Bilirubin",
    "Alkaline_Phosphotase":       "Alkaline Phosphotase",
    "Alamine_Aminotransferase":   "Alamine Aminotransferase",
    "Aspartate_Aminotransferase": "Aspartate Aminotransferase",
    "Total_Protiens":             "Total Protiens",
    "Albumin":                    "Albumin",
    "Albumin_and_Globulin_Ratio": "Albumin & Globulin Ratio",
}

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('liver_model.pkl'), joblib.load('scaler.pkl')

model, scaler = load_model()

# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">🩺 AI-POWERED DIAGNOSTICS</div>
    <h1>Liver Health AI — Diagnosis Pro</h1>
    <p>Enter the patient's biochemical markers and receive an instant AI-powered risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  INPUT FORM  — exact CSV column names
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">🧬 Demographics & Bilirubin</div>', unsafe_allow_html=True)
    age    = st.number_input("Age",                    min_value=1,   max_value=100,  value=45)
    gender = st.selectbox(   "Gender",                ["Male", "Female"])
    tb     = st.number_input("Total_Bilirubin",        min_value=0.1, max_value=75.0, value=0.9,  step=0.1, help="Normal: 0.1 – 1.2 mg/dL")
    db     = st.number_input("Direct_Bilirubin",       min_value=0.0, max_value=20.0, value=0.2,  step=0.1, help="Normal: 0.0 – 0.3 mg/dL")
    ap     = st.number_input("Alkaline_Phosphotase",   min_value=10,  max_value=2500, value=180,            help="Normal: 44 – 147 U/L")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🧫 Enzymes & Protein Panel</div>', unsafe_allow_html=True)
    sgpt     = st.number_input("Alamine_Aminotransferase",   min_value=5,   max_value=2500, value=35,           help="SGPT / ALT — Normal: 7 – 56 U/L")
    sgot     = st.number_input("Aspartate_Aminotransferase", min_value=5,   max_value=2500, value=30,           help="SGOT / AST — Normal: 10 – 40 U/L")
    tp       = st.number_input("Total_Protiens",             min_value=1.0, max_value=10.0, value=7.0, step=0.1,help="Normal: 6.0 – 8.3 g/dL")
    alb      = st.number_input("Albumin",                    min_value=0.5, max_value=10.0, value=3.8, step=0.1,help="Normal: 3.5 – 5.0 g/dL")
    ag_ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=0.1, max_value=3.0,  value=1.1, step=0.1,help="Normal: 0.8 – 2.0")
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

    risk_color   = "#c0392b" if risk_pct >= 70 else ("#b8860b" if risk_pct >= 40 else "#1a7a6e")
    status_label = "HIGH RISK"  if risk_pct >= 70 else ("MODERATE" if risk_pct >= 40 else "LOW RISK")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:{risk_color}">{risk_pct:.1f}%</div><div class="lbl">Disease Risk</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:#0077b6">{safe_pct:.1f}%</div><div class="lbl">Healthy Confidence</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:{risk_color}">{status_label}</div><div class="lbl">Risk Level</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="val" style="color:#334155">{age}y / {gender[0]}</div><div class="lbl">Patient Profile</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Bar ──────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        fig_gauge = go.Figure(go.Indicator(
            mode   = "gauge+number+delta",
            value  = risk_pct,
            delta  = {'reference': 50, 'valueformat': '.1f',
                    'increasing': {'color': '#c0392b'},
                    'decreasing': {'color': '#1a7a6e'}},
            number = {'suffix': '%', 'font': {'size': 56, 'color': '#1a202c', 'family': 'Plus Jakarta Sans'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title  = {'text': "<b>Liver Disease Risk Score</b>",
                    'font': {'size': 15, 'color': '#1a202c', 'family': 'Plus Jakarta Sans'}},
            gauge  = {
                'axis': {'range': [0, 100], 'tickwidth': 1,
                        'tickcolor': '#334155', 'tickfont': {'color': '#1a202c', 'size': 12}},
                'bar':  {'color': risk_color, 'thickness': 0.30},
                'bgcolor': 'white', 'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': '#b7e4c7'},
                    {'range': [40, 70], 'color': '#ffe69c'},
                    {'range': [70,100], 'color': '#f5c6cb'},
                ],
                'threshold': {'line': {'color': risk_color, 'width': 4},
                            'thickness': 0.85, 'value': risk_pct}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=30, r=30), height=340,
            font=dict(family='Plus Jakarta Sans', color='#1a202c'))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with ch2:
        keys         = list(NORMAL_RANGES.keys())
        disp_labels  = [DISPLAY_LABELS[k] for k in keys]
        patient_vals = [tb, db, ap, sgpt, sgot, tp, alb, ag_ratio]

        def normalise(val, lo, hi):
            return (val - lo) / (hi - lo) * 100 if hi != lo else 100

        norm_patient = [normalise(v, NORMAL_RANGES[k][0], NORMAL_RANGES[k][1])
                        for v, k in zip(patient_vals, keys)]
        bar_colors   = ["#c0392b" if v < 0 or v > 100 else "#0077b6" for v in norm_patient]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=disp_labels, y=norm_patient,
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in norm_patient],
            textposition="outside",
            textfont=dict(color='#1a202c', size=11, family='Plus Jakarta Sans'),
            hovertemplate="<b>%{x}</b><br>Value: %{customdata}<br>%{y:.1f}% of normal range<extra></extra>",
            customdata=[f"{v:.2f}" for v in patient_vals],
        ))
        fig_bar.add_hrect(y0=0, y1=100, fillcolor="rgba(42,157,143,0.08)", line_width=0)
        fig_bar.add_shape(type="line", x0=-0.5, x1=len(keys)-0.5, y0=0,   y1=0,
                        line=dict(color="#1a7a6e", width=2, dash="dot"))
        fig_bar.add_shape(type="line", x0=-0.5, x1=len(keys)-0.5, y0=100, y1=100,
                        line=dict(color="#1a7a6e", width=2, dash="dot"))
        fig_bar.update_layout(
            title=dict(text="<b>Biomarkers vs Normal Range</b>",
                    font=dict(size=14, color='#1a202c', family='Plus Jakarta Sans')),
            yaxis_title="% of Normal Range", xaxis_tickangle=-30,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#cbd5e1', zerolinecolor='#94a3b8',
                    tickfont=dict(color='#1a202c', size=11),
                    title_font=dict(color='#1a202c')),
            xaxis=dict(linecolor='#cbd5e1', tickfont=dict(color='#1a202c', size=10)),
            margin=dict(t=60, b=110, l=50, r=20), height=340,
            font=dict(family='Plus Jakarta Sans', color='#1a202c'), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar ────────────────────────────────────────────────
    st.markdown("#### 🕸️ Biomarker Radar — Patient vs Normal Reference")

    radar_theta   = disp_labels + [disp_labels[0]]
    patient_radar = [max(0, min(200, v)) for v in norm_patient] + [max(0, min(200, norm_patient[0]))]
    normal_radar  = [100] * len(keys) + [100]
    r2 = int(risk_color[1:3], 16)
    g2 = int(risk_color[3:5], 16)
    b2 = int(risk_color[5:7], 16)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normal_radar, theta=radar_theta, fill='toself', name='Normal Range',
        fillcolor='rgba(42,157,143,0.15)', line=dict(color='#1a7a6e', width=2.5, dash='dash')))
    fig_radar.add_trace(go.Scatterpolar(
        r=patient_radar, theta=radar_theta, fill='toself', name='Patient',
        fillcolor=f'rgba({r2},{g2},{b2},0.18)', line=dict(color=risk_color, width=3)))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 200],
                            tickfont=dict(size=10, color='#1a202c'),
                            gridcolor='#94a3b8', linecolor='#64748b'),
            angularaxis=dict(tickfont=dict(size=11, color='#1a202c', family='Plus Jakarta Sans'),
                            linecolor='#94a3b8', gridcolor='#cbd5e1')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='center', x=0.5,
                    font=dict(size=12, color='#1a202c', family='Plus Jakarta Sans'),
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='#e2e8f0', borderwidth=1),
        margin=dict(t=50, b=50, l=80, r=80), height=420,
        font=dict(family='Plus Jakarta Sans', color='#1a202c'))
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Result Banner ────────────────────────────────────────
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="result-danger">
            <h2>🚨 Diagnosis: Liver Disease Indicators Detected</h2>
            <p><b>Calculated risk: {risk_pct:.1f}%</b> — The patient's biochemical profile shows patterns
            associated with liver disease. Immediate clinical follow-up and further diagnostic
            testing are strongly advised.</p>
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
