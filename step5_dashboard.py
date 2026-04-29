import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# THEME TOGGLE
# -----------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

theme_choice = st.sidebar.radio(
    "🎨 Theme Mode",
    ["Light", "Dark"],
    index=0 if st.session_state.theme_mode == "Light" else 1
)

st.session_state.theme_mode = theme_choice

if st.session_state.theme_mode == "Dark":
    bg = "linear-gradient(135deg, #070b16 0%, #0f172a 45%, #1a1020 100%)"
    text = "#ffffff"
    card_bg = "rgba(15, 23, 42, 0.78)"
    plot_template = "plotly_dark"
    sidebar_border = "rgba(255,255,255,0.10)"
    info_bg = "#111827"
else:
    bg = "linear-gradient(135deg, #f6f8ff 0%, #eef3ff 45%, #fff7fb 100%)"
    text = "#101828"
    card_bg = "rgba(255,255,255,0.78)"
    plot_template = "plotly_white"
    sidebar_border = "rgba(120,120,160,0.18)"
    info_bg = "#ffffff"

# -----------------------------
# CSS
# -----------------------------
st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}

.block-container {{
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}}

h1 {{
    font-size: 3rem !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #4f46e5, #ec4899, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

h2, h3 {{
    font-weight: 800 !important;
}}

section[data-testid="stSidebar"] {{
    background: {card_bg};
    backdrop-filter: blur(18px);
    border-right: 1px solid {sidebar_border};
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stNumberInput input {{
    border-radius: 14px !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}}

.stButton > button {{
    border-radius: 14px !important;
    background: linear-gradient(135deg, #ef4444, #ec4899) !important;
    color: white !important;
    font-weight: 800 !important;
    border: none !important;
    padding: 0.7rem 1rem !important;
    box-shadow: 0 12px 30px rgba(236, 72, 153, 0.35);
    transition: all 0.25s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 16px 40px rgba(236, 72, 153, 0.45);
}}

.risk-box {{
    padding: 34px;
    border-radius: 26px;
    text-align: center;
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 0.5px;
    margin: 14px 0 24px 0;
    box-shadow: 0 25px 60px rgba(0,0,0,0.18);
    animation: fadeUp 0.7s ease both;
}}

.low {{
    background: linear-gradient(135deg, #dcfce7, #22c55e);
    border: 2px solid #16a34a;
    color: #052e16;
}}

.medium {{
    background: linear-gradient(135deg, #fef3c7, #f59e0b);
    border: 2px solid #d97706;
    color: #3b2500;
}}

.high {{
    background: linear-gradient(135deg, #fee2e2, #ef4444);
    border: 2px solid #dc2626;
    color: #450a0a;
}}

[data-testid="stPlotlyChart"] {{
    background: {card_bg};
    border-radius: 24px;
    padding: 14px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
}}

div[data-testid="stDataFrame"] {{
    border-radius: 18px !important;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
}}

[data-testid="stMetric"] {{
    background: {card_bg};
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
}}

@keyframes fadeUp {{
    from {{
        opacity: 0;
        transform: translateY(18px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD FILES
# -----------------------------
@st.cache_resource
def load_artifacts():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    pca = pickle.load(open("pca_model.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    features = pickle.load(open("all_features.pkl", "rb"))
    cluster_names = json.load(open("cluster_names.json", "r"))
    return scaler, pca, kmeans, features, cluster_names


@st.cache_data
def load_data():
    X_raw = pd.read_csv("X_raw_labeled.csv")
    X_pca = np.load("X_pca.npy")
    labels = np.load("labels_kmeans.npy")
    return X_raw, X_pca, labels


scaler, pca, kmeans, FEATURES, cluster_names = load_artifacts()
X_raw, X_pca, labels = load_data()

# -----------------------------
# PATIENT-SPECIFIC RISK LOGIC
# -----------------------------
def patient_risk_interpretation(v):
    score = 0
    flags = []

    if v["systolic_bp"] >= 140:
        score += 4
        flags.append("High systolic BP")
    elif v["systolic_bp"] >= 130:
        score += 2
        flags.append("Elevated systolic BP")
    elif v["systolic_bp"] >= 120:
        score += 1
        flags.append("Borderline systolic BP")

    if v["diastolic_bp"] >= 90:
        score += 3
        flags.append("High diastolic BP")
    elif v["diastolic_bp"] >= 80:
        score += 1
        flags.append("Borderline diastolic BP")

    if v["blood_sugar"] >= 8:
        score += 3
        flags.append("High blood sugar")
    elif v["blood_sugar"] >= 7:
        score += 1
        flags.append("Borderline blood sugar")

    if v["body_temp"] >= 100.4:
        score += 3
        flags.append("Fever")
    elif v["body_temp"] >= 99.5:
        score += 1
        flags.append("Slightly high temperature")

    if v["heart_rate"] >= 110:
        score += 3
        flags.append("High heart rate")
    elif v["heart_rate"] >= 100:
        score += 1
        flags.append("Borderline high heart rate")

    if v["age"] < 18 or v["age"] > 40:
        score += 1
        flags.append("Age-related risk")

    if score >= 5:
        return "HIGH RISK", flags, score
    elif score >= 2:
        return "MEDIUM RISK", flags, score
    else:
        return "LOW RISK", flags, score


# -----------------------------
# CLUSTER PREDICTION
# -----------------------------
def predict_cluster(vitals):
    row = pd.DataFrame([{feature: vitals.get(feature, np.nan) for feature in FEATURES}])

    row = row.apply(pd.to_numeric, errors="coerce")

    for col in FEATURES:
        if pd.isna(row.loc[0, col]):
            row.loc[0, col] = X_raw[col].median()

    row_values = row[FEATURES].values

    row_scaled = scaler.transform(row_values)
    row_pca = pca.transform(row_scaled)

    cluster = int(kmeans.predict(row_pca)[0])
    cluster_risk = cluster_names.get(str(cluster), "UNKNOWN")

    return cluster, cluster_risk, row_pca


def risk_style(risk):
    if "HIGH" in risk:
        return "high", "🔴"
    elif "MEDIUM" in risk:
        return "medium", "🟡"
    else:
        return "low", "🟢"


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
# 🏥 Maternal Health Risk Predictor
### Intelligent Unsupervised Risk Profiling for Maternal Health Monitoring
""")

st.caption("Using K-Means, DBSCAN, Agglomerative Clustering, PCA, and patient-specific clinical scoring")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("👩 New Patient Assessment")

patient_id = st.sidebar.text_input("Patient ID", "P0001")

age = st.sidebar.slider("Age", 10, 60, 28)
systolic_bp = st.sidebar.number_input("Systolic BP", 60, 220, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 150, 80)
blood_sugar = st.sidebar.number_input("Blood Sugar", 4.0, 20.0, 7.0, step=0.1)
body_temp = st.sidebar.number_input("Body Temp", 95.0, 105.0, 98.6, step=0.1)
heart_rate = st.sidebar.number_input("Heart Rate", 40, 180, 76)

submit = st.sidebar.button("Assess Patient", type="primary")

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Final Patient Assessment")

    if submit:
        vitals = {
            "age": age,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "blood_sugar": blood_sugar,
            "body_temp": body_temp,
            "heart_rate": heart_rate
        }

        cluster, cluster_risk, patient_pca = predict_cluster(vitals)
        patient_risk, flags, score = patient_risk_interpretation(vitals)

        final_risk = patient_risk
        css, emoji = risk_style(final_risk)

        st.markdown(
            f"""
            <div class="risk-box {css}">
                {emoji}<br>
                {final_risk}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### 🧾 Details")
        st.write(f"**Patient ID:** {patient_id}")
        st.write(f"**Assigned Cluster:** {cluster}")
        st.write(f"**Population / Cluster Risk:** {cluster_risk}")
        st.write(f"**Individual Patient Risk Score:** {score}")

        st.info(
            "Cluster risk shows population-level grouping. "
            "Final patient risk is based on the individual patient's current vitals."
        )

        if flags:
            st.write("### ⚠️ Risk Factors")
            for f in flags:
                st.write(f"- {f}")
        else:
            st.success("No major risk factors detected.")

        if final_risk == "HIGH RISK":
            st.error("🚨 Recommended: urgent clinical monitoring.")
        elif final_risk == "MEDIUM RISK":
            st.warning("⚠️ Recommended: regular monitoring and follow-up.")
        else:
            st.success("✅ Recommended: routine prenatal care.")
    else:
        st.info("Enter patient values and click **Assess Patient**.")

with col2:
    st.subheader("🗺️ Cluster Visualization")

    plot_df = pd.DataFrame({
        "PCA1": X_pca[:, 0],
        "PCA2": X_pca[:, 1],
        "cluster": labels.astype(str)
    })

    plot_df["risk"] = plot_df["cluster"].map(cluster_names)

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="risk",
        opacity=0.65,
        template=plot_template,
        title="KMeans Patient Clusters",
        color_discrete_map={
            "LOW RISK": "#22c55e",
            "MEDIUM RISK": "#f59e0b",
            "HIGH RISK": "#ef4444"
        }
    )

    if submit:
        fig.add_scatter(
            x=[patient_pca[0, 0]],
            y=[patient_pca[0, 1]],
            mode="markers+text",
            marker=dict(
                size=22,
                symbol="star",
                color="#ffffff" if st.session_state.theme_mode == "Dark" else "#111827",
                line=dict(width=2, color="#ec4899")
            ),
            text=[patient_id],
            textposition="top center",
            name="New Patient"
        )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Risk Group"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# STATS
# -----------------------------
st.divider()

st.subheader("📊 Dataset Statistics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Records", f"{len(X_raw):,}")

with c2:
    st.metric("Clusters", len(set(labels)))

with c3:
    st.metric("Primary Model", "KMeans")

# -----------------------------
# CLUSTER TABLE
# -----------------------------
st.subheader("📋 Cluster Profile Table")

profile = pd.read_csv("cluster_profile.csv")
st.dataframe(profile, use_container_width=True)

