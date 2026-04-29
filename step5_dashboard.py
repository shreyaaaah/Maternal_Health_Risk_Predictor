import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0f1117;
    color: white;
}
.risk-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}
.low {
    background: linear-gradient(135deg, #073b1d, #0b6623);
    border: 2px solid #00cc66;
}
.medium {
    background: linear-gradient(135deg, #4a3400, #8a5c00);
    border: 2px solid #ffaa00;
}
.high {
    background: linear-gradient(135deg, #4a0f0f, #8b0000);
    border: 2px solid #ff4444;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD FILES
# -----------------------------
@st.cache_resource
def load_artifacts():
    scaler = pickle.load(open("scaler.pkl", "rb"))
    imputer = pickle.load(open("imputer.pkl", "rb"))
    pca = pickle.load(open("pca_model.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    features = pickle.load(open("all_features.pkl", "rb"))
    cluster_names = json.load(open("cluster_names.json", "r"))
    return scaler, imputer, pca, kmeans, features, cluster_names

@st.cache_data
def load_data():
    X_raw = pd.read_csv("X_raw_labeled.csv")
    X_pca = np.load("X_pca.npy")
    labels = np.load("labels_kmeans.npy")
    return X_raw, X_pca, labels

scaler, imputer, pca, kmeans, FEATURES, cluster_names = load_artifacts()
X_raw, X_pca, labels = load_data()

# -----------------------------
# PATIENT-SPECIFIC RULE INTERPRETER
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
# PREDICT CLUSTER
# -----------------------------
def predict_cluster(vitals):
    row = pd.DataFrame([{feature: vitals[feature] for feature in FEATURES}])
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)
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
# UI
# -----------------------------
st.title("🏥 Maternal Health Risk Predictor")
st.caption("Unsupervised Learning System using K-Means, DBSCAN, and Agglomerative Clustering")

st.sidebar.header("👩 New Patient Assessment")

patient_id = st.sidebar.text_input("Patient ID", "P0001")

age = st.sidebar.slider("Age", 10, 60, 28)
systolic_bp = st.sidebar.number_input("Systolic BP", 60, 220, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 40, 150, 80)
blood_sugar = st.sidebar.number_input("Blood Sugar", 4.0, 20.0, 7.0, step=0.1)
body_temp = st.sidebar.number_input("Body Temp", 95.0, 105.0, 98.6, step=0.1)
heart_rate = st.sidebar.number_input("Heart Rate", 40, 180, 76)

submit = st.sidebar.button("Assess Patient", type="primary")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Current Assessment")

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

        st.write("### Details")
        st.write(f"**Patient ID:** {patient_id}")
        st.write(f"**Assigned Cluster:** {cluster}")
        st.write(f"**Cluster Profile Risk:** {cluster_risk}")
        st.write(f"**Patient Risk Score:** {score}")

        if flags:
            st.write("### Risk Factors")
            for f in flags:
                st.write(f"- {f}")
        else:
            st.success("No major risk factors detected.")

        if final_risk == "HIGH RISK":
            st.error("Recommended: urgent clinical monitoring.")
        elif final_risk == "MEDIUM RISK":
            st.warning("Recommended: regular monitoring and follow-up.")
        else:
            st.success("Recommended: routine prenatal care.")

    else:
        st.info("Enter patient values and click Assess Patient.")

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
        opacity=0.55,
        template="plotly_dark",
        title="KMeans Patient Clusters"
    )

    if submit:
        fig.add_scatter(
            x=[patient_pca[0, 0]],
            y=[patient_pca[0, 1]],
            mode="markers+text",
            marker=dict(size=18, symbol="star", color="white"),
            text=[patient_id],
            textposition="top center",
            name="New Patient"
        )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("📊 Dataset Statistics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Records", len(X_raw))

with c2:
    st.metric("Clusters", len(set(labels)))

with c3:
    st.metric("Model", "KMeans Primary")

st.subheader("📋 Cluster Profile Table")
profile = pd.read_csv("cluster_profile.csv")
st.dataframe(profile, use_container_width=True)

