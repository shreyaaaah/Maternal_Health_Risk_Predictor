"""
STEP 5: Real-Time Streamlit Dashboard
Run with: streamlit run step5_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Maternal Health Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3650;
        margin: 8px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #3d1a1a, #4a2020);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #3d2e0a, #4a380a);
        border: 1px solid #ffaa00;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #0a3d1a, #0a4a20);
        border: 1px solid #00cc44;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .alert-badge {
        background: #ff4444;
        color: white;
        border-radius: 50%;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        imputer = pickle.load(open('imputer.pkl', 'rb'))
        kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
        pca = pickle.load(open('pca_model.pkl', 'rb'))
        umap_model = pickle.load(open('umap_model.pkl', 'rb'))
        cluster_names = json.load(open('cluster_names.json', 'r'))
        return scaler, imputer, kmeans, pca, umap_model, cluster_names
    except Exception as e:
        st.error(f"⚠️ Models not found. Please run steps 1-4 first.\nError: {e}")
        st.stop()

@st.cache_data
def load_data():
    X_raw = pd.read_csv('X_raw.csv')
    X_umap = np.load('X_umap.npy')
    labels = np.load('labels_kmeans.npy')
    source = pd.read_csv('source_labels.csv')
    return X_raw, X_umap, labels, source

scaler, imputer, kmeans, pca, umap_model, cluster_names = load_models()
X_raw, X_umap, labels, source = load_data()

# ─────────────────────────────────────────────
# SESSION STATE — Patient Log
# ─────────────────────────────────────────────

if 'patient_log' not in st.session_state:
    st.session_state.patient_log = []

if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0

# ─────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────

ALL_FEATURES = ['age', 'systolic_bp', 'diastolic_bp', 'blood_sugar', 'body_temp',
                'heart_rate', 'fhr_baseline', 'accelerations', 'fetal_movement',
                'uterine_contractions', 'pct_abnormal_stv', 'mean_stv',
                'pct_abnormal_ltv', 'mean_ltv', 'light_decelerations',
                'severe_decelerations', 'prolonged_decelerations',
                'histogram_width', 'histogram_min', 'histogram_max',
                'histogram_peaks', 'histogram_zeros', 'histogram_mode',
                'histogram_mean', 'histogram_median', 'histogram_variance',
                'histogram_tendency', 'pulse_pressure']

def predict_cluster(vitals_dict):
    vitals_dict['pulse_pressure'] = vitals_dict.get('systolic_bp', 120) - vitals_dict.get('diastolic_bp', 80)
    
    row = pd.DataFrame([{feat: vitals_dict.get(feat, np.nan) for feat in ALL_FEATURES}])
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)
    row_pca = pca.transform(row_scaled)
    cluster = kmeans.predict(row_pca)[0]
    
    cluster_name = cluster_names.get(str(cluster), cluster_names.get(cluster, "Unknown"))
    return int(cluster), cluster_name

def get_risk_color(name):
    if "HIGH" in name: return "#ff4444", "risk-high", "🔴"
    if "MEDIUM" in name: return "#ffaa00", "risk-medium", "🟡"
    return "#00cc44", "risk-low", "🟢"

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown('<p class="header-title">🏥 Maternal Health Risk Monitor</p>', unsafe_allow_html=True)
    st.caption("Real-Time Unsupervised Clustering System | UCI Maternal + CTG Dataset")
with col_time:
    st.metric("Live Time", datetime.now().strftime("%H:%M:%S"))
    st.metric("Total Patients in System", len(X_raw))

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — New Patient Input
# ─────────────────────────────────────────────

st.sidebar.markdown("## 👩 New Patient Assessment")
st.sidebar.markdown("Enter patient vitals below:")

with st.sidebar.form("patient_form"):
    patient_id = st.text_input("Patient ID", value=f"P{len(st.session_state.patient_log)+1:04d}")
    
    st.markdown("**📋 Maternal Vitals**")
    age = st.slider("Age (years)", 10, 60, 28)
    systolic = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
    diastolic = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
    blood_sugar = st.number_input("Blood Sugar (mmol/L)", 4.0, 20.0, 7.0, step=0.1)
    body_temp = st.number_input("Body Temp (°F)", 95.0, 105.0, 98.6, step=0.1)
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 160, 76)
    
    st.markdown("**🫀 Fetal Signals (optional)**")
    fhr = st.number_input("Fetal HR Baseline (bpm)", 80, 200, 140)
    uc = st.number_input("Uterine Contractions/sec", 0.0, 1.0, 0.003, step=0.001, format="%.3f")
    
    submitted = st.form_submit_button("🔍 Assess Patient", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# MAIN CONTENT — 3 Columns
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns([1.2, 2, 1.5])

# ── COLUMN 1: Assessment Result ──

with col1:
    st.markdown("### 🔍 Current Assessment")
    
    if submitted:
        vitals = {
            'age': age, 'systolic_bp': systolic, 'diastolic_bp': diastolic,
            'blood_sugar': blood_sugar, 'body_temp': body_temp,
            'heart_rate': heart_rate, 'fhr_baseline': fhr,
            'uterine_contractions': uc
        }
        
        cluster_id, cluster_name = predict_cluster(vitals)
        color, css_class, emoji = get_risk_color(cluster_name)
        
        # Show result
        st.markdown(f"""
        <div class="{css_class}">
            <h2 style="color:{color}; margin:0">{emoji}</h2>
            <h3 style="color:{color}; margin:4px 0">{cluster_name}</h3>
            <p style="color:#aaa; margin:0; font-size:0.85rem">Cluster {cluster_id}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Log patient
        st.session_state.patient_log.append({
            'id': patient_id, 'time': datetime.now().strftime("%H:%M:%S"),
            'cluster': cluster_id, 'risk': cluster_name,
            'age': age, 'systolic_bp': systolic, 'heart_rate': heart_rate,
            'blood_sugar': blood_sugar
        })
        
        if "HIGH" in cluster_name:
            st.session_state.alert_count += 1
            st.error(f"🚨 ALERT: Patient {patient_id} flagged as HIGH RISK!")
            st.markdown("#### 🏥 Clinical Recommendations:")
            st.markdown("- **Urgent OB/GYN Consultation Required**")
            st.markdown("- Continuous Fetal Monitoring (CTG)")
            st.markdown("- BP stabilization and blood sugar assessment")
        elif "MEDIUM" in cluster_name:
            st.warning(f"⚠️ Patient {patient_id} — Monitor closely")
            st.markdown("#### 📝 Recommended Actions:")
            st.markdown("- Repeat vitals in 4 hours")
            st.markdown("- Dietary review and hydration")
        else:
            st.success(f"✅ Patient {patient_id} — Stable")
            st.markdown("#### ✅ Routine Care:")
            st.markdown("- Standard prenatal checkup schedule")
    
    else:
        st.info("👈 Enter patient vitals in the sidebar to assess risk")
    
    # Recent log
    st.markdown("### 📋 Recent Patients")
    if st.session_state.patient_log:
        log_df = pd.DataFrame(st.session_state.patient_log[-8:][::-1])
        for _, row in log_df.iterrows():
            c, _, emoji = get_risk_color(row['risk'])
            st.markdown(f"`{row['id']}` {emoji} {row['risk']} @ {row['time']}")
    else:
        st.caption("No patients assessed yet")

# ── COLUMN 2: UMAP Patient Map ──

with col2:
    st.markdown("### 🗺️ Patient Risk Cluster Map")
    
    # Build plotly UMAP scatter
    cluster_labels_display = [cluster_names.get(str(l), cluster_names.get(int(l), f"C{l}"))
                               for l in labels]
    
    color_map = {}
    for name in set(cluster_labels_display):
        if "HIGH" in name: color_map[name] = "#ff4444"
        elif "MEDIUM" in name: color_map[name] = "#ffaa00"
        else: color_map[name] = "#00cc44"
    
    df_plot = pd.DataFrame({
        'UMAP_1': X_umap[:, 0],
        'UMAP_2': X_umap[:, 1],
        'Cluster': cluster_labels_display,
        'Source': source['source'],
        'Heart Rate': X_raw['heart_rate'].round(1),
        'Systolic BP': X_raw['systolic_bp'].round(1),
    })
    
    fig = px.scatter(
        df_plot, x='UMAP_1', y='UMAP_2', color='Cluster',
        color_discrete_map=color_map,
        hover_data=['Source', 'Heart Rate', 'Systolic BP'],
        opacity=0.5, template='plotly_dark',
        title=f'All {len(X_raw):,} Patients — Live Risk Map'
    )
    
    # Add new patient if assessed
    if submitted and st.session_state.patient_log:
        last = st.session_state.patient_log[-1]
        c, _, _ = get_risk_color(last['risk'])
        fig.add_trace(go.Scatter(
            x=[0], y=[0],  # approximate center
            mode='markers+text',
            marker=dict(size=20, color=c, symbol='star', line=dict(width=2, color='white')),
            text=[f"NEW: {last['id']}"],
            textposition='top center',
            name=f"NEW: {last['id']}"
        ))
    
    fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
    fig.update_layout(
        height=420, legend=dict(orientation='h', y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,1)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ── COLUMN 3: Stats ──

with col3:
    st.markdown("### 📊 Cluster Statistics")
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    for cid in cluster_counts.index:
        name = cluster_names.get(str(cid), cluster_names.get(int(cid), f"Cluster {cid}"))
        count = cluster_counts[cid]
        pct = count / len(labels) * 100
        color, _, emoji = get_risk_color(name)
        
        st.markdown(f"**{emoji} {name}**")
        st.progress(int(pct), text=f"{count:,} patients ({pct:.1f}%)")
        st.caption("")
    
    st.divider()
    
    # Alert summary
    st.markdown("### 🚨 Session Alerts")
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.metric("High Risk Alerts", st.session_state.alert_count, delta=None)
    with alert_col2:
        st.metric("Patients Assessed", len(st.session_state.patient_log))
    
    if st.session_state.alert_count > 0:
        high_risk = [p for p in st.session_state.patient_log if "HIGH" in p['risk']]
        st.markdown("**🔴 High Risk Patients:**")
        for p in high_risk[-5:]:
            st.markdown(f"- `{p['id']}` | BP: {p['systolic_bp']} | HR: {p['heart_rate']}")

# ─────────────────────────────────────────────
# BOTTOM ROW — Feature Insights
# ─────────────────────────────────────────────

st.divider()
st.markdown("### 📈 Cluster Feature Insights")

feat_col1, feat_col2 = st.columns(2)

key_feats = ['heart_rate', 'systolic_bp', 'blood_sugar', 'pct_abnormal_stv']

with feat_col1:
    df_box = X_raw[key_feats[:2] + ['cluster']].copy()
    df_box['cluster'] = df_box['cluster'].map(
        lambda x: cluster_names.get(str(x), cluster_names.get(int(x), f"C{x}")))
    df_melt = df_box.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig2 = px.box(df_melt, x='cluster', y='Value', color='Feature',
                  template='plotly_dark', title='Heart Rate & Blood Pressure by Cluster',
                  color_discrete_sequence=['#00d4ff', '#7b61ff'])
    fig2.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

with feat_col2:
    df_box2 = X_raw[key_feats[2:] + ['cluster']].copy()
    df_box2['cluster'] = df_box2['cluster'].map(
        lambda x: cluster_names.get(str(x), cluster_names.get(int(x), f"C{x}")))
    df_melt2 = df_box2.melt(id_vars='cluster', var_name='Feature', value_name='Value')
    fig3 = px.box(df_melt2, x='cluster', y='Value', color='Feature',
                  template='plotly_dark', title='Blood Sugar & Fetal Variability by Cluster',
                  color_discrete_sequence=['#ff6b6b', '#ffd93d'])
    fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🔬 Unsupervised ML System | K-Means + HDBSCAN + PCA + UMAP | UCI Maternal Health + CTG Datasets")
st.caption("⚠️ This is a research/academic project — not a clinical decision tool.")
