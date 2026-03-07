import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Explainable ML Model for High-Risk Prostate Cancer Progression Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom SCI-style CSS
# =========================
st.markdown("""
<style>
    .main {
        background-color: #f6f9fc;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .title-box {
        background: linear-gradient(90deg, #0f4c81 0%, #1b6ca8 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .subtitle-box {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 5px solid #1b6ca8;
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid #e5edf5;
        margin-bottom: 0.8rem;
    }
    .footer {
        margin-top: 1.5rem;
        padding-top: 0.6rem;
        border-top: 1px solid #dbe5f0;
        color: #4a6072;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown(
    """
    <div class="title-box">
        <h2 style="margin:0;">Explainable ML Model for High-Risk Prostate Cancer Progression Prediction</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle-box">
    <b>Clinical Objective:</b> Predict individual future risk of progression to 
    <b>high-risk prostate cancer</b> (positive class = 1) using explainable machine learning.
    <br><br>
    <i>Note: The Random Forest (RF) model is highly recommended for patients with tPSA falling into the diagnostic gray zone (4.0 - 10.0 ng/mL).</i>
    </div>
    """,
    unsafe_allow_html=True
)

# Hero image (确保目录下有这张图，没有的话会忽略)
try:
    st.image("prostate_cancer_banner.jpg", caption="High-Risk Prostate Cancer Progression", use_column_width=True)
except:
    pass

# =========================
# Setup Models and Data
# =========================
# 定义模型路径字典
MODEL_DICT = {
    "Random Forest (RF) - Recommended for PSA Gray Zone (4-10)": "RF_best.pkl",
    "Multilayer Perceptron (MLP)": "MLP_best.pkl"
}

DATA_FILE = "Final_Cleaned_Data.xlsx"
TARGET_COL = "status"
ID_COL = "ID"
FEATURES = ['tPSA', 'CK_MB', 'LDH', 'RBC']

@st.cache_resource
def load_model(model_path):
    # 这里加载模型，如果报错提示找不到，请检查路径是否正确，如 "saved_models/RF_best.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE)

df = load_data()

if ID_COL in df.columns:
    df_feat = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
else:
    df_feat = df.drop(columns=[TARGET_COL], errors="ignore")

# =========================
# Model Selection & Input panel
# =========================
st.markdown('<div class="card"><b>1. Select Predictive Model</b></div>', unsafe_allow_html=True)
selected_model_name = st.selectbox(
    "Choose the machine learning algorithm:",
    list(MODEL_DICT.keys()),
    index=0 # 默认选中第一个 (RF)
)
current_model_path = MODEL_DICT[selected_model_name]
model = load_model(current_model_path)


st.markdown('<div class="card"><b>2. Patient Feature Input</b></div>', unsafe_allow_html=True)

feature_ranges = {}
for f in FEATURES:
    col = df_feat[f]
    if pd.api.types.is_numeric_dtype(col):
        mn = float(np.nanmin(col.values))
        mx = float(np.nanmax(col.values))
        dv = float(np.nanmedian(col.values))
        if mn == mx:
            mx = mn + 1.0
        feature_ranges[f] = {"type": "numerical", "min": mn, "max": mx, "default": dv}
    else:
        opts = [str(x) for x in col.dropna().unique().tolist()] or ["0", "1"]
        feature_ranges[f] = {"type": "categorical", "options": opts, "default": opts[0]}

left, right = st.columns(2)
vals = []

for i, (feat, p) in enumerate(feature_ranges.items()):
    box = left if i % 2 == 0 else right
    with box:
        if p["type"] == "numerical":
            v = st.number_input(
                f"{feat} ({p['min']:.3f} - {p['max']:.3f})",
                min_value=float(p["min"]),
                max_value=float(p["max"]),
                value=float(p["default"])
            )
        else:
            v = st.selectbox(f"{feat}", p["options"])
            try:
                v = float(v)
            except:
                pass
        vals.append(v)

X_input = pd.DataFrame([vals], columns=FEATURES)

# =========================
# Prediction + Visualization
# =========================
if st.button(f"Predict with {selected_model_name.split(' ')[0]}", type="primary", use_container_width=True):
    
    # 灰区提醒逻辑
    tpsa_val = X_input['tPSA'].values[0]
    if (4.0 <= tpsa_val <= 10.0) and ("MLP" in selected_model_name):
        st.warning("⚠️ **Clinical Notice:** The patient's tPSA is in the gray zone (4.0 - 10.0 ng/mL). According to our study, the Random Forest (RF) model demonstrates superior discriminative accuracy in this specific cohort. Consider switching to the RF model above.")

    pred = model.predict(X_input)[0]
    proba_pos = model.predict_proba(X_input)[0][1] * 100 if hasattr(model, "predict_proba") else 0.0

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="card"><b>Model Classification</b></div>', unsafe_allow_html=True)
        if pred == 1:
            st.error("Predicted outcome: **High risk of progression to prostate cancer**")
        else:
            st.success("Predicted outcome: **Lower risk of progression to prostate cancer**")

        st.info(f"Predicted probability of progression to **high-risk prostate cancer**: **{proba_pos:.2f}%**")

    with c2:
        st.markdown('<div class="card"><b>Risk Gauge</b></div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pos,
            number={"suffix": "%"},
            title={"text": "Progression Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f4c81"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<div class="card"><b>Explainability (SHAP)</b></div>', unsafe_allow_html=True)

    try:
        with st.spinner('Generating explainability plots...'):
            # 使用 KernelExplainer 兼容所有模型
            bg = shap.sample(df_feat[FEATURES], min(100, len(df_feat)), random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, bg)

            shap_values = explainer.shap_values(X_input, nsamples=200)

            # 统一抽取正类(1)单样本SHAP向量 -> (n_features,)
            if isinstance(shap_values, list):
                sv_class1 = np.array(shap_values[1])[0]
                ev = explainer.expected_value
                base_class1 = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)
            else:
                arr = np.array(shap_values)
                if arr.ndim == 3:
                    sv_class1 = arr[0, :, 1]
                elif arr.ndim == 2:
                    sv_class1 = arr[0]
                else:
                    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

                ev = explainer.expected_value
                base_class1 = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)

            exp = shap.Explanation(
                values=sv_class1,
                base_values=base_class1,
                data=X_input.iloc[0].values,
                feature_names=FEATURES
            )

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("**SHAP Waterfall Plot**")
                fig_wf = plt.figure(figsize=(8, 4.2), dpi=200)
                shap.plots.waterfall(exp, max_display=min(10, len(FEATURES)), show=False)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)

            with p2:
                st.markdown("**SHAP Force Plot**")
                fig_force = plt.figure(figsize=(8, 4.2), dpi=200)
                shap.force_plot(
                    base_class1,
                    sv_class1,
                    X_input.iloc[0],
                    feature_names=FEATURES,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig_force, use_container_width=True)
                plt.close(fig_force)

            st.markdown("**Feature Contribution Table**")
            abs_sv = np.abs(sv_class1)
            total = abs_sv.sum() if abs_sv.sum() != 0 else 1.0
            pct = abs_sv / total * 100

            contribution_df = pd.DataFrame({
                "Feature": FEATURES,
                "Input Value": X_input.iloc[0].values,
                "SHAP Value": sv_class1,
                "Direction": ["Increase progression risk" if v > 0 else "Decrease progression risk" for v in sv_class1],
                "Contribution (%)": pct
            }).sort_values("Contribution (%)", ascending=False)

            st.dataframe(
                contribution_df.style.format({
                    "Input Value": "{:.4f}",
                    "SHAP Value": "{:.4f}",
                    "Contribution (%)": "{:.2f}"
                }),
                use_container_width=True
            )

            st.markdown("**Contribution Bar Chart**")
            fig_bar, ax = plt.subplots(figsize=(9, 4), dpi=220)
            bar_colors = ["#E53935" if v > 0 else "#1E88E5" for v in contribution_df["SHAP Value"]]
            ax.barh(contribution_df["Feature"], contribution_df["SHAP Value"], color=bar_colors)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("SHAP value")
            ax.set_title("Red: increase progression risk | Blue: decrease progression risk")
            ax.invert_yaxis()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

    except Exception as e:
        st.warning(f"SHAP visualization failed: {e}")

# =========================
# Footer
# =========================
st.markdown(
    """
    <div class="footer">
        <b>Author:</b> Sheng Liang<br>
        <b>Affiliation:</b> Hengzhou City People's Hospital, Hengzhou, Guangxi, China
    </div>
    """,
    unsafe_allow_html=True
)
