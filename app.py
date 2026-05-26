"""
Real Estate Price Predictor — AI/ML Portfolio Project
Author : Aman Chaudhary
Stack  : Streamlit · XGBoost · Scikit-Learn · Pandas · Joblib
Dataset: California Housing (20,640 rows, sklearn built-in)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time
from pathlib import Path
from train import train_and_save, MODEL_PATH, META_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RealEstateIQ — Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
  .metric-card {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
  }
  .metric-val { font-size: 1.8rem; font-weight: 600; color: #f0c040; }
  .metric-key { font-size: 0.72rem; color: #888; letter-spacing: 0.08em;
                text-transform: uppercase; margin-top: 4px; }
  .predict-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #f0c040; border-radius: 16px;
    padding: 2rem; text-align: center; margin-top: 1rem;
  }
  .predict-price { font-size: 2.8rem; font-weight: 700; color: #f0c040; }
  .predict-range { font-size: 0.85rem; color: #aaa; margin-top: 6px; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.05em;
  }
  .badge-green { background: rgba(29,158,117,0.15); color: #1d9e75;
                 border: 1px solid rgba(29,158,117,0.3); }
  .badge-red   { background: rgba(226,75,74,0.15);  color: #e24b4a;
                 border: 1px solid rgba(226,75,74,0.3); }
  .stSlider > div { padding-top: 4px; }
  h1 { font-size: 1.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Training model on 20K+ records…")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        train_and_save()
    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    return model, meta

model, meta = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 RealEstateIQ")
st.markdown(
    "**Real-time property price predictor** — XGBoost model trained on "
    "**20,640 California housing records** · R² **{:.2f}** · RMSE reduced "
    "**{:.0f}%** over linear baseline".format(meta["r2"], meta["rmse_improvement_pct"])
)
st.divider()

# ── Model metrics row ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("R² Score",          f"{meta['r2']:.3f}",          "Hold-out test set"),
    ("RMSE Improvement",  f"{meta['rmse_improvement_pct']:.0f}%", "vs linear baseline"),
    ("Training Rows",     f"{meta['train_rows']:,}",     "California Housing"),
    ("Features Used",     str(meta['n_features']),       "Engineered + raw"),
]
for col, (key, val, sub) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-key">{key}<br><span style="color:#666">{sub}</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 Property Details")
st.sidebar.markdown("Adjust the sliders to predict price.")

med_inc     = st.sidebar.slider("Median Income (×$10K)",     0.5, 15.0, 5.0,  0.1)
house_age   = st.sidebar.slider("House Age (years)",         1,   52,   20)
avg_rooms   = st.sidebar.slider("Avg Rooms per Household",   1.0, 14.0, 5.5,  0.1)
avg_bedrooms= st.sidebar.slider("Avg Bedrooms per Household",0.5, 5.0,  1.1,  0.1)
population  = st.sidebar.slider("Block Population",          3,   3500, 1200)
avg_occup   = st.sidebar.slider("Avg Occupancy",             1.0, 10.0, 3.0,  0.1)
latitude    = st.sidebar.slider("Latitude",                  32.5, 42.0, 36.8, 0.1)
longitude   = st.sidebar.slider("Longitude",                -124.4,-114.3,-119.6,0.1)

st.sidebar.divider()
st.sidebar.markdown("**Stack:** XGBoost · Scikit-Learn · Streamlit")
st.sidebar.markdown("**By:** Aman Chaudhary")

# ── Predict ───────────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "MedInc":      med_inc,
    "HouseAge":    house_age,
    "AveRooms":    avg_rooms,
    "AveBedrms":   avg_bedrooms,
    "Population":  population,
    "AveOccup":    avg_occup,
    "Latitude":    latitude,
    "Longitude":   longitude,
    # Engineered features (must match train.py)
    "rooms_per_person":    avg_rooms   / max(avg_occup, 0.1),
    "bedrooms_ratio":      avg_bedrooms / max(avg_rooms, 0.1),
    "income_per_room":     med_inc     / max(avg_rooms, 0.1),
    "pop_per_household":   population  / max(avg_occup, 0.1),
}])

t0    = time.perf_counter()
pred  = model.predict(input_df)[0]
ms    = (time.perf_counter() - t0) * 1000
price = max(pred * 100_000, 50_000)   # dataset unit = $100K

low   = price * 0.88
high  = price * 1.12

# ── Main prediction panel ─────────────────────────────────────────────────────
col_pred, col_analysis = st.columns([1, 1], gap="large")

with col_pred:
    st.markdown("### Predicted Price")
    st.markdown(f"""
    <div class="predict-box">
      <div class="predict-price">${price:,.0f}</div>
      <div class="predict-range">Estimated range: ${low:,.0f} – ${high:,.0f}</div>
      <div style="margin-top:12px;font-size:0.75rem;color:#666;">
        Inference: {ms:.2f}ms &nbsp;·&nbsp; Model: XGBoost
      </div>
    </div>""", unsafe_allow_html=True)

    # Price tier badge
    st.markdown("")
    if price < 150_000:
        tier, badge = "Budget",   "badge-green"
    elif price < 300_000:
        tier, badge = "Mid-range","badge-green"
    elif price < 500_000:
        tier, badge = "Premium",  "badge-red"
    else:
        tier, badge = "Luxury",   "badge-red"

    st.markdown(
        f'<span class="badge {badge}">🏷 {tier} property</span>',
        unsafe_allow_html=True
    )

with col_analysis:
    st.markdown("### Input Summary")
    summary = pd.DataFrame({
        "Feature": [
            "Median Income", "House Age", "Avg Rooms",
            "Avg Bedrooms", "Population", "Avg Occupancy",
            "Latitude", "Longitude"
        ],
        "Value": [
            f"${med_inc*10:.0f}K",  f"{house_age} yrs",
            f"{avg_rooms:.1f}",     f"{avg_bedrooms:.1f}",
            f"{population:,}",      f"{avg_occup:.1f}",
            f"{latitude:.1f}°N",    f"{longitude:.1f}°W",
        ]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.markdown("### Feature Importance")

feat_names = [
    "MedInc","HouseAge","AveRooms","AveBedrms","Population",
    "AveOccup","Latitude","Longitude",
    "rooms_per_person","bedrooms_ratio","income_per_room","pop_per_household"
]
importances = model.feature_importances_
fi_df = (
    pd.DataFrame({"Feature": feat_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
st.bar_chart(fi_df.set_index("Feature")["Importance"], use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: California Housing (sklearn.datasets) · "
    "Model: XGBoost + 5-fold GridSearchCV · "
    "Built by Aman Chaudhary · "
    "[GitHub](https://github.com/amn-00)"
)
