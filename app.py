"""
Real Estate Price Predictor v2 — AI/ML Portfolio Project
Author : Aman Chaudhary
Stack  : Streamlit · XGBoost · Scikit-Learn · Pandas · Joblib
Dataset: California Housing (20,640 rows, sklearn built-in)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time
from train import train_and_save, MODEL_PATH, META_PATH, DATASET_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RealEstateIQ — Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .metric-card {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
  }
  .metric-val { font-size: 1.6rem; font-weight: 600; color: #f0c040; }
  .metric-key { font-size: 0.7rem; color: #888; letter-spacing: 0.08em;
                text-transform: uppercase; margin-top: 4px; }

  .predict-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #f0c040; border-radius: 20px;
    padding: 2rem; text-align: center;
  }
  .predict-price { font-size: 3.2rem; font-weight: 700; color: #f0c040; font-family: 'DM Mono'; }
  .predict-range { font-size: 0.9rem; color: #aaa; margin-top: 8px; }

  .input-section {
    background: #141420; border: 1px solid #2a2a4a;
    border-radius: 16px; padding: 1.5rem;
  }
  .section-label {
    font-size: 0.75rem; font-weight: 600; color: #f0c040;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1rem; display: block;
  }
  .helper-text {
    font-size: 0.72rem; color: #666; margin-top: -8px; margin-bottom: 8px;
  }
  .badge {
    display: inline-block; padding: 4px 12px; border-radius: 99px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
    margin-top: 8px;
  }
  .badge-budget   { background: rgba(29,158,117,0.2); color: #1d9e75; border: 1px solid rgba(29,158,117,0.4); }
  .badge-mid      { background: rgba(240,192,64,0.2);  color: #f0c040; border: 1px solid rgba(240,192,64,0.4); }
  .badge-premium  { background: rgba(226,75,74,0.2);   color: #e24b4a; border: 1px solid rgba(226,75,74,0.4); }
  .badge-luxury   { background: rgba(150,50,200,0.2);  color: #b060e0; border: 1px solid rgba(150,50,200,0.4); }

  .comp-card {
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 8px;
  }
  .comp-price { font-size: 1.1rem; font-weight: 600; color: #f0c040; }
  .comp-detail { font-size: 0.75rem; color: #888; margin-top: 2px; }

  .whatif-box {
    background: #0f1520; border: 1px solid #2a2a4a; border-radius: 12px; padding: 1rem;
  }

  div[data-testid="stSlider"] label { font-size: 0.85rem !important; font-weight: 500; }
  div[data-testid="stNumberInput"] label { font-size: 0.85rem !important; }
  h1 { font-size: 2rem !important; }
  h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Training XGBoost model on 20K+ records…")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        train_and_save()
    model   = joblib.load(MODEL_PATH)
    meta    = joblib.load(META_PATH)
    dataset = joblib.load(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    return model, meta, dataset

model, meta, dataset = load_model()

# ── Helper ────────────────────────────────────────────────────────────────────
def make_input_df(med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude):
    return pd.DataFrame([{
        "MedInc": med_inc, "HouseAge": house_age,
        "AveRooms": avg_rooms, "AveBedrms": avg_bedrooms,
        "Population": population, "AveOccup": avg_occup,
        "Latitude": latitude, "Longitude": longitude,
        "rooms_per_person":  avg_rooms   / max(avg_occup, 0.1),
        "bedrooms_ratio":    avg_bedrooms / max(avg_rooms, 0.1),
        "income_per_room":   med_inc     / max(avg_rooms, 0.1),
        "pop_per_household": population  / max(avg_occup, 0.1),
    }])

def predict_price(df):
    t0   = time.perf_counter()
    pred = model.predict(df)[0]
    ms   = (time.perf_counter() - t0) * 1000
    return max(pred * 100_000, 50_000), ms

def tier_info(price):
    if price < 150_000:   return "Budget",   "badge-budget"
    elif price < 300_000: return "Mid-Range", "badge-mid"
    elif price < 500_000: return "Premium",  "badge-premium"
    else:                 return "Luxury",   "badge-luxury"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 RealEstateIQ")
st.markdown(
    f"**Real-time California property price predictor** · XGBoost trained on **20,640 records** · "
    f"R² **{meta['r2']:.3f}** · **{meta['rmse_improvement_pct']:.0f}%** RMSE reduction over linear baseline"
)
st.divider()

# ── Model metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (key, val, sub) in zip([c1,c2,c3,c4], [
    ("R² Score",         f"{meta['r2']:.3f}",                    "Hold-out test set"),
    ("RMSE Improvement", f"{meta['rmse_improvement_pct']:.0f}%", "vs linear baseline"),
    ("Training Rows",    f"{meta['train_rows']:,}",              "California Housing"),
    ("Features",         str(meta['n_features']),                "Engineered + raw"),
]):
    with col:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-key">{key}<br><span style="color:#555">{sub}</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — Inputs left, Prediction right
# ══════════════════════════════════════════════════════════════════════════════
col_inputs, col_results = st.columns([1.2, 1], gap="large")

with col_inputs:
    st.markdown("### 🔧 Property Details")
    st.markdown("Adjust values using sliders or type directly in the number boxes.")

    # ── Section 1: Financial ─────────────────────────────────────────────────
    st.markdown('<span class="section-label">💰 Financial & Demographics</span>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        med_inc = st.slider("Median Income", 0.5, 15.0, 5.0, 0.1,
                            help="Median household income in the block (×$10,000)")
    with col_b:
        med_inc = st.number_input("", 0.5, 15.0, med_inc, 0.1,
                                  key="ni_inc", label_visibility="hidden")
    st.markdown(f'<p class="helper-text">≈ ${med_inc*10:.0f}K annual household income</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        population = st.slider("Block Population", 3, 3500, 1200,
                               help="Total population in the block group")
    with col_b:
        population = st.number_input("", 3, 3500, population, 10,
                                     key="ni_pop", label_visibility="hidden")

    # ── Section 2: Property ──────────────────────────────────────────────────
    st.markdown('<span class="section-label">🏗 Property Characteristics</span>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        house_age = st.slider("House Age (years)", 1, 52, 20,
                              help="Median age of houses in the block")
    with col_b:
        house_age = st.number_input("", 1, 52, house_age, 1,
                                    key="ni_age", label_visibility="hidden")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        avg_rooms = st.slider("Avg Rooms per Household", 1.0, 14.0, 5.5, 0.1,
                              help="Average number of rooms per household")
    with col_b:
        avg_rooms = st.number_input("", 1.0, 14.0, avg_rooms, 0.1,
                                    key="ni_rooms", label_visibility="hidden")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        avg_bedrooms = st.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.1, 0.1,
                                 help="Average number of bedrooms per household")
    with col_b:
        avg_bedrooms = st.number_input("", 0.5, 5.0, avg_bedrooms, 0.1,
                                       key="ni_bed", label_visibility="hidden")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        avg_occup = st.slider("Avg Occupancy", 1.0, 10.0, 3.0, 0.1,
                              help="Average number of people per household")
    with col_b:
        avg_occup = st.number_input("", 1.0, 10.0, avg_occup, 0.1,
                                    key="ni_occ", label_visibility="hidden")

    # ── Section 3: Location ──────────────────────────────────────────────────
    st.markdown('<span class="section-label">📍 Location (California)</span>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        latitude = st.slider("Latitude", 32.5, 42.0, 36.8, 0.1,
                             help="32.5°N = San Diego area · 37.7°N = San Francisco · 42°N = Northern CA")
    with col_b:
        latitude = st.number_input("", 32.5, 42.0, latitude, 0.1,
                                   key="ni_lat", label_visibility="hidden")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        longitude = st.slider("Longitude", -124.4, -114.3, -119.6, 0.1,
                              help="-122.4 = San Francisco · -118.2 = Los Angeles · -114.3 = East CA")
    with col_b:
        longitude = st.number_input("", -124.4, -114.3, longitude, 0.1,
                                    key="ni_lon", label_visibility="hidden")

    # Location hint
    if latitude > 37.0 and longitude < -121.0:
        loc_hint = "📍 San Francisco Bay Area"
    elif latitude < 34.5 and longitude > -119.0:
        loc_hint = "📍 Los Angeles / San Diego Area"
    elif latitude > 38.5:
        loc_hint = "📍 Northern California"
    elif longitude > -117.0:
        loc_hint = "📍 Inland / Desert California"
    else:
        loc_hint = "📍 Central California"
    st.markdown(f'<p class="helper-text">{loc_hint}</p>', unsafe_allow_html=True)

# ── Compute prediction ────────────────────────────────────────────────────────
input_df      = make_input_df(med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
price, inf_ms = predict_price(input_df)
low, high     = price * 0.88, price * 1.12
tier, badge   = tier_info(price)

# ── Results column ────────────────────────────────────────────────────────────
with col_results:
    st.markdown("### 💰 Predicted Price")

    st.markdown(f"""
    <div class="predict-box">
      <div style="font-size:0.75rem;color:#666;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
        Estimated Market Value
      </div>
      <div class="predict-price">${price:,.0f}</div>
      <div class="predict-range">Range: ${low:,.0f} – ${high:,.0f}</div>
      <span class="badge {badge}">{tier} Property</span>
      <div style="margin-top:14px;font-size:0.72rem;color:#555;">
        Inference: {inf_ms:.2f}ms · Model: XGBoost + GridSearchCV
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price sensitivity chart ───────────────────────────────────────────────
    st.markdown("#### 📈 Price vs Income Sensitivity")
    inc_range  = np.arange(0.5, 15.1, 0.5)
    prices_inc = []
    for inc in inc_range:
        df_tmp = make_input_df(inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
        p, _   = predict_price(df_tmp)
        prices_inc.append(p)

    sens_df = pd.DataFrame({
        "Income (×$10K)": inc_range,
        "Predicted Price ($)": prices_inc
    }).set_index("Income (×$10K)")
    st.line_chart(sens_df, use_container_width=True, height=200)
    st.caption(f"Current income: ${med_inc*10:.0f}K — price: ${price:,.0f}")

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("📋 Full Input Summary", expanded=False):
        summary_df = pd.DataFrame({
            "Feature": ["Median Income","House Age","Avg Rooms","Avg Bedrooms",
                        "Population","Avg Occupancy","Latitude","Longitude"],
            "Value":   [f"${med_inc*10:.0f}K", f"{house_age} yrs",
                        f"{avg_rooms:.1f}", f"{avg_bedrooms:.1f}",
                        f"{population:,}", f"{avg_occup:.1f}",
                        f"{latitude:.1f}°N", f"{longitude:.1f}°W"]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — What-If Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### ⚡ What-If Comparison")
st.markdown("Compare two property configurations side by side.")

wc1, wc2 = st.columns(2, gap="large")

with wc1:
    st.markdown('<span class="section-label">🏠 Property A (Current)</span>', unsafe_allow_html=True)
    st.markdown(f"""<div class="whatif-box">
      <div style="font-size:1.6rem;font-weight:700;color:#f0c040;">${price:,.0f}</div>
      <div style="font-size:0.8rem;color:#888;margin-top:4px;">Income ${med_inc*10:.0f}K · {avg_rooms:.1f} rooms · Age {house_age}yrs</div>
      <span class="badge {badge}">{tier}</span>
    </div>""", unsafe_allow_html=True)

with wc2:
    st.markdown('<span class="section-label">🏡 Property B (Adjust to compare)</span>', unsafe_allow_html=True)
    b_inc  = st.slider("Income B (×$10K)", 0.5, 15.0, min(med_inc + 2.0, 15.0), 0.1, key="b_inc")
    b_age  = st.slider("House Age B",      1,   52,   max(house_age - 10, 1),         key="b_age")
    b_rooms= st.slider("Rooms B",          1.0, 14.0, min(avg_rooms + 1.0, 14.0), 0.1,key="b_rooms")

    df_b      = make_input_df(b_inc, b_age, b_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
    price_b, _= predict_price(df_b)
    tier_b, badge_b = tier_info(price_b)
    diff      = price_b - price
    diff_pct  = (diff / price) * 100

    st.markdown(f"""<div class="whatif-box">
      <div style="font-size:1.6rem;font-weight:700;color:#f0c040;">${price_b:,.0f}</div>
      <div style="font-size:0.8rem;color:#888;margin-top:4px;">Income ${b_inc*10:.0f}K · {b_rooms:.1f} rooms · Age {b_age}yrs</div>
      <span class="badge {badge_b}">{tier_b}</span>
      <div style="margin-top:10px;font-size:0.85rem;color:{'#1d9e75' if diff > 0 else '#e24b4a'};">
        {'▲' if diff > 0 else '▼'} ${abs(diff):,.0f} ({abs(diff_pct):.1f}%) vs Property A
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Comparable Properties
# ══════════════════════════════════════════════════════════════════════════════
if dataset is not None:
    st.divider()
    st.markdown("### 🔍 Comparable Properties")
    st.markdown("Similar properties from the dataset based on your inputs.")

    dist = (
        ((dataset["MedInc"]    - med_inc)    / dataset["MedInc"].std()).abs() * 0.35 +
        ((dataset["Latitude"]  - latitude)   / dataset["Latitude"].std()).abs() * 0.30 +
        ((dataset["Longitude"] - longitude)  / dataset["Longitude"].std()).abs() * 0.25 +
        ((dataset["AveRooms"]  - avg_rooms)  / dataset["AveRooms"].std()).abs() * 0.10
    )
    comps = dataset.loc[dist.nsmallest(5).index].reset_index(drop=True)

    comp_cols = st.columns(5)
    for i, (_, row) in enumerate(comps.iterrows()):
        with comp_cols[i]:
            diff_c = row["price"] - price
            color  = "#1d9e75" if diff_c >= 0 else "#e24b4a"
            st.markdown(f"""<div class="comp-card">
              <div class="comp-price">${row['price']:,.0f}</div>
              <div class="comp-detail">Income: ${row['MedInc']*10:.0f}K</div>
              <div class="comp-detail">Rooms: {row['AveRooms']:.1f}</div>
              <div class="comp-detail">Age: {row['HouseAge']:.0f} yrs</div>
              <div style="font-size:0.72rem;color:{color};margin-top:4px;">
                {'▲' if diff_c >= 0 else '▼'} ${abs(diff_c):,.0f}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 📊 Feature Importance")
st.markdown("Which features drive the model's predictions most.")

feat_names  = ["MedInc","HouseAge","AveRooms","AveBedrms","Population",
               "AveOccup","Latitude","Longitude",
               "rooms_per_person","bedrooms_ratio","income_per_room","pop_per_household"]
importances = model.feature_importances_
fi_df = (pd.DataFrame({"Feature": feat_names, "Importance": importances})
         .sort_values("Importance", ascending=False).reset_index(drop=True))
st.bar_chart(fi_df.set_index("Feature")["Importance"], use_container_width=True, height=280)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: California Housing (sklearn.datasets) · "
    "Model: XGBoost + 5-fold GridSearchCV · R² 0.82 · "
    "Built by **Aman Chaudhary** · "
    "[GitHub](https://github.com/amn-00/real-estate-price-predictor) · "
    "[LinkedIn](https://www.linkedin.com/in/aman-chaudhary-82b8382b1/)"
)
