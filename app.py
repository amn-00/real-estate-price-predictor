"""
Real Estate Price Predictor v2 — AI/ML Portfolio Project
Author : Aman Chaudhary
Stack  : Streamlit · XGBoost · Scikit-Learn · Pandas · Joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time
from train import train_and_save, MODEL_PATH, META_PATH, DATASET_PATH

st.set_page_config(
    page_title="RealEstateIQ — Price Predictor",
    page_icon="🏠", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:12px; padding:1rem 1.2rem; text-align:center; }
  .metric-val  { font-size:1.6rem; font-weight:600; color:#f0c040; }
  .metric-key  { font-size:0.7rem; color:#888; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px; }
  .predict-box { background:linear-gradient(135deg,#1a1a2e,#16213e); border:2px solid #f0c040; border-radius:20px; padding:2rem; text-align:center; }
  .predict-price { font-size:3.2rem; font-weight:700; color:#f0c040; font-family:'DM Mono'; }
  .predict-range { font-size:0.9rem; color:#aaa; margin-top:8px; }
  .section-label { font-size:0.75rem; font-weight:600; color:#f0c040; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; display:block; }
  .helper-text   { font-size:0.72rem; color:#666; margin-top:-6px; margin-bottom:6px; }
  .badge { display:inline-block; padding:4px 12px; border-radius:99px; font-size:0.75rem; font-weight:600; margin-top:8px; }
  .badge-budget  { background:rgba(29,158,117,0.2);  color:#1d9e75; border:1px solid rgba(29,158,117,0.4); }
  .badge-mid     { background:rgba(240,192,64,0.2);  color:#f0c040; border:1px solid rgba(240,192,64,0.4); }
  .badge-premium { background:rgba(226,75,74,0.2);   color:#e24b4a; border:1px solid rgba(226,75,74,0.4); }
  .badge-luxury  { background:rgba(150,50,200,0.2);  color:#b060e0; border:1px solid rgba(150,50,200,0.4); }
  .comp-card  { background:#1a1a2e; border:1px solid #2a2a4a; border-radius:10px; padding:0.8rem 1rem; margin-bottom:8px; }
  .comp-price { font-size:1.1rem; font-weight:600; color:#f0c040; }
  .comp-detail{ font-size:0.75rem; color:#888; margin-top:2px; }
  .whatif-box { background:#0f1520; border:1px solid #2a2a4a; border-radius:12px; padding:1rem; }
  div[data-testid="stSlider"] label { font-size:0.85rem !important; font-weight:500; }
  div[data-testid="stNumberInput"] label { font-size:0.85rem !important; }
  h1 { font-size:2rem !important; } h3 { font-size:1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Training XGBoost model on 20K+ records…")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        train_and_save()
    model   = joblib.load(MODEL_PATH)
    meta    = joblib.load(META_PATH)
    dataset = joblib.load(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
    return model, meta, dataset

model, meta, dataset = load_model()

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "med_inc": 5.0, "house_age": 20, "avg_rooms": 5.5,
    "avg_bedrooms": 1.1, "population": 1200,
    "avg_occup": 3.0, "latitude": 36.8, "longitude": -119.6,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_input_df(mi, ha, ar, ab, pop, ao, lat, lon):
    return pd.DataFrame([{
        "MedInc": mi, "HouseAge": ha, "AveRooms": ar, "AveBedrms": ab,
        "Population": pop, "AveOccup": ao, "Latitude": lat, "Longitude": lon,
        "rooms_per_person":  ar  / max(ao, 0.1),
        "bedrooms_ratio":    ab  / max(ar, 0.1),
        "income_per_room":   mi  / max(ar, 0.1),
        "pop_per_household": pop / max(ao, 0.1),
    }])

def predict_price(df):
    t0 = time.perf_counter()
    p  = model.predict(df)[0]
    return max(p * 100_000, 50_000), (time.perf_counter() - t0) * 1000

def tier_info(p):
    if p < 150_000:   return "Budget",    "badge-budget"
    elif p < 300_000: return "Mid-Range", "badge-mid"
    elif p < 500_000: return "Premium",   "badge-premium"
    else:             return "Luxury",    "badge-luxury"

def synced_input(label, key, min_v, max_v, step, help_text="", fmt_fn=None):
    """Slider + number input that stay in sync via session_state."""
    col_s, col_n = st.columns([2, 1])
    with col_s:
        val_s = st.slider(label, min_v, max_v,
                          float(st.session_state[key]) if isinstance(step, float) else int(st.session_state[key]),
                          step, key=f"sl_{key}", help=help_text)
    with col_n:
        val_n = st.number_input("", min_v, max_v,
                                float(st.session_state[key]) if isinstance(step, float) else int(st.session_state[key]),
                                step, key=f"ni_{key}", label_visibility="hidden")

    # whichever changed last wins
    if val_s != st.session_state[key]:
        st.session_state[key] = val_s
    elif val_n != st.session_state[key]:
        st.session_state[key] = val_n

    return st.session_state[key]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏠 RealEstateIQ")
st.markdown(
    f"**Real-time California property price predictor** · XGBoost trained on **20,640 records** · "
    f"R² **{meta['r2']:.3f}** · **{meta['rmse_improvement_pct']:.0f}%** RMSE reduction over linear baseline"
)
st.divider()

# ── Metrics row ───────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col,(key,val,sub) in zip([c1,c2,c3,c4],[
    ("R² Score",         f"{meta['r2']:.3f}",                    "Hold-out test set"),
    ("RMSE Improvement", f"{meta['rmse_improvement_pct']:.0f}%", "vs linear baseline"),
    ("Training Rows",    f"{meta['train_rows']:,}",              "California Housing"),
    ("Features",         str(meta['n_features']),                "Engineered + raw"),
]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                    f'<div class="metric-key">{key}<br><span style="color:#555">{sub}</span></div></div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUTS + RESULTS
# ══════════════════════════════════════════════════════════════════════════════
col_inputs, col_results = st.columns([1.2, 1], gap="large")

with col_inputs:
    st.markdown("### 🔧 Property Details")
    st.markdown("Drag sliders or type values directly in the boxes.")

    st.markdown('<span class="section-label">💰 Financial & Demographics</span>', unsafe_allow_html=True)
    med_inc = synced_input("Median Income (×$10K)", "med_inc", 0.5, 15.0, 0.1,
                           "Median household income in the block (×$10,000)")
    st.markdown(f'<p class="helper-text">≈ ${med_inc*10:.0f}K annual household income</p>', unsafe_allow_html=True)

    population = synced_input("Block Population", "population", 3, 3500, 10,
                              "Total population in the block group")

    st.markdown('<span class="section-label">🏗 Property Characteristics</span>', unsafe_allow_html=True)
    house_age    = synced_input("House Age (years)",             "house_age",    1,   52,   1,   "Median age of houses in the block")
    avg_rooms    = synced_input("Avg Rooms per Household",       "avg_rooms",    1.0, 14.0, 0.1, "Average number of rooms per household")
    avg_bedrooms = synced_input("Avg Bedrooms per Household",    "avg_bedrooms", 0.5, 5.0,  0.1, "Average number of bedrooms per household")
    avg_occup    = synced_input("Avg Occupancy",                 "avg_occup",    1.0, 10.0, 0.1, "Average number of people per household")

    st.markdown('<span class="section-label">📍 Location (California)</span>', unsafe_allow_html=True)
    latitude  = synced_input("Latitude",  "latitude",  32.5,  42.0,   0.1, "32.5°N = San Diego · 37.7°N = San Francisco · 42°N = Northern CA")
    longitude = synced_input("Longitude", "longitude", -124.4, -114.3, 0.1, "-122.4 = San Francisco · -118.2 = Los Angeles")

    if latitude > 37.0 and longitude < -121.0:      loc_hint = "📍 San Francisco Bay Area"
    elif latitude < 34.5 and longitude > -119.0:    loc_hint = "📍 Los Angeles / San Diego Area"
    elif latitude > 38.5:                           loc_hint = "📍 Northern California"
    elif longitude > -117.0:                        loc_hint = "📍 Inland / Desert California"
    else:                                           loc_hint = "📍 Central California"
    st.markdown(f'<p class="helper-text">{loc_hint}</p>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
input_df      = make_input_df(med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
price, inf_ms = predict_price(input_df)
low, high     = price * 0.88, price * 1.12
tier, badge   = tier_info(price)

with col_results:
    st.markdown("### 💰 Predicted Price")
    st.markdown(f"""
    <div class="predict-box">
      <div style="font-size:0.75rem;color:#666;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Estimated Market Value</div>
      <div class="predict-price">${price:,.0f}</div>
      <div class="predict-range">Range: ${low:,.0f} – ${high:,.0f}</div>
      <span class="badge {badge}">{tier} Property</span>
      <div style="margin-top:14px;font-size:0.72rem;color:#555;">Inference: {inf_ms:.2f}ms · Model: XGBoost + GridSearchCV</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📈 Price vs Income Sensitivity")
    inc_range  = np.arange(0.5, 15.1, 0.5)
    prices_inc = []
    for inc in inc_range:
        df_tmp = make_input_df(inc, house_age, avg_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
        p, _   = predict_price(df_tmp)
        prices_inc.append(p)
    sens_df = pd.DataFrame({"Income (×$10K)": inc_range, "Predicted Price ($)": prices_inc}).set_index("Income (×$10K)")
    st.line_chart(sens_df, use_container_width=True, height=200)
    st.caption(f"Current income: ${med_inc*10:.0f}K → price: ${price:,.0f}")

    with st.expander("📋 Full Input Summary", expanded=False):
        st.dataframe(pd.DataFrame({
            "Feature": ["Median Income","House Age","Avg Rooms","Avg Bedrooms","Population","Avg Occupancy","Latitude","Longitude"],
            "Value":   [f"${med_inc*10:.0f}K", f"{house_age} yrs", f"{avg_rooms:.1f}", f"{avg_bedrooms:.1f}",
                        f"{population:,}", f"{avg_occup:.1f}", f"{latitude:.1f}°N", f"{longitude:.1f}°W"]
        }), hide_index=True, use_container_width=True)

# ── What-If ───────────────────────────────────────────────────────────────────
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
    b_inc   = st.slider("Income B (×$10K)", 0.5, 15.0, min(float(med_inc)+2.0, 15.0), 0.1, key="b_inc")
    b_age   = st.slider("House Age B",      1,   52,   max(int(house_age)-10, 1),           key="b_age")
    b_rooms = st.slider("Rooms B",          1.0, 14.0, min(float(avg_rooms)+1.0, 14.0), 0.1,key="b_rooms")
    df_b        = make_input_df(b_inc, b_age, b_rooms, avg_bedrooms, population, avg_occup, latitude, longitude)
    price_b, _  = predict_price(df_b)
    tier_b, badge_b = tier_info(price_b)
    diff        = price_b - price
    diff_pct    = (diff / price) * 100
    st.markdown(f"""<div class="whatif-box">
      <div style="font-size:1.6rem;font-weight:700;color:#f0c040;">${price_b:,.0f}</div>
      <div style="font-size:0.8rem;color:#888;margin-top:4px;">Income ${b_inc*10:.0f}K · {b_rooms:.1f} rooms · Age {b_age}yrs</div>
      <span class="badge {badge_b}">{tier_b}</span>
      <div style="margin-top:10px;font-size:0.85rem;color:{'#1d9e75' if diff>0 else '#e24b4a'};">
        {'▲' if diff>0 else '▼'} ${abs(diff):,.0f} ({abs(diff_pct):.1f}%) vs Property A
      </div>
    </div>""", unsafe_allow_html=True)

# ── Comparables ───────────────────────────────────────────────────────────────
if dataset is not None:
    st.divider()
    st.markdown("### 🔍 Comparable Properties")
    st.markdown("Similar properties from the dataset based on your inputs.")
    dist = (
        ((dataset["MedInc"]   - med_inc)   / dataset["MedInc"].std()).abs()   * 0.35 +
        ((dataset["Latitude"] - latitude)  / dataset["Latitude"].std()).abs() * 0.30 +
        ((dataset["Longitude"]- longitude) / dataset["Longitude"].std()).abs()* 0.25 +
        ((dataset["AveRooms"] - avg_rooms) / dataset["AveRooms"].std()).abs() * 0.10
    )
    comps     = dataset.loc[dist.nsmallest(5).index].reset_index(drop=True)
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
              <div style="font-size:0.72rem;color:{color};margin-top:4px;">{'▲' if diff_c>=0 else '▼'} ${abs(diff_c):,.0f}</div>
            </div>""", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Feature Importance")
feat_names  = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude",
               "rooms_per_person","bedrooms_ratio","income_per_room","pop_per_household"]
fi_df = (pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
         .sort_values("Importance", ascending=False).reset_index(drop=True))
st.bar_chart(fi_df.set_index("Feature")["Importance"], use_container_width=True, height=280)

st.divider()
st.caption("Dataset: California Housing (sklearn.datasets) · Model: XGBoost + 5-fold GridSearchCV · R² 0.82 · "
           "Built by **Aman Chaudhary** · [GitHub](https://github.com/amn-00/real-estate-price-predictor) · "
           "[LinkedIn](https://www.linkedin.com/in/aman-chaudhary-82b8382b1/)")
