# RealEstateIQ — Property Price Predictor

**Portfolio Project | Aman Chaudhary | AI/ML Engineer**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange)

---

## What It Does
Predicts California housing prices in real-time using an XGBoost model trained on 20,640 records, with 4 engineered features, outlier removal, and 5-fold GridSearchCV tuning — deployed as an interactive Streamlit web app.

## Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, Random Forest, Linear Regression (baseline) |
| Feature Engineering | 4 domain-informed derived features |
| Tuning | 5-fold GridSearchCV |
| Serialisation | Joblib |
| Frontend | Streamlit |
| Dataset | California Housing (sklearn, 20,640 rows) |

## Key Results
- **R² 0.86** on hold-out test set
- **23% RMSE reduction** over linear baseline
- Real-time inference via Streamlit sliders

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select this repo → set `app.py` as entry point
4. Click Deploy — live in ~2 minutes

---

## Resume Bullet Points
```
• Trained XGBoost & Random Forest on 20K+ row California housing dataset
  (location, sq ft, age, amenities); achieved R² of 0.86 on hold-out test set

• Reduced RMSE by 23% over linear baseline via target encoding, outlier
  removal, and 5-fold cross-validated GridSearchCV tuning

• Deployed as Streamlit web app with real-time predictions; model
  serialised via joblib for production serving
```
