# 🌫️ Indian Urban Air Quality Intelligence System
### ML-Powered Proxy AQI Prediction & Health Exposure Analytics |

![Python](https://img.shields.io/badge/Python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange) !
---

## 📌 Project Overview

This is an end-to-end machine learning and business intelligence project designed to predict the **Air Quality Index (AQI)** and assess compound **Health Risk Scores** for 24 major Indian cities. 

### 💡 Core Engineering Highlights (Placement-Ready)
To ensure production standards and theoretical rigor for data science interviews, this project implements:
1. **Physically-Grounded Atmospheric Simulation**: Re-engineered a synthetic dataset of 10,000 records modeling actual atmospheric dynamics (e.g., wind speed dilution, rainfall particulate scavenging/washout, and temperature-driven Ozone photochemistry).
2. **Elimination of Target Leakage**: Rather than using chemical pollutants ($PM_{2.5}, PM_{10}, NO_2$) to trivially approximate the deterministic CPCB AQI formula ($R^2 = 1.0$), this system trains a **Proxy Predictor** using *only* weather and traffic density metrics (sensor-free air quality estimation).
3. **Spatial Cross-Validation (GroupKFold)**: Standard random train-test splitting shuffles records from the same cities into both sets, causing spatial leakage. We evaluate all models using **GroupKFold cross-validation grouped by City** to prove the models generalize to *completely unseen cities*.
4. **Interactive Streamlit Portfolio Application**: A web dashboard for recruiters to perform real-time inferences with meteorological sliders, explore model performance charts, and review data distributions.

---

## 🤖 Model Performance & Architecture

The pipeline compares simple baselines (Mean predictors, Linear Regression, Decision Trees) against optimized ensemble methods (Random Forest, Gradient Boosting) tuned via `RandomizedSearchCV`.

| Model Scenario | Target | Features Used | Baseline Score | Optimized Model Score |
|----------------|--------|---------------|----------------|-----------------------|
| **Proxy AQI Regressor** | AQI value (0-500) | Weather & Traffic Only | $R^2 = 0.817$ (Linear Reg) | **$R^2 = 0.956$** (Gradient Boosting) |
| **Proxy AQI Classifier** | AQI Category (6 classes) | Weather & Traffic Only | Accuracy = 83.0% (Decision Tree) | **Accuracy = 89.4%** (Random Forest) |
| **PM2.5 Sensor Estimator**| PM2.5 (µg/m³) | Weather + Other Pollutants | $R^2 = 0.825$ (Linear Reg) | **$R^2 = 0.955$** (Random Forest) |
| **Health Risk Score** | Exposure Score (0-100)| All Features | $R^2 = 0.947$ (Linear Reg) | **$R^2 = 0.998$** (Random Forest) |

*All metrics are evaluated via GroupKFold cross-validation to ensure spatial robustness.*

---

## 📁 Project Structure

```text
AQI_Project/
│
├── data/
│   └── raw_data.csv                  # Physically-grounded dataset (10,000 records)
│
├── src/                              # Modular python package
│   ├── cpcb_calculator.py            # Indian CPCB sub-index math interpolation
│   ├── data_generator.py             # Meteorological & emission simulator
│   ├── features.py                   # Feature engineering pipeline
│   ├── models.py                     # Model training, GroupKFold, and tuning
│   └── plots.py                      # Matplotlib & Seaborn visualization scripts
│
├── outputs/                          # 10 publication-ready analytical charts
│   ├── 01_aqi_distribution.png
│   ├── 02_city_aqi_comparison.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_pollutant_scatter.png
│   ├── 05_metro_vs_tier2.png
│   ├── 06_regression_performance.png
│   ├── 07_classifier_performance.png
│   ├── 08_feature_importance.png
│   ├── 09_health_risk_by_city.png
│   └── 10_model_comparison.png
│
├── models/                           # Saved serialized models (.pkl)
│   ├── proxy_aqi_regressor.pkl       # Best Model A (Gradient Boosting)
│   ├── proxy_aqi_classifier.pkl      # Best Model B (Random Forest)
│   ├── pm25_sensor_estimator.pkl     # Best Model C (Random Forest)
│   ├── health_risk_regressor.pkl     # Best Model D (Random Forest)
│   └── label_encoder.pkl             # AQI Category Encoder
│
├── powerbi_data/                     # Power BI import files (backward-compatible)
│   ├── fact_aqi_enriched.csv         # Main enriched table with predictions
│   ├── dim_city_summary.csv          # City KPIs
│   ├── dim_category_distribution.csv # Category % by city
│   ├── dim_risk_tier.csv             # Risk tier breakdown
│   ├── dim_pollutant_by_city.csv     # Pollutant averages
│   └── model_metrics.csv             # Model performance comparatives
│
├── tests/
│   └── test_cpcb.py                  # Unit tests verifying CPCB calculation bounds
│
├── app.py                            # Interactive Streamlit Web Portfolio App
├── main.py                           # Unified pipeline execution script
├── requirements.txt                  # Dependency list
└── README.md
```

---

## 🚀 How to Run

### 1. Installation
Install all project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
To regenerate the physical dataset, perform feature engineering, train/tune all models, generate evaluation reports, and export files:
```bash
python main.py
```

### 3. Run Unit Tests
Verify CPCB calculator logic and code integrity:
```bash
python -m unittest tests/test_cpcb.py
```

### 4. Launch the Interactive Web Dashboard
Run the Streamlit application to showcase the model to recruiters:
```bash
streamlit run app.py
```

---

## 📈 Key Atmospheric Insights

- **Wind Dilution**: High wind speed acts as a primary dispersion driver, resulting in a strong negative correlation with the overall AQI ($r = -0.26$).
- **Rain Washout**: Coarse particulate matter ($PM_{10}$) shows faster wet deposition rates than fine particulates ($PM_{2.5}$) during monsoon simulations.
- **Regional Climate Swings**: Mountain regions (Srinagar) show low atmospheric dispersion compared to coastal cities (Mumbai, Chennai) where strong sea breezes buffer pollutant accumulation.

---


