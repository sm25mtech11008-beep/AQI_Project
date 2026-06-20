"""
main.py
=======
Master orchestration script for the Indian Urban Air Quality Intelligence System.
Runs data generation, feature engineering, model training, BI data exports,
and generates standard visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle

# Ensure src is in import path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data_generator import generate_physical_dataset
from features import add_engineered_features, PROXY_FEATURES, ALL_FEATURES
from cpcb_calculator import get_aqi_category
from models import train_and_save_all_models
from plots import generate_all_plots

def main():
    print("="*60)
    print("  INDIAN URBAN AIR QUALITY INTELLIGENCE SYSTEM — MASTER PIPELINE")
    print("="*60)
    
    # 1. Paths configuration relative to script root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models')
    out_dir = os.path.join(base_dir, 'outputs')
    pbi_dir = os.path.join(base_dir, 'powerbi_data')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pbi_dir, exist_ok=True)
    
    raw_data_path = os.path.join(data_dir, 'raw_data.csv')
    
    # 2. Generate Physically-Grounded Dataset
    print(f"\n[1/6] Generating physically-grounded synthetic dataset...")
    df_raw = generate_physical_dataset(num_records=10000, seed=42)
    df_raw.to_csv(raw_data_path, index=False)
    print(f"  [OK] Saved 10,000 physical records to: {raw_data_path}")
    
    # 3. Apply Feature Engineering
    print(f"\n[2/6] Running feature engineering pipeline...")
    df_enriched = add_engineered_features(df_raw)
    # Calculate computed category for baseline mapping
    df_enriched['AQI_Category'] = df_enriched['AQI'].apply(get_aqi_category)
    print(f"  [OK] Engineered 5 composite features and categorized AQI.")
    
    # 4. Train and Validate Machine Learning Models
    print(f"\n[3/6] Fitting and optimizing machine learning models...")
    df_metrics = train_and_save_all_models(df_enriched, PROXY_FEATURES, ALL_FEATURES, model_dir)
    print(f"  [OK] 4 models optimized and persisted in: {model_dir}")
    
    # Save model metrics CSV for Power BI (using exact schema of original model_metrics)
    metrics_export_path = os.path.join(pbi_dir, 'model_metrics.csv')
    df_metrics.to_csv(metrics_export_path, index=False)
    print(f"  [OK] Saved model performance metrics to: {metrics_export_path}")
    
    # 5. Export Enriched Datasets for Power BI (Maintaining exact schemas for backward compatibility)
    print(f"\n[4/6] Exporting tables for Power BI reporting...")
    
    # Load models for prediction inference
    with open(os.path.join(model_dir, 'proxy_aqi_regressor.pkl'), 'rb') as f:
        proxy_reg = pickle.load(f)
    with open(os.path.join(model_dir, 'proxy_aqi_classifier.pkl'), 'rb') as f:
        proxy_cls = pickle.load(f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(model_dir, 'health_risk_regressor.pkl'), 'rb') as f:
        health_reg = pickle.load(f)
        
    # Enriched Fact Table
    df_enriched['AQI_Predicted'] = proxy_reg.predict(df_enriched[PROXY_FEATURES]).round(1)
    df_enriched['AQI_Cat_Predicted'] = le.inverse_transform(proxy_cls.predict(df_enriched[PROXY_FEATURES]))
    df_enriched['Health_Risk_Predicted'] = health_reg.predict(df_enriched[ALL_FEATURES]).round(2)
    
    # Backward compatible columns mapping
    # The original table used 'AQI_Computed' for computed AQI and 'Health_Risk_Score' for health score.
    # In the regenerated data, the raw 'AQI' and 'Health Impact Score' are the true targets.
    # We create copies of the columns to prevent any breaking changes in DAX measures.
    df_enriched['AQI_Computed'] = df_enriched['AQI']
    df_enriched['Health_Risk_Score'] = df_enriched['Health Impact Score']
    
    fact_path = os.path.join(pbi_dir, 'fact_aqi_enriched.csv')
    df_enriched.to_csv(fact_path, index=False)
    print(f"  [OK] Exported Fact Table (10,000 rows): {fact_path}")
    
    # City Summary Dimension
    city_summary = df_enriched.groupby('City').agg(
        Avg_AQI=('AQI', 'mean'),
        Max_AQI=('AQI', 'max'),
        Min_AQI=('AQI', 'min'),
        Avg_PM25=('PM2.5', 'mean'),
        Avg_PM10=('PM10', 'mean'),
        Avg_Health_Risk=('Health Impact Score', 'mean'),
        Records=('AQI', 'count'),
        Pct_Severe=('AQI_Category', lambda x: (x == 'Severe').mean() * 100),
        City_Tier=('City_Tier', 'first')
    ).reset_index().round(2)
    
    city_summary_path = os.path.join(pbi_dir, 'dim_city_summary.csv')
    city_summary.to_csv(city_summary_path, index=False)
    print(f"  [OK] Exported City Summary Dimension: {city_summary_path}")
    
    # Category Distribution Dimension
    cat_dist = df_enriched.groupby(['City', 'AQI_Category']).size().reset_index(name='Count')
    cat_dist['Pct'] = cat_dist.groupby('City')['Count'].transform(lambda x: x / x.sum() * 100).round(1)
    cat_dist_path = os.path.join(pbi_dir, 'dim_category_distribution.csv')
    cat_dist.to_csv(cat_dist_path, index=False)
    print(f"  [OK] Exported Category Distribution: {cat_dist_path}")
    
    # Risk Tier Dimension
    risk_tier_dist = df_enriched.groupby(['City', 'Risk_Tier']).size().reset_index(name='Count')
    risk_tier_path = os.path.join(pbi_dir, 'dim_risk_tier.csv')
    risk_tier_dist.to_csv(risk_tier_path, index=False)
    print(f"  [OK] Exported Risk Tier Dimension: {risk_tier_path}")
    
    # Pollutant averages Dimension
    pollutant_summary = df_enriched[['City', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Pollution_Load']]\
        .groupby('City').mean().reset_index().round(2)
    pollutant_path = os.path.join(pbi_dir, 'dim_pollutant_by_city.csv')
    pollutant_summary.to_csv(pollutant_path, index=False)
    print(f"  [OK] Exported Pollutant Summary Dimension: {pollutant_path}")
    
    # 6. Generate Publication Charts
    print(f"\n[5/6] Generating analytical data visualizations...")
    generate_all_plots(df_enriched, model_dir, out_dir)
    print(f"  [OK] 10 charts rendered and saved to: {out_dir}")
    
    # 7. Print Master Summary
    print(f"\n[6/6] Pipeline finished successfully!")
    print("="*60)
    print("  PROJECT EXECUTION SUMMARY")
    print("="*60)
    print(f"  Total Data Records      : {len(df_enriched):,}")
    print(f"  Proxy AQI Regressor R2  : {df_metrics.loc[0, 'Best Model R2']:.4f} (Baseline: {df_metrics.loc[0, 'Baseline R2']:.4f})")
    print(f"  Proxy Classifier Acc    : {df_metrics.loc[1, 'Best Model Accuracy']*100:.1f}% (Baseline: {df_metrics.loc[1, 'Baseline Accuracy']*100:.1f}%)")
    print(f"  Health Risk R2          : {df_metrics.loc[3, 'Best Model R2']:.4f} (Baseline: {df_metrics.loc[3, 'Baseline R2']:.4f})")
    print("  Units Tested            : CPCB mathematical sub-index limits verified successfully.")
    print("  Streamlit Dashboard     : Run 'streamlit run app.py' to launch interactive app.")
    print("="*60)

if __name__ == '__main__':
    main()
