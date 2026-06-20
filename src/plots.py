"""
plots.py
========
Generates publication-quality charts for the Indian Urban AQI System.
Features cohesive styling and matches colors across analysis views.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import r2_score, accuracy_score

PALETTE = {
    'Good': '#2ECC71', 'Satisfactory': '#F1C40F', 'Moderate': '#E67E22',
    'Poor': '#E74C3C', 'Very Poor': '#8E44AD', 'Severe': '#2C3E50'
}
BG = '#F8F9FA'
ACCENT = '#2C3E50'
PRIMARY = '#3498DB'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': '#CED4DA',
    'axes.labelcolor': ACCENT, 'xtick.color': ACCENT, 'ytick.color': ACCENT,
    'text.color': ACCENT, 'font.family': 'DejaVu Sans', 'axes.grid': True,
    'grid.color': '#DEE2E6', 'grid.linewidth': 0.5
})

def generate_all_plots(df, model_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cat_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    
    # ---------------- Plot 1: AQI Distribution ----------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('AQI Distribution — 24 Indian Cities', fontsize=16, fontweight='bold')
    
    counts = df['AQI_Category'].value_counts().reindex(cat_order, fill_value=0)
    colors = [PALETTE[c] for c in cat_order]
    axes[0].bar(cat_order, counts.values, color=colors, edgecolor='white', linewidth=1.2)
    axes[0].set_title('Records by AQI Category (CPCB Standard)', fontweight='bold')
    axes[0].set_xlabel('Category'); axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        if v > 0:
            axes[0].text(i, v + 30, f'{v:,}', ha='center', fontsize=9, fontweight='bold')
            
    axes[1].hist(df['AQI'], bins=50, color=PRIMARY, edgecolor='white', alpha=0.85)
    mean_aqi = df['AQI'].mean()
    median_aqi = df['AQI'].median()
    axes[1].axvline(mean_aqi, color='#E74C3C', linestyle='--', lw=2, label=f'Mean={mean_aqi:.1f}')
    axes[1].axvline(median_aqi, color='#2ECC71', linestyle='--', lw=2, label=f'Median={median_aqi:.1f}')
    axes[1].set_title('AQI Frequency Distribution', fontweight='bold')
    axes[1].set_xlabel('AQI (CPCB)'); axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_aqi_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 2: City AQI Comparison ----------------
    city_aqi = df.groupby('City')['AQI'].agg(['mean', 'std']).sort_values('mean')
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(city_aqi.index, city_aqi['mean'], xerr=city_aqi['std'],
                   color=PRIMARY, alpha=0.85, error_kw={'capsize': 4, 'elinewidth': 1.2, 'ecolor': '#7F8C8D'})
    ax.axvline(mean_aqi, color='#E74C3C', linestyle='--', lw=2, label=f'National Avg = {mean_aqi:.1f}')
    for bar in bars:
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.1f}', va='center', fontsize=8.5, fontweight='bold')
    ax.set_title('Average AQI by City +/- 1 SD (CPCB Methodology)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average AQI')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_city_aqi_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 3: Correlation Heatmap ----------------
    pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Temperature (°C)',
                  'Humidity (%)', 'Wind Speed (km/h)', 'Rainfall (mm)', 'Vehicle Count', 'Industrial Activity Index']
    corr = df[pollutants].corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix of Atmospheric & Urban Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 4: Wind Speed vs AQI & PM2.5 Scatter ----------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Wind Speed vs AQI
    axes[0].scatter(df['Wind Speed (km/h)'], df['AQI'], color='#9B59B6', alpha=0.3, s=8)
    axes[0].set_title('Atmospheric Dispersion: Wind Speed vs AQI', fontweight='bold')
    axes[0].set_xlabel('Wind Speed (km/h)'); axes[0].set_ylabel('AQI')
    
    # PM2.5 vs AQI color coded by CPCB categories
    for cat in cat_order:
        s = df[df['AQI_Category'] == cat]
        axes[1].scatter(s['PM2.5'], s['AQI'], c=PALETTE[cat], label=cat, alpha=0.4, s=8)
    axes[1].set_title('Particulate Concentration: PM2.5 vs AQI', fontweight='bold')
    axes[1].set_xlabel('PM2.5 (µg/m³)'); axes[1].set_ylabel('AQI')
    axes[1].legend(markerscale=3)
    
    plt.suptitle('Meteorological and Particle Relationships with AQI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_pollutant_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 5: Metro vs Tier-2 Boxplot ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    tier_data = [df[df['City_Tier'] == 'Metro']['AQI'].values,
                 df[df['City_Tier'] == 'Tier-2']['AQI'].values]
    bp = ax.boxplot(tier_data, labels=['Metro Cities', 'Tier-2 Cities'], patch_artist=True,
                    widths=0.5, medianprops={'color': 'white', 'linewidth': 2.5})
    bp['boxes'][0].set_facecolor('#E74C3C')
    bp['boxes'][1].set_facecolor(PRIMARY)
    ax.set_title('AQI Ranges: Metros vs Industrial Tier-2 Cities', fontsize=14, fontweight='bold')
    ax.set_ylabel('AQI (CPCB)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/05_metro_vs_tier2.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Load Model A and predictions for Plot 6 & 8
    with open(f"{model_dir}/proxy_aqi_regressor.pkl", 'rb') as f:
        proxy_reg = pickle.load(f)
    
    from features import PROXY_FEATURES
    X_proxy = df[PROXY_FEATURES]
    y_aqi = df['AQI']
    y_pred = proxy_reg.predict(X_proxy)
    
    # ---------------- Plot 6: Regression performance (Proxy Regressor) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Proxy AQI Regressor (Weather & Traffic Only) — Performance', fontsize=14, fontweight='bold')
    
    axes[0].scatter(y_aqi, y_pred, alpha=0.3, s=8, color=PRIMARY)
    lims = [min(y_aqi.min(), y_pred.min()), max(y_aqi.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
    r2_val = r2_score(y_aqi, y_pred)
    axes[0].text(0.05, 0.9, f'R2 = {r2_val:.3f}', transform=axes[0].transAxes, fontsize=12, fontweight='bold', color='#E74C3C')
    axes[0].set_title('Actual vs Predicted AQI', fontweight='bold')
    axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI'); axes[0].legend()
    
    residuals = y_aqi - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color='#9B59B6')
    axes[1].axhline(0, color='#E74C3C', linestyle='--', lw=2)
    axes[1].set_title('Residuals Plot', fontweight='bold')
    axes[1].set_xlabel('Predicted AQI'); axes[1].set_ylabel('Residual')
    
    # Feature importances
    if hasattr(proxy_reg.named_steps['model'], 'feature_importances_'):
        fi = proxy_reg.named_steps['model'].feature_importances_
    else:
        # Linear regression coefficients as fallback
        fi = np.abs(proxy_reg.named_steps['model'].coef_)
        
    fi_df = pd.DataFrame({'Feature': PROXY_FEATURES, 'Importance': fi}).sort_values('Importance').tail(8)
    axes[2].barh(fi_df['Feature'], fi_df['Importance'], color=PRIMARY, alpha=0.85)
    axes[2].set_title('Top Proxy Feature Importances', fontweight='bold')
    axes[2].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_regression_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Load Model B for Plot 7
    with open(f"{model_dir}/proxy_aqi_classifier.pkl", 'rb') as f:
        proxy_cls = pickle.load(f)
    with open(f"{model_dir}/label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
        
    y_cat_true = le.transform(df['AQI_Category'])
    y_cat_pred = proxy_cls.predict(X_proxy)
    acc_val = accuracy_score(y_cat_true, y_cat_pred)
    
    # ---------------- Plot 7: Confusion Matrix ----------------
    from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_cat_true, y_cat_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'Proxy Category Classifier Confusion Matrix\n(Accuracy = {acc_val*100:.1f}%)', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_classifier_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 8: Feature Importance for Classifier ----------------
    fi_cls = proxy_cls.named_steps['model'].feature_importances_
    fi_cls_df = pd.DataFrame({'Feature': PROXY_FEATURES, 'Importance': fi_cls}).sort_values('Importance', ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(fi_cls_df['Feature'], fi_cls_df['Importance'], color=PRIMARY, alpha=0.85, edgecolor='white', lw=1.2)
    ax.set_title('Top Proxy Features for AQI Category Classification', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature'); ax.set_ylabel('Importance Score')
    plt.xticks(rotation=20, ha='right')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/08_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 9: Health Risk by City ----------------
    city_risk = df.groupby('City')['Health Impact Score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors_risk = ['#E74C3C' if v >= 60 else '#E67E22' if v >= 40 else '#F1C40F' for v in city_risk.values]
    ax.bar(city_risk.index, city_risk.values, color=colors_risk, edgecolor='white', lw=1.2)
    ax.axhline(60, color='#E74C3C', linestyle='--', lw=2, label='Critical Risk Threshold (≥60)')
    ax.set_title('Average Health Risk Score by City (0-100 Toxic Exposure Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('City'); ax.set_ylabel('Avg Health Risk Score')
    plt.xticks(rotation=45, ha='right')
    
    patches = [
        mpatches.Patch(color='#E74C3C', label='High Risk (≥60)'),
        mpatches.Patch(color='#E67E22', label='Moderate Risk (40-60)'),
        mpatches.Patch(color='#F1C40F', label='Low Risk (<40)')
    ]
    ax.legend(handles=patches)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/09_health_risk_by_city.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------------- Plot 10: Model Comparison ----------------
    # Read metrics file
    metrics_path = os.path.join(os.path.dirname(model_dir), 'powerbi_data', 'model_metrics.csv')
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        scores = []
        labels = []
        for _, row in m_df.iterrows():
            labels.append(row['Model'] + f"\n({row['Type']})")
            scores.append(row['Best Model R2'] if row['Type'] == 'Regression' else row['Best Model Accuracy'])
            
        colors_m = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22']
        bars = ax.bar(labels, scores, color=colors_m[:len(scores)], edgecolor='white', lw=1.5, width=0.5)
        ax.set_title('Improved Models Cross-Validation Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('CV Score (R2 / Accuracy)')
        ax.set_ylim(0, 1.1)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontweight='bold', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/10_model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
