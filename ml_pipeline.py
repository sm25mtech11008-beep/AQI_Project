"""
=============================================================
 Indian Urban Air Quality Intelligence System
 ML Pipeline — AQI Prediction & Health Risk Classification
 Dataset: 24 Indian Cities | 2019–2024 | 10,000 Records
 AQI computed via India CPCB sub-index methodology
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline

DATA_PATH = "/home/claude/AQI_Project/data/raw_data.csv"
OUT_DIR   = "/home/claude/AQI_Project/outputs"
MODEL_DIR = "/home/claude/AQI_Project/models"
PBI_DIR   = "/home/claude/AQI_Project/powerbi_data"
for d in [OUT_DIR, MODEL_DIR, PBI_DIR]:
    os.makedirs(d, exist_ok=True)

PALETTE = {
    'Good':'#2ECC71','Satisfactory':'#F1C40F','Moderate':'#E67E22',
    'Poor':'#E74C3C','Very Poor':'#8E44AD','Severe':'#2C3E50',
}
CITY_COLOR = '#3498DB'; BG = '#F8F9FA'; ACCENT = '#2C3E50'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':'#CED4DA',
    'axes.labelcolor':ACCENT,'xtick.color':ACCENT,'ytick.color':ACCENT,
    'text.color':ACCENT,'font.family':'DejaVu Sans','axes.grid':True,
    'grid.color':'#DEE2E6','grid.linewidth':0.5,
})

print("="*60)
print("  INDIAN AQI INTELLIGENCE SYSTEM — ML PIPELINE")
print("="*60)

# ── 1. LOAD & FEATURE ENGINEERING ──────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Loaded {len(df):,} records | {df['City'].nunique()} cities")

# CPCB Sub-index for PM2.5
def si_pm25(c):
    bp = [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,380,401,500)]
    for lo,hi,ilo,ihi in bp:
        if lo <= c <= hi: return ilo + (c-lo)/(hi-lo)*(ihi-ilo)
    return 500

# CPCB Sub-index for PM10
def si_pm10(c):
    bp = [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)]
    for lo,hi,ilo,ihi in bp:
        if lo <= c <= hi: return ilo + (c-lo)/(hi-lo)*(ihi-ilo)
    return 500

# CPCB Sub-index for NO2
def si_no2(c):
    bp = [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)]
    for lo,hi,ilo,ihi in bp:
        if lo <= c <= hi: return ilo + (c-lo)/(hi-lo)*(ihi-ilo)
    return 500

df['SI_PM25'] = df['PM2.5'].apply(si_pm25)
df['SI_PM10'] = df['PM10'].apply(si_pm10)
df['SI_NO2']  = df['NO2'].apply(si_no2)

# AQI = max sub-index (CPCB methodology)
df['AQI_Computed'] = df[['SI_PM25','SI_PM10','SI_NO2']].max(axis=1).clip(0,500).round(0).astype(int)

def aqi_cat(v):
    if v <= 50: return 'Good'
    elif v <= 100: return 'Satisfactory'
    elif v <= 200: return 'Moderate'
    elif v <= 300: return 'Poor'
    elif v <= 400: return 'Very Poor'
    else: return 'Severe'

df['AQI_Category'] = df['AQI_Computed'].apply(aqi_cat)

# Additional engineered features
df['Pollution_Load']  = (df['PM2.5']*0.40 + df['PM10']*0.25 + df['NO2']*0.15 +
                          df['SO2']*0.10 + df['CO']*0.07 + df['O3']*0.03)
df['Climate_Stress']  = (df['Temperature (°C)']*0.4 + df['Humidity (%)']*0.3 -
                          df['Wind Speed (km/h)']*0.2 - df['Rainfall (mm)']*0.1)
df['Urban_Pressure']  = (df['Vehicle Count']/1e5) * df['Industrial Activity Index']
df['PM_Ratio']        = df['PM2.5'] / (df['PM10'] + 1)
df['Oxidant_Load']    = df['O3'] + df['NO2']

def risk_tier(v):
    if v <= 100: return 'Low Risk'
    elif v <= 200: return 'Moderate Risk'
    elif v <= 300: return 'High Risk'
    else: return 'Critical Risk'
df['Risk_Tier'] = df['AQI_Computed'].apply(risk_tier)

metros = ['Delhi','Mumbai','Kolkata','Bangalore','Chennai','Hyderabad']
df['City_Tier'] = df['City'].apply(lambda c: 'Metro' if c in metros else 'Tier-2')

print("✓ CPCB sub-index AQI computed from PM2.5, PM10, NO2")
print("✓ 5 new engineered features added")

# ── 2. EDA CHARTS ───────────────────────────────────────────
print("\n[EDA] Generating charts...")
cat_order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']

# Chart 1: AQI Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('AQI Distribution — 24 Indian Cities (2019–2024)', fontsize=16, fontweight='bold')
counts = df['AQI_Category'].value_counts().reindex(cat_order, fill_value=0)
colors = [PALETTE[c] for c in cat_order]
axes[0].bar(cat_order, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Records by AQI Category (CPCB Standard)', fontweight='bold')
axes[0].set_xlabel('Category'); axes[0].set_ylabel('Count')
for i,v in enumerate(counts.values):
    if v > 0: axes[0].text(i, v+30, f'{v:,}', ha='center', fontsize=9, fontweight='bold')
axes[1].hist(df['AQI_Computed'], bins=50, color=CITY_COLOR, edgecolor='white', alpha=0.85)
axes[1].axvline(df['AQI_Computed'].mean(), color='#E74C3C', linestyle='--', lw=2,
                label=f'Mean={df["AQI_Computed"].mean():.0f}')
axes[1].axvline(df['AQI_Computed'].median(), color='#2ECC71', linestyle='--', lw=2,
                label=f'Median={df["AQI_Computed"].median():.0f}')
axes[1].set_title('AQI Frequency Distribution', fontweight='bold')
axes[1].set_xlabel('AQI (CPCB)'); axes[1].set_ylabel('Frequency'); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_aqi_distribution.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 01_aqi_distribution.png")

# Chart 2: City AQI comparison
city_aqi = df.groupby('City')['AQI_Computed'].agg(['mean','std']).sort_values('mean')
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(city_aqi.index, city_aqi['mean'], xerr=city_aqi['std'],
               color=CITY_COLOR, alpha=0.85, error_kw={'capsize':4,'elinewidth':1.5,'ecolor':'#7F8C8D'})
ax.axvline(df['AQI_Computed'].mean(), color='#E74C3C', linestyle='--', lw=2,
           label=f'National Avg = {df["AQI_Computed"].mean():.0f}')
for bar in bars:
    ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.0f}', va='center', fontsize=8.5, fontweight='bold')
ax.set_title('Average AQI by City ± 1 SD (CPCB Sub-index Method)', fontsize=14, fontweight='bold')
ax.set_xlabel('Average AQI'); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_city_aqi_comparison.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 02_city_aqi_comparison.png")

# Chart 3: Correlation heatmap
pollutants = ['AQI_Computed','PM2.5','PM10','NO2','CO','SO2','O3',
              'Pollution_Load','Urban_Pressure','Climate_Stress','Oxidant_Load']
corr = df[pollutants].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'shrink':0.8})
ax.set_title('Pollutant & Engineered Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 03_correlation_heatmap.png")

# Chart 4: PM2.5 vs AQI scatter by category
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for cat in cat_order:
    s = df[df['AQI_Category']==cat]
    axes[0].scatter(s['PM2.5'], s['AQI_Computed'], c=PALETTE[cat], label=cat, alpha=0.3, s=6)
axes[0].set_title('PM2.5 vs AQI (CPCB)', fontweight='bold')
axes[0].set_xlabel('PM2.5 (µg/m³)'); axes[0].set_ylabel('AQI')
axes[0].legend(markerscale=3, fontsize=9)
for cat in cat_order:
    s = df[df['AQI_Category']==cat]
    axes[1].scatter(s['Pollution_Load'], s['AQI_Computed'], c=PALETTE[cat], label=cat, alpha=0.3, s=6)
axes[1].set_title('Pollution Load Index vs AQI', fontweight='bold')
axes[1].set_xlabel('Pollution Load Index'); axes[1].set_ylabel('AQI')
axes[1].legend(markerscale=3, fontsize=9)
plt.suptitle('Pollutant Relationships with CPCB AQI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_pollutant_scatter.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 04_pollutant_scatter.png")

# Chart 5: Metro vs Tier-2 boxplot
fig, ax = plt.subplots(figsize=(12, 6))
tier_data = [df[df['City_Tier']=='Metro']['AQI_Computed'].values,
             df[df['City_Tier']=='Tier-2']['AQI_Computed'].values]
bp = ax.boxplot(tier_data, labels=['Metro Cities','Tier-2 Cities'],
                patch_artist=True, widths=0.5,
                medianprops={'color':'white','linewidth':2.5})
bp['boxes'][0].set_facecolor('#E74C3C'); bp['boxes'][1].set_facecolor('#3498DB')
ax.set_title('AQI Distribution: Metro vs Tier-2 Cities', fontsize=14, fontweight='bold')
ax.set_ylabel('AQI Value (CPCB)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_metro_vs_tier2.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 05_metro_vs_tier2.png")

# ── 3. MODEL 1: AQI REGRESSION ─────────────────────────────
print("\n[MODEL 1] AQI Regression (Random Forest + Gradient Boosting)...")

FEATURES = ['PM2.5','PM10','NO2','CO','SO2','O3',
            'Temperature (°C)','Humidity (%)','Wind Speed (km/h)',
            'Rainfall (mm)','Pressure (hPa)','Vehicle Count',
            'Industrial Activity Index','Pollution_Load',
            'Climate_Stress','Urban_Pressure','PM_Ratio','Oxidant_Load']

X     = df[FEATURES]
y_reg = df['AQI_Computed']

X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

rf_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestRegressor(n_estimators=200, max_depth=20,
                                      min_samples_split=5, random_state=42, n_jobs=-1))
])
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)
print(f"  RF Regressor → MAE={mae_rf:.2f} | RMSE={rmse_rf:.2f} | R²={r2_rf:.4f}")

gb_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  GradientBoostingRegressor(n_estimators=150, max_depth=6,
                                          learning_rate=0.1, random_state=42))
])
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb  = r2_score(y_test, y_pred_gb)
print(f"  GB Regressor → MAE={mae_gb:.2f} | R²={r2_gb:.4f}")

best_reg = rf_reg if r2_rf >= r2_gb else gb_reg
best_r2  = max(r2_rf, r2_gb)
best_mae = mae_rf if r2_rf >= r2_gb else mae_gb

# Chart 6: Regression performance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('AQI Prediction Model — Performance Analysis', fontsize=14, fontweight='bold')
axes[0].scatter(y_test, y_pred_rf, alpha=0.3, s=8, color=CITY_COLOR)
lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
axes[0].plot(lims, lims, 'r--', lw=2, label='Perfect Prediction')
axes[0].set_title('Actual vs Predicted AQI', fontweight='bold')
axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI')
axes[0].text(0.05, 0.9, f'R² = {r2_rf:.3f}', transform=axes[0].transAxes,
             fontsize=12, fontweight='bold', color='#E74C3C')
axes[0].legend()
residuals = y_test - y_pred_rf
axes[1].scatter(y_pred_rf, residuals, alpha=0.3, s=8, color='#9B59B6')
axes[1].axhline(0, color='#E74C3C', linestyle='--', lw=2)
axes[1].set_title('Residuals Plot', fontweight='bold')
axes[1].set_xlabel('Predicted AQI'); axes[1].set_ylabel('Residuals')
fi = rf_reg.named_steps['model'].feature_importances_
fi_df = pd.DataFrame({'Feature':FEATURES,'Importance':fi}).sort_values('Importance').tail(12)
axes[2].barh(fi_df['Feature'], fi_df['Importance'], color=CITY_COLOR, alpha=0.85)
axes[2].set_title('Top Feature Importances', fontweight='bold')
axes[2].set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_regression_performance.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 06_regression_performance.png")

# ── 4. MODEL 2: AQI CATEGORY CLASSIFIER ────────────────────
print("\n[MODEL 2] AQI Category Classifier...")

le = LabelEncoder()
y_cls = le.fit_transform(df['AQI_Category'])
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_cls, test_size=0.2,
                                                    random_state=42, stratify=y_cls)
rf_cls = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(n_estimators=200, max_depth=20,
                                       min_samples_split=5, random_state=42, n_jobs=-1))
])
rf_cls.fit(X_tr_c, y_tr_c)
y_pred_c = rf_cls.predict(X_te_c)
acc = accuracy_score(y_te_c, y_pred_c)
cv  = cross_val_score(rf_cls, X, y_cls, cv=5, scoring='accuracy', n_jobs=-1)
print(f"  RF Classifier  → Accuracy = {acc:.4f} ({acc*100:.1f}%)")
print(f"  5-Fold CV Acc  → {cv.mean():.4f} ± {cv.std():.4f}")
class_names = le.classes_

# Chart 7: Confusion matrix + per-class accuracy
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('AQI Category Classifier — Performance', fontsize=14, fontweight='bold')
cm = confusion_matrix(y_te_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title(f'Confusion Matrix (Accuracy={acc*100:.1f}%)', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
per_cls = cm.diagonal() / cm.sum(axis=1)
bar_colors = [PALETTE.get(c, CITY_COLOR) for c in class_names]
axes[1].bar(class_names, per_cls, color=bar_colors, edgecolor='white', lw=1.5)
axes[1].set_title('Per-Category Classification Accuracy', fontweight='bold')
axes[1].set_xlabel('AQI Category'); axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0,1.15)
for i,v in enumerate(per_cls):
    axes[1].text(i, v+0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_classifier_performance.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 07_classifier_performance.png")

# Chart 8: Feature importance for classifier
fi_cls = rf_cls.named_steps['model'].feature_importances_
fi_cls_df = pd.DataFrame({'Feature':FEATURES,'Importance':fi_cls}).sort_values('Importance',ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(fi_cls_df['Feature'], fi_cls_df['Importance'], color=CITY_COLOR,
              alpha=0.85, edgecolor='white', lw=1.5)
ax.set_title('Top 10 Features — AQI Category Prediction', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature'); ax.set_ylabel('Importance Score')
plt.xticks(rotation=35, ha='right')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 08_feature_importance.png")

# ── 5. MODEL 3: HEALTH RISK SCORE ──────────────────────────
print("\n[MODEL 3] Health Risk Score Regression...")

# Composite Health Risk Score (0-100) grounded in pollutant exposure
df['Health_Risk_Score'] = (
    (df['AQI_Computed']/500)*50 +
    (df['PM2.5']/250)*25 +
    (df['Pollution_Load']/df['Pollution_Load'].max())*15 +
    (df['Urban_Pressure']/df['Urban_Pressure'].max())*10
).clip(0,100).round(2)

def risk_level(s):
    if s < 20: return 'Low'
    elif s < 45: return 'Moderate'
    elif s < 70: return 'High'
    else: return 'Critical'
df['Health_Risk_Level'] = df['Health_Risk_Score'].apply(risk_level)

y_hr = df['Health_Risk_Score']
X_tr_hr, X_te_hr, y_tr_hr, y_te_hr = train_test_split(X, y_hr, test_size=0.2, random_state=42)
rf_hr = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
])
rf_hr.fit(X_tr_hr, y_tr_hr)
y_pred_hr = rf_hr.predict(X_te_hr)
r2_hr  = r2_score(y_te_hr, y_pred_hr)
mae_hr = mean_absolute_error(y_te_hr, y_pred_hr)
print(f"  Health Risk Model → MAE={mae_hr:.2f} | R²={r2_hr:.4f}")

# Chart 9: Health risk by city
city_risk = df.groupby('City')['Health_Risk_Score'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 7))
colors_risk = ['#E74C3C' if v>=50 else '#E67E22' if v>=35 else '#F1C40F' for v in city_risk.values]
ax.bar(city_risk.index, city_risk.values, color=colors_risk, edgecolor='white', lw=1.5)
ax.axhline(50, color='#E74C3C', linestyle='--', lw=2)
ax.set_title('Average Health Risk Score by City (0–100 Scale)', fontsize=14, fontweight='bold')
ax.set_xlabel('City'); ax.set_ylabel('Health Risk Score')
plt.xticks(rotation=45, ha='right')
patches = [mpatches.Patch(color='#E74C3C', label='High Risk (≥50)'),
           mpatches.Patch(color='#E67E22', label='Moderate (35–50)'),
           mpatches.Patch(color='#F1C40F', label='Lower (<35)')]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_health_risk_by_city.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 09_health_risk_by_city.png")

# Chart 10: Model comparison
fig, ax = plt.subplots(figsize=(10, 5))
models_names = ['AQI Regressor\n(Random Forest)','AQI Regressor\n(Gradient Boost)',
                'Category Classifier\n(Random Forest)','Health Risk\n(Random Forest)']
scores = [r2_rf, r2_gb, acc, r2_hr]
colors_m = ['#3498DB','#2ECC71','#9B59B6','#E67E22']
bars = ax.bar(models_names, scores, color=colors_m, edgecolor='white', lw=2, width=0.6)
ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score / Accuracy'); ax.set_ylim(0,1.1)
for bar, val in zip(bars, scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ 10_model_comparison.png")

# ── 6. SAVE MODELS ──────────────────────────────────────────
print("\n[SAVE] Persisting models...")
for name, obj in [('rf_aqi_regressor',rf_reg),('gb_aqi_regressor',gb_reg),
                   ('rf_aqi_classifier',rf_cls),('rf_health_risk',rf_hr),
                   ('label_encoder',le)]:
    with open(f"{MODEL_DIR}/{name}.pkl",'wb') as f: pickle.dump(obj,f)
print("  ✓ 5 model/encoder files saved")

# ── 7. POWER BI EXPORTS ─────────────────────────────────────
print("\n[POWER BI] Exporting enriched CSVs...")

df['AQI_Predicted']         = rf_reg.predict(X)
df['AQI_Cat_Predicted']     = le.inverse_transform(rf_cls.predict(X))
df['Health_Risk_Predicted'] = rf_hr.predict(X)

df.to_csv(f"{PBI_DIR}/fact_aqi_enriched.csv", index=False)
print("  ✓ fact_aqi_enriched.csv")

city_summary = df.groupby('City').agg(
    Avg_AQI=('AQI_Computed','mean'), Max_AQI=('AQI_Computed','max'),
    Min_AQI=('AQI_Computed','min'), Avg_PM25=('PM2.5','mean'),
    Avg_PM10=('PM10','mean'), Avg_Health_Risk=('Health_Risk_Score','mean'),
    Records=('AQI_Computed','count'),
    Pct_Severe=('AQI_Category', lambda x: (x=='Severe').mean()*100),
    City_Tier=('City_Tier','first'),
).reset_index().round(2)
city_summary.to_csv(f"{PBI_DIR}/dim_city_summary.csv", index=False)
print("  ✓ dim_city_summary.csv")

cat_dist = df.groupby(['City','AQI_Category']).size().reset_index(name='Count')
cat_dist['Pct'] = cat_dist.groupby('City')['Count'].transform(lambda x: x/x.sum()*100).round(1)
cat_dist.to_csv(f"{PBI_DIR}/dim_category_distribution.csv", index=False)
print("  ✓ dim_category_distribution.csv")

df.groupby(['City','Risk_Tier']).size().reset_index(name='Count').to_csv(
    f"{PBI_DIR}/dim_risk_tier.csv", index=False)
print("  ✓ dim_risk_tier.csv")

df[['City','PM2.5','PM10','NO2','CO','SO2','O3','Pollution_Load']].groupby('City')\
  .mean().reset_index().round(2).to_csv(f"{PBI_DIR}/dim_pollutant_by_city.csv", index=False)
print("  ✓ dim_pollutant_by_city.csv")

pd.DataFrame({
    'Model':['RF AQI Regressor','GB AQI Regressor','RF Category Classifier','RF Health Risk'],
    'Type':['Regression','Regression','Classification','Regression'],
    'Metric':['R²','R²','Accuracy','R²'],
    'Score':[round(r2_rf,4),round(r2_gb,4),round(acc,4),round(r2_hr,4)],
    'MAE':[round(mae_rf,2),round(mae_gb,2),'N/A',round(mae_hr,2)],
}).to_csv(f"{PBI_DIR}/model_metrics.csv", index=False)
print("  ✓ model_metrics.csv")

# ── 8. SUMMARY ──────────────────────────────────────────────
print("\n"+"="*60)
print("  PROJECT COMPLETE — FINAL METRICS")
print("="*60)
print(f"  Dataset       : 10,000 records | 24 cities | 18 features")
print(f"  AQI Method    : CPCB sub-index (PM2.5, PM10, NO2)")
print(f"  Model 1 (RF)  : AQI Regression  → R²={r2_rf:.3f} | MAE={mae_rf:.1f}")
print(f"  Model 2 (GB)  : AQI Regression  → R²={r2_gb:.3f} | MAE={mae_gb:.1f}")
print(f"  Model 3 (RF)  : AQI Category    → Acc={acc*100:.1f}% | CV={cv.mean()*100:.1f}%")
print(f"  Model 4 (RF)  : Health Risk      → R²={r2_hr:.3f} | MAE={mae_hr:.2f}")
print(f"  Charts        : 10 publication-ready PNG files")
print(f"  Power BI Data : 5 enriched CSVs")
print(f"  Models Saved  : 5 .pkl files")
print("="*60)
