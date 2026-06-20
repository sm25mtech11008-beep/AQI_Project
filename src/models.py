"""
models.py
=========
Trains, tunes, and evaluates ML models for the AQI Intelligence System.
Includes GroupKFold CV by city, baseline comparisons, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

def evaluate_regressor_group_cv(model, X, y, groups, cv_splits=5):
    """Evaluates a regressor using GroupKFold cross-validation by city."""
    gkf = GroupKFold(n_splits=cv_splits)
    r2_scores = []
    mae_scores = []
    
    # We convert to numpy for safe indexing with KFold splits
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    groups_arr = groups.values if isinstance(groups, pd.Series) else groups
    
    for train_idx, test_idx in gkf.split(X_arr, y_arr, groups_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        # Clone the model pipeline
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        
        r2_scores.append(r2_score(y_te, preds))
        mae_scores.append(mean_absolute_error(y_te, preds))
        
    return np.mean(r2_scores), np.std(r2_scores), np.mean(mae_scores), np.std(mae_scores)

def evaluate_classifier_group_cv(model, X, y, groups, cv_splits=5):
    """Evaluates a classifier using GroupKFold cross-validation by city."""
    gkf = GroupKFold(n_splits=cv_splits)
    acc_scores = []
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    groups_arr = groups.values if isinstance(groups, pd.Series) else groups
    
    for train_idx, test_idx in gkf.split(X_arr, y_arr, groups_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        
        acc_scores.append(accuracy_score(y_te, preds))
        
    return np.mean(acc_scores), np.std(acc_scores)

def tune_random_forest_regressor(X, y, groups):
    """Tunes hyperparameters of Random Forest Regressor using GroupKFold CV."""
    print("  Tuning Random Forest hyperparameters...")
    param_dist = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [8, 12, 16, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    gkf = GroupKFold(n_splits=3)
    
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=6,
        cv=gkf, scoring='r2', random_state=42, n_jobs=-1
    )
    search.fit(X, y, groups=groups)
    print(f"  Best params: {search.best_params_}")
    print(f"  Best tuning R2: {search.best_score_:.4f}")
    return search.best_estimator_

def train_and_save_all_models(df, proxy_features, all_features, model_dir):
    """Trains all improved model configurations and saves them to model_dir."""
    os.makedirs(model_dir, exist_ok=True)
    metrics = []
    
    groups = df['City']
    
    # ----------------- MODEL A: Proxy AQI Regressor (Weather/Traffic -> AQI) -----------------
    print("\n--- Training Model A: Proxy AQI Regressor ---")
    X_proxy = df[proxy_features]
    y_aqi = df['AQI']
    
    # Baseline 1: Dummy Regressor (predicts mean)
    dummy_reg = Pipeline([('scaler', StandardScaler()), ('model', DummyRegressor(strategy='mean'))])
    d_r2, _, d_mae, _ = evaluate_regressor_group_cv(dummy_reg, X_proxy, y_aqi, groups)
    print(f"  Baseline (Mean) -> CV R2 = {d_r2:.4f} | MAE = {d_mae:.2f}")
    
    # Baseline 2: Linear Regression
    lr_reg = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    lr_r2, _, lr_mae, _ = evaluate_regressor_group_cv(lr_reg, X_proxy, y_aqi, groups)
    print(f"  Baseline (Linear Regression) -> CV R2 = {lr_r2:.4f} | MAE = {lr_mae:.2f}")
    
    # Candidate 1: Gradient Boosting
    gb_reg = Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))])
    gb_r2, _, gb_mae, _ = evaluate_regressor_group_cv(gb_reg, X_proxy, y_aqi, groups)
    print(f"  Gradient Boosting Regressor -> CV R2 = {gb_r2:.4f} | MAE = {gb_mae:.2f}")
    
    # Candidate 2: Tuned Random Forest
    tuned_rf_reg = tune_random_forest_regressor(X_proxy, y_aqi, groups)
    rf_r2, rf_r2_std, rf_mae, rf_mae_std = evaluate_regressor_group_cv(tuned_rf_reg, X_proxy, y_aqi, groups)
    print(f"  Tuned Random Forest Regressor -> CV R2 = {rf_r2:.4f} +/- {rf_r2_std:.4f} | MAE = {rf_mae:.2f} +/- {rf_mae_std:.2f}")
    
    # Select Best Regressor for Proxy
    best_proxy_reg = tuned_rf_reg if rf_r2 >= gb_r2 else gb_reg
    best_proxy_reg.fit(X_proxy, y_aqi)
    
    # Save Model A
    with open(os.path.join(model_dir, 'proxy_aqi_regressor.pkl'), 'wb') as f:
        pickle.dump(best_proxy_reg, f)
    metrics.append({
        'Model': 'Proxy AQI Regressor', 'Type': 'Regression', 
        'Baseline R2': round(lr_r2, 4), 'Best Model R2': round(max(rf_r2, gb_r2), 4),
        'Baseline MAE': round(lr_mae, 2), 'Best Model MAE': round(min(rf_mae, gb_mae), 2)
    })

    # ----------------- MODEL B: Proxy AQI Classifier (Weather/Traffic -> AQI Cat) -----------------
    print("\n--- Training Model B: Proxy AQI Classifier ---")
    le_cat = LabelEncoder()
    y_cat = le_cat.fit_transform(df['AQI_Category'])
    
    # Baseline: Dummy Classifier
    dummy_cls = Pipeline([('scaler', StandardScaler()), ('model', DummyClassifier(strategy='most_frequent'))])
    d_acc, _ = evaluate_classifier_group_cv(dummy_cls, X_proxy, y_cat, groups)
    print(f"  Baseline (Most Frequent) -> CV Acc = {d_acc*100:.1f}%")
    
    # Candidate 1: Decision Tree
    dt_cls = Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeClassifier(max_depth=6, random_state=42))])
    dt_acc, _ = evaluate_classifier_group_cv(dt_cls, X_proxy, y_cat, groups)
    print(f"  Decision Tree Classifier -> CV Acc = {dt_acc*100:.1f}%")
    
    # Candidate 2: Random Forest
    rf_cls = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
    ])
    rf_acc, rf_acc_std = evaluate_classifier_group_cv(rf_cls, X_proxy, y_cat, groups)
    print(f"  Random Forest Classifier -> CV Acc = {rf_acc*100:.1f}% +/- {rf_acc_std*100:.1f}%")
    
    best_proxy_cls = rf_cls if rf_acc >= dt_acc else dt_cls
    best_proxy_cls.fit(X_proxy, y_cat)
    
    # Save Model B & Label Encoder
    with open(os.path.join(model_dir, 'proxy_aqi_classifier.pkl'), 'wb') as f:
        pickle.dump(best_proxy_cls, f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le_cat, f)
        
    metrics.append({
        'Model': 'Proxy AQI Classifier', 'Type': 'Classification', 
        'Baseline Accuracy': round(dt_acc, 4), 'Best Model Accuracy': round(max(rf_acc, dt_acc), 4),
        'Baseline MAE': 'N/A', 'Best Model MAE': 'N/A'
    })

    # ----------------- MODEL C: PM2.5 Estimator (Proxy + Other Pollutants -> PM2.5) -----------------
    print("\n--- Training Model C: PM2.5 Missing Sensor Estimator ---")
    pm25_features = [f for f in all_features if f not in {'PM2.5', 'Pollution_Load', 'PM_Ratio'}]
    X_pm25 = df[pm25_features]
    y_pm25 = df['PM2.5']
    
    lr_pm25 = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    lr_r2_pm, _, lr_mae_pm, _ = evaluate_regressor_group_cv(lr_pm25, X_pm25, y_pm25, groups)
    
    rf_pm25 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    rf_r2_pm, rf_r2_pm_std, rf_mae_pm, _ = evaluate_regressor_group_cv(rf_pm25, X_pm25, y_pm25, groups)
    print(f"  Random Forest PM2.5 Estimator -> CV R2 = {rf_r2_pm:.4f} +/- {rf_r2_pm_std:.4f} | MAE = {rf_mae_pm:.2f}")
    
    rf_pm25.fit(X_pm25, y_pm25)
    with open(os.path.join(model_dir, 'pm25_sensor_estimator.pkl'), 'wb') as f:
        pickle.dump(rf_pm25, f)
        
    metrics.append({
        'Model': 'PM2.5 Sensor Estimator', 'Type': 'Regression', 
        'Baseline R2': round(lr_r2_pm, 4), 'Best Model R2': round(rf_r2_pm, 4),
        'Baseline MAE': round(lr_mae_pm, 2), 'Best Model MAE': round(rf_mae_pm, 2)
    })

    # ----------------- MODEL D: Health Risk Regressor (All Features -> Health Impact Score) -----------------
    print("\n--- Training Model D: Health Risk Regressor ---")
    X_all = df[all_features]
    y_health = df['Health Impact Score']
    
    lr_health = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    lr_r2_h, _, lr_mae_h, _ = evaluate_regressor_group_cv(lr_health, X_all, y_health, groups)
    
    rf_health = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ])
    rf_r2_h, rf_r2_h_std, rf_mae_h, _ = evaluate_regressor_group_cv(rf_health, X_all, y_health, groups)
    print(f"  Random Forest Health Risk -> CV R2 = {rf_r2_h:.4f} +/- {rf_r2_h_std:.4f} | MAE = {rf_mae_h:.2f}")
    
    rf_health.fit(X_all, y_health)
    with open(os.path.join(model_dir, 'health_risk_regressor.pkl'), 'wb') as f:
        pickle.dump(rf_health, f)
        
    metrics.append({
        'Model': 'Health Risk Regressor', 'Type': 'Regression', 
        'Baseline R2': round(lr_r2_h, 4), 'Best Model R2': round(rf_r2_h, 4),
        'Baseline MAE': round(lr_mae_h, 2), 'Best Model MAE': round(rf_mae_h, 2)
    })
    
    # Save comparison metrics as CSV for Power BI
    df_metrics = pd.DataFrame(metrics)
    
    return df_metrics
