"""
features.py
===========
Applies feature engineering to the raw air quality dataset.
Supports creating distinct feature sets for proxy-based (weather/urban only)
and full sensor-based prediction tasks.
"""

import pandas as pd
import numpy as np

def add_engineered_features(df):
    """
    Applies feature engineering and returns a copy of the dataframe
    with engineered features added.
    """
    df = df.copy()
    
    # 1. Weather-Urban Proxy Features (No chemical sensors required)
    df['Climate_Stress'] = (
        df['Temperature (°C)'] * 0.4 + 
        df['Humidity (%)'] * 0.3 - 
        df['Wind Speed (km/h)'] * 0.2 - 
        df['Rainfall (mm)'] * 0.1
    )
    df['Urban_Pressure'] = (df['Vehicle Count'] / 1e5) * df['Industrial Activity Index']
    
    # 2. Chemical Pollutant Features (Requires chemical sensors)
    df['Pollution_Load'] = (
        df['PM2.5'] * 0.40 + 
        df['PM10'] * 0.25 + 
        df['NO2'] * 0.15 + 
        df['SO2'] * 0.10 + 
        df['CO'] * 0.07 + 
        df['O3'] * 0.03
    )
    df['PM_Ratio'] = df['PM2.5'] / (df['PM10'] + 1)
    df['Oxidant_Load'] = df['O3'] + df['NO2']
    
    # 3. Categorical Encodings & Metros
    metros = {'Delhi', 'Mumbai', 'Kolkata', 'Bangalore', 'Chennai', 'Hyderabad'}
    df['City_Tier'] = df['City'].apply(lambda c: 'Metro' if c in metros else 'Tier-2')
    
    # Risk tiers based on AQI
    def get_risk_tier(aqi):
        if aqi <= 100: return 'Low Risk'
        elif aqi <= 200: return 'Moderate Risk'
        elif aqi <= 300: return 'High Risk'
        else: return 'Critical Risk'
    df['Risk_Tier'] = df['AQI'].apply(get_risk_tier)
    
    return df

# Define feature sets
PROXY_FEATURES = [
    'Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 
    'Rainfall (mm)', 'Pressure (hPa)', 'Vehicle Count', 
    'Industrial Activity Index', 'Climate_Stress', 'Urban_Pressure'
]

POLLUTANT_FEATURES = [
    'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3',
    'Pollution_Load', 'PM_Ratio', 'Oxidant_Load'
]

ALL_FEATURES = PROXY_FEATURES + POLLUTANT_FEATURES
