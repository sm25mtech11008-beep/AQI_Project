"""
data_generator.py
=================
Generates a physically-grounded synthetic dataset for 24 Indian cities
to simulate realistic correlations between meteorology, urban activity,
chemical pollutants, AQI, and health impact scores.
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure src can import other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpcb_calculator import calculate_aqi

def generate_physical_dataset(num_records=10000, seed=42):
    np.random.seed(seed)
    
    cities = [
        'Delhi', 'Mumbai', 'Kolkata', 'Bangalore', 'Chennai', 'Hyderabad',
        'Rajkot', 'Bhopal', 'Srinagar', 'Jaipur', 'Nashik', 'Vadodara',
        'Varanasi', 'Patna', 'Agra', 'Meerut', 'Nagpur', 'Thane',
        'Pune', 'Ludhiana', 'Ahmedabad', 'Indore', 'Lucknow', 'Surat'
    ]
    
    metros = {'Delhi', 'Mumbai', 'Kolkata', 'Bangalore', 'Chennai', 'Hyderabad'}
    industrial_hubs = {'Thane', 'Nashik', 'Vadodara', 'Ahmedabad', 'Surat', 'Ludhiana'}
    coastal_cities = {'Mumbai', 'Chennai', 'Surat'}
    mountain_cities = {'Srinagar'}
    
    data = []
    
    for i in range(num_records):
        city = np.random.choice(cities)
        
        # 1. Weather Baselines & Generation
        # Regional weather variations
        if city in mountain_cities:
            temp_mean, temp_std = 14, 8      # Cooler mountain climate
            humid_mean, humid_std = 70, 10    # Humid valley
            wind_mean, wind_std = 6, 2        # Low wind dispersion
        elif city in coastal_cities:
            temp_mean, temp_std = 29, 3      # Warm and stable
            humid_mean, humid_std = 78, 7     # Very humid coastal air
            wind_mean, wind_std = 16, 4       # Strong sea breeze
        else:
            temp_mean, temp_std = 26, 9      # Inland extreme seasonal swings
            humid_mean, humid_std = 45, 15    # Drier
            wind_mean, wind_std = 10, 3       # Moderate wind
            
        temp = np.random.normal(temp_mean, temp_std)
        # Humidity negatively correlated with temp
        humid_noise = np.random.normal(0, 5)
        humidity = np.clip(humid_mean - 0.8 * (temp - temp_mean) + humid_noise, 15, 98)
        
        # Wind speed log-normal to prevent negative winds
        wind = np.random.lognormal(mean=np.log(wind_mean), sigma=0.25)
        
        # Rainfall zero-inflated (dry most days, wet monsoon days)
        is_raining = np.random.rand() < (0.25 if city in coastal_cities else 0.12)
        rainfall = np.random.exponential(scale=15.0) if is_raining else 0.0
        
        # Pressure negatively correlated with temperature and altitude
        pressure = 1013.25 - 0.12 * temp - (150 if city in mountain_cities else 0) + np.random.normal(0, 2)
        
        # 2. Urban Activity Parameters
        # Vehicle count (metros have vastly more cars)
        veh_base = 350000 if city in metros else 80000
        vehicles = int(np.random.normal(veh_base, veh_base * 0.25))
        vehicles = max(5000, vehicles)
        
        # Industrial Index (0 to 10 scale)
        ind_base = 8.0 if city in industrial_hubs else (5.0 if city in metros else 3.5)
        ind_activity = np.clip(np.random.normal(ind_base, 1.2), 0.5, 10.0)
        
        # 3. Pollutant Concentration Emission Modeling (Physically Grounded)
        # We start with basic emission formulas driven by traffic and industry
        pm25_emissions = 15 + (vehicles * 0.00018) + (ind_activity * 9.0)
        pm10_emissions = 30 + (vehicles * 0.00028) + (ind_activity * 15.0)
        no2_emissions  = 8  + (vehicles * 0.00012) + (ind_activity * 4.0)
        co_emissions   = 0.4 + (vehicles * 0.000012) + (ind_activity * 0.1)
        so2_emissions  = 2  + (ind_activity * 4.5)
        
        # Meteorological Dispersion Factor (high wind dilutes pollutants)
        # Dispersion index scales down pollutants
        dispersion = 1.0 + (wind - 5.0) * 0.075
        dispersion = max(0.4, dispersion)
        
        # Rain Washout (scavenging effect of particulates)
        washout_pm25 = max(0.2, 1.0 - 0.025 * rainfall)
        washout_pm10 = max(0.1, 1.0 - 0.04 * rainfall) # Coarser dust washes out faster
        
        # Apply physical effects
        pm25 = (pm25_emissions / dispersion) * washout_pm25 + np.random.normal(0, 4)
        pm10 = (pm10_emissions / dispersion) * washout_pm10 + np.random.normal(0, 8)
        no2  = (no2_emissions / dispersion) + np.random.normal(0, 2)
        co   = (co_emissions / dispersion) + np.random.normal(0, 0.1)
        so2  = (so2_emissions / dispersion) + np.random.normal(0, 0.8)
        
        # Ozone (photochemical reaction driven by sunlight/temp and precursor NOx/NO2)
        # O3 is higher on hot, dry, low-wind days
        o3_production = 15 + 1.2 * temp - 0.2 * humidity + 1.5 * no2
        o3 = (o3_production / dispersion) + np.random.normal(0, 3)
        
        # Ensure values are strictly positive and physically possible
        pm25 = max(1.5, pm25)
        pm10 = max(3.0, pm10)
        no2 = max(1.0, no2)
        co = max(0.05, co)
        so2 = max(0.5, so2)
        o3 = max(1.0, o3)
        
        # 4. Target CPCB AQI Calculation
        computed_aqi = calculate_aqi(pm25, pm10, no2, co, so2, o3)
        
        # Real-world AQI is reported with minor sensor/averaging noise
        reported_aqi = int(np.clip(computed_aqi + np.random.normal(0, 1.5), 1, 500))
        
        # 5. Health Impact Score Modeling (0 to 100)
        # Based on toxicological dose-response curve to AQI & particulate exposure
        # High AQI, high PM2.5 concentration, and high humidity worsen health impact
        health_logits = -3.5 + 0.02 * computed_aqi + 0.015 * pm25 + 0.005 * humidity
        health_prob = 1.0 / (1.0 + np.exp(-health_logits))
        health_score = int(round(health_prob * 100))
        
        data.append({
            'City': city,
            'AQI': reported_aqi,
            'PM2.5': round(pm25, 4),
            'PM10': round(pm10, 4),
            'NO2': round(no2, 4),
            'CO': round(co, 4),
            'SO2': round(so2, 4),
            'O3': round(o3, 4),
            'Temperature (°C)': round(temp, 4),
            'Humidity (%)': round(humidity, 4),
            'Wind Speed (km/h)': round(wind, 4),
            'Rainfall (mm)': round(rainfall, 4),
            'Pressure (hPa)': round(pressure, 4),
            'Vehicle Count': vehicles,
            'Industrial Activity Index': round(ind_activity, 4),
            'Health Impact Score': health_score
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # Test generation of 10 rows
    df_sample = generate_physical_dataset(10)
    print("Sample generated rows:")
    print(df_sample[['City', 'AQI', 'PM2.5', 'Wind Speed (km/h)', 'Rainfall (mm)', 'Health Impact Score']])
    
    # Save the actual dataset to check correlations
    df_full = generate_physical_dataset(10000)
    print("\nCorrelations in generated data:")
    print(df_full.corr(numeric_only=True)['AQI'].round(3))
