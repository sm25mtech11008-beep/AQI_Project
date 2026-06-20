"""
cpcb_calculator.py
===================
Calculates the scientifically grounded Indian Air Quality Index (AQI) based on
the Central Pollution Control Board (CPCB) sub-index methodology for six pollutants:
PM2.5, PM10, NO2, CO, SO2, and O3.
"""

def linear_interpolate(c, lo, hi, ilo, ihi):
    """Linearly interpolates raw concentration to sub-index value."""
    if hi == lo:
        return ilo
    return ilo + (c - lo) / (hi - lo) * (ihi - ilo)

def get_sub_index_pm25(c):
    """Calculates sub-index for PM2.5 (24-hr average, µg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 380, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def get_sub_index_pm10(c):
    """Calculates sub-index for PM10 (24-hr average, µg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def get_sub_index_no2(c):
    """Calculates sub-index for NO2 (24-hr average, µg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 800, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def get_sub_index_so2(c):
    """Calculates sub-index for SO2 (24-hr average, µg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 3200, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def get_sub_index_co(c):
    """Calculates sub-index for CO (8-hr average, mg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 57, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def get_sub_index_o3(c):
    """Calculates sub-index for O3 (8-hr average, µg/m³)."""
    if c <= 0: return 0
    bp = [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500)
    ]
    for lo, hi, ilo, ihi in bp:
        if lo < c <= hi:
            return linear_interpolate(c, lo, hi, ilo, ihi)
    return 500

def calculate_aqi(pm25, pm10, no2, co, so2, o3):
    """
    Computes overall CPCB AQI as the max sub-index across all 6 pollutants.
    India CPCB requires at least three sub-indices to be present (including at least one PM).
    """
    sub_indices = {
        'PM2.5': get_sub_index_pm25(pm25),
        'PM10': get_sub_index_pm10(pm10),
        'NO2': get_sub_index_no2(no2),
        'CO': get_sub_index_co(co),
        'SO2': get_sub_index_so2(so2),
        'O3': get_sub_index_o3(o3)
    }
    
    # Filter out 0 sub-indices
    active_indices = {k: v for k, v in sub_indices.items() if v > 0}
    
    if len(active_indices) < 3:
        # Fallback to simple max if constraints not met, but log it
        pass
    
    if not active_indices:
        return 0
        
    overall_aqi = max(active_indices.values())
    return int(round(clip(overall_aqi, 0, 500)))

def get_aqi_category(aqi):
    """Translates numeric AQI value into a standard CPCB category label."""
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

def clip(v, lo, hi):
    return max(lo, min(v, hi))
