import unittest
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cpcb_calculator import (
    get_sub_index_pm25,
    get_sub_index_pm10,
    get_sub_index_no2,
    get_aqi_category,
    calculate_aqi
)

class TestCPCBAQICalculator(unittest.TestCase):
    
    def test_pm25_calculation(self):
        # Good range (0-30 -> 0-50)
        self.assertAlmostEqual(get_sub_index_pm25(15), 25.0)
        
        # Satisfactory range (30-60 -> 51-100)
        self.assertAlmostEqual(get_sub_index_pm25(45), 75.5)
        
        # Severe range boundary
        self.assertAlmostEqual(get_sub_index_pm25(380), 500.0)
        
        # Out of bounds (capped at 500)
        self.assertEqual(get_sub_index_pm25(400), 500)

    def test_pm10_calculation(self):
        # Good range (0-50 -> 0-50)
        self.assertAlmostEqual(get_sub_index_pm10(25), 25.0)
        
        # Moderate range (100-250 -> 101-200)
        self.assertAlmostEqual(get_sub_index_pm10(175), 150.5)

    def test_no2_calculation(self):
        # Good range (0-40 -> 0-50)
        self.assertAlmostEqual(get_sub_index_no2(20), 25.0)

    def test_aqi_category_mapping(self):
        self.assertEqual(get_aqi_category(35), 'Good')
        self.assertEqual(get_aqi_category(75), 'Satisfactory')
        self.assertEqual(get_aqi_category(150), 'Moderate')
        self.assertEqual(get_aqi_category(250), 'Poor')
        self.assertEqual(get_aqi_category(350), 'Very Poor')
        self.assertEqual(get_aqi_category(450), 'Severe')

    def test_overall_aqi_calculation(self):
        # Max of the three indices should drive the AQI
        # PM2.5 = 15 -> Index = 25
        # PM10 = 25 -> Index = 25
        # NO2 = 20 -> Index = 25
        self.assertEqual(calculate_aqi(15, 25, 20, 0, 0, 0), 25)
        
        # PM2.5 = 45 -> Index = 76 (rounded)
        # PM10 = 25 -> Index = 25
        # NO2 = 20 -> Index = 25
        # Overall AQI should be max = 76
        self.assertEqual(calculate_aqi(45, 25, 20, 0, 0, 0), 76)

if __name__ == '__main__':
    unittest.main()
