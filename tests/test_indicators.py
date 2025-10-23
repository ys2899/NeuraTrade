"""
Unit tests for Technical Indicators
"""

import unittest
import numpy as np
import pandas as pd
from neuratrade.utils.indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data
        np.random.seed(42)
        self.data = pd.Series(100 + np.cumsum(np.random.randn(100)))
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        sma = TechnicalIndicators.calculate_sma(self.data, window=20)
        
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.data))
        
        # SMA should be NaN for first 19 values
        self.assertTrue(np.isnan(sma.iloc[:19]).all())
        
        # SMA should have values after window
        self.assertFalse(np.isnan(sma.iloc[19]))
    
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation"""
        ema = TechnicalIndicators.calculate_ema(self.data, window=20)
        
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(self.data))
    
    def test_calculate_rsi(self):
        """Test Relative Strength Index calculation"""
        rsi = TechnicalIndicators.calculate_rsi(self.data, window=14)
        
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.data))
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        macd, signal, histogram = TechnicalIndicators.calculate_macd(self.data)
        
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        
        self.assertEqual(len(macd), len(self.data))
        self.assertEqual(len(signal), len(self.data))
        self.assertEqual(len(histogram), len(self.data))
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            self.data, window=20, num_std=2
        )
        
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # Upper band should be above middle, middle above lower
        valid_indices = ~upper.isna()
        self.assertTrue((upper[valid_indices] >= middle[valid_indices]).all())
        self.assertTrue((middle[valid_indices] >= lower[valid_indices]).all())
    
    def test_calculate_atr(self):
        """Test Average True Range calculation"""
        # Create sample high, low, close data
        high = self.data + np.random.rand(len(self.data))
        low = self.data - np.random.rand(len(self.data))
        close = self.data
        
        atr = TechnicalIndicators.calculate_atr(high, low, close, window=14)
        
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.data))
        
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr >= 0).all())
    
    def test_calculate_obv(self):
        """Test On-Balance Volume calculation"""
        volume = pd.Series(np.random.randint(1000, 10000, len(self.data)))
        
        obv = TechnicalIndicators.calculate_obv(self.data, volume)
        
        self.assertIsInstance(obv, pd.Series)
        self.assertEqual(len(obv), len(self.data))


if __name__ == '__main__':
    unittest.main()
