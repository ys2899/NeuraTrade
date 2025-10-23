"""
Technical indicators for stock analysis
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculate various technical indicators for trading signals"""
    
    @staticmethod
    def calculate_sma(data, window=20):
        """
        Calculate Simple Moving Average
        
        Args:
            data (pd.Series): Price data
            window (int): Window size
            
        Returns:
            pd.Series: SMA values
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data, window=20):
        """
        Calculate Exponential Moving Average
        
        Args:
            data (pd.Series): Price data
            window (int): Window size
            
        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """
        Calculate Relative Strength Index
        
        Args:
            data (pd.Series): Price data
            window (int): Window size
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pd.Series): Price data
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.Series): Price data
            window (int): Window size
            num_std (int): Number of standard deviations
            
        Returns:
            tuple: (Upper band, Middle band, Lower band)
        """
        middle_band = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(high, low, close, window=14):
        """
        Calculate Average True Range
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            window (int): Window size
            
        Returns:
            pd.Series: ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(close, volume):
        """
        Calculate On-Balance Volume
        
        Args:
            close (pd.Series): Close prices
            volume (pd.Series): Volume data
            
        Returns:
            pd.Series: OBV values
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
