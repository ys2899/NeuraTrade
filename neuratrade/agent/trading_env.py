"""
Trading environment for reinforcement learning
"""

import numpy as np
import pandas as pd
from neuratrade.config.config import Config
from neuratrade.utils.indicators import TechnicalIndicators


class TradingEnvironment:
    """Trading environment that simulates stock trading"""
    
    def __init__(self, data, initial_balance=None):
        """
        Initialize trading environment
        
        Args:
            data (pd.DataFrame): Historical stock data
            initial_balance (float): Initial balance in USD
        """
        self.data = data
        self.initial_balance = initial_balance or Config.INITIAL_BALANCE
        self.current_step = 0
        self.reset()
        
        # Calculate technical indicators
        self._prepare_indicators()
    
    def _prepare_indicators(self):
        """Calculate and store technical indicators"""
        close_prices = self.data['Close']
        
        self.sma = TechnicalIndicators.calculate_sma(close_prices, window=20)
        self.rsi = TechnicalIndicators.calculate_rsi(close_prices, window=14)
        macd, signal, _ = TechnicalIndicators.calculate_macd(close_prices)
        self.macd = macd
        self.macd_signal = signal
        
        # Fill NaN values
        self.sma = self.sma.fillna(close_prices)
        self.rsi = self.rsi.fillna(50)
        self.macd = self.macd.fillna(0)
        self.macd_signal = self.macd_signal.fillna(0)
    
    def reset(self):
        """Reset the environment to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = Config.WINDOW_SIZE
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        
        Returns:
            np.array: State vector containing market and portfolio information
        """
        if self.current_step >= len(self.data):
            return None
        
        current_price = self.data['Close'].iloc[self.current_step]
        
        # Normalize values
        price_norm = current_price / 1000.0  # Simple normalization
        balance_norm = self.balance / self.initial_balance
        shares_norm = self.shares_held / Config.MAX_SHARES_PER_TRADE
        
        # Technical indicators (normalized)
        sma_ratio = current_price / (self.sma.iloc[self.current_step] + 1e-10)
        rsi_norm = self.rsi.iloc[self.current_step] / 100.0
        macd_norm = np.tanh(self.macd.iloc[self.current_step] / 10.0)
        macd_signal_norm = np.tanh(self.macd_signal.iloc[self.current_step] / 10.0)
        
        # Price change indicators
        price_changes = []
        for i in range(1, min(4, self.current_step + 1)):
            prev_price = self.data['Close'].iloc[self.current_step - i]
            change = (current_price - prev_price) / (prev_price + 1e-10)
            price_changes.append(change)
        
        # Pad if necessary
        while len(price_changes) < 3:
            price_changes.append(0)
        
        state = np.array([
            price_norm,
            balance_norm,
            shares_norm,
            sma_ratio,
            rsi_norm,
            macd_norm,
            macd_signal_norm,
            price_changes[0],
            price_changes[1],
            price_changes[2]
        ])
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action (int): 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}
        
        current_price = self.data['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            shares_to_buy = min(
                Config.MAX_SHARES_PER_TRADE,
                int(self.balance / (current_price * (1 + Config.COMMISSION_FEE)))
            )
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + Config.COMMISSION_FEE)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
        
        elif action == 2:  # Sell
            shares_to_sell = min(Config.MAX_SHARES_PER_TRADE, self.shares_held)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - Config.COMMISSION_FEE)
                self.balance += revenue
                self.shares_held -= shares_to_sell
                self.total_shares_sold += shares_to_sell
                self.total_sales_value += revenue
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': revenue
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth
        if self.current_step < len(self.data):
            current_price = self.data['Close'].iloc[self.current_step]
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held
        }
        
        return next_state, reward, done, info
    
    def get_portfolio_value(self):
        """Get current portfolio value"""
        if self.current_step < len(self.data):
            current_price = self.data['Close'].iloc[self.current_step]
            return self.balance + self.shares_held * current_price
        return self.balance
