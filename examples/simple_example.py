"""
Simple example demonstrating basic usage of NeuraTrade
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuratrade import TradingAgent, DataFetcher
from neuratrade.agent import TradingEnvironment


def main():
    """Simple example of training and testing a trading agent"""
    
    print("=" * 60)
    print("NeuraTrade - Simple Example")
    print("=" * 60)
    
    # 1. Fetch stock data
    print("\n1. Fetching stock data...")
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data('AAPL', '2022-01-01', '2023-12-31')
    
    # 2. Create trading environment
    print("\n2. Creating trading environment...")
    env = TradingEnvironment(data)
    
    # 3. Create trading agent
    print("\n3. Creating trading agent...")
    agent = TradingAgent()
    
    # 4. Train agent (quick training with 10 episodes)
    print("\n4. Training agent (10 episodes)...")
    history = agent.train(env, episodes=10)
    
    # 5. Test agent
    print("\n5. Testing agent...")
    # Reset environment for testing
    env.reset()
    results = agent.test(env)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try examples/train_agent.py for full training")
    print("  - Try examples/backtest_agent.py for backtesting")
    print("  - Experiment with different stocks and parameters")


if __name__ == '__main__':
    main()
