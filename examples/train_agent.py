"""
Example script for training a trading agent on US stock data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuratrade import TradingAgent, DataFetcher
from neuratrade.agent import TradingEnvironment
from neuratrade.config import Config
import matplotlib.pyplot as plt


def train_trading_agent(ticker='AAPL', start_date='2020-01-01', end_date='2023-12-31', episodes=50):
    """
    Train a trading agent on historical stock data
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        episodes (int): Number of training episodes
    """
    print(f"\n{'='*60}")
    print(f"Training Trading Agent for {ticker}")
    print(f"{'='*60}")
    
    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(ticker, start_date, end_date)
    
    # Split data into train and test
    split_idx = int(len(data) * Config.TRAIN_TEST_SPLIT)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nData split:")
    print(f"Training data: {len(train_data)} days")
    print(f"Testing data: {len(test_data)} days")
    
    # Create environments
    train_env = TradingEnvironment(train_data)
    test_env = TradingEnvironment(test_data)
    
    # Create and train agent
    agent = TradingAgent()
    
    # Train
    history = agent.train(train_env, episodes=episodes)
    
    # Test
    test_results = agent.test(test_env)
    
    # Plot training progress
    plot_training_results(history, ticker)
    
    # Save model
    model_path = f'models/{ticker}_trading_agent.h5'
    os.makedirs('models', exist_ok=True)
    agent.save(model_path)
    
    return agent, history, test_results


def plot_training_results(history, ticker):
    """Plot training results"""
    episodes = [h['episode'] for h in history]
    rewards = [h['total_reward'] for h in history]
    profits = [h['profit_pct'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(episodes, rewards, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'Training Progress - {ticker}')
    ax1.grid(True, alpha=0.3)
    
    # Plot profit percentage
    ax2.plot(episodes, profits, 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Profit (%)')
    ax2.set_title(f'Profit Percentage per Episode - {ticker}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_training_results.png', dpi=150)
    print(f"\nTraining plot saved to results/{ticker}_training_results.png")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a stock trading agent')
    parser.add_argument('--ticker', type=str, default='AAPL', 
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='Start date (default: 2020-01-01)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='End date (default: 2023-12-31)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes (default: 50)')
    
    args = parser.parse_args()
    
    agent, history, results = train_trading_agent(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        episodes=args.episodes
    )
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}")
