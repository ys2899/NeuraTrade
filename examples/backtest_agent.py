"""
Example script for backtesting a trained trading agent
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuratrade import TradingAgent, DataFetcher
from neuratrade.agent import TradingEnvironment
from neuratrade.config import Config
import matplotlib.pyplot as plt
import pandas as pd


def backtest_agent(ticker='AAPL', start_date='2023-01-01', end_date='2023-12-31', 
                   model_path=None):
    """
    Backtest a trained trading agent
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for backtesting
        end_date (str): End date for backtesting
        model_path (str): Path to saved model weights
    """
    print(f"\n{'='*60}")
    print(f"Backtesting Trading Agent for {ticker}")
    print(f"{'='*60}")
    
    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(ticker, start_date, end_date)
    
    # Create environment
    env = TradingEnvironment(data)
    
    # Create agent
    agent = TradingAgent()
    
    # Load model if provided
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("\nWarning: No model loaded. Using untrained agent.")
    
    # Run backtest
    state = env.reset()
    done = False
    portfolio_values = []
    buy_and_hold_values = []
    
    initial_price = data['Close'].iloc[Config.WINDOW_SIZE]
    initial_shares = env.initial_balance / initial_price
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        if next_state is not None:
            state = next_state
            portfolio_values.append(info['net_worth'])
            
            # Calculate buy and hold strategy value
            current_price = data['Close'].iloc[env.current_step]
            buy_and_hold_value = initial_shares * current_price
            buy_and_hold_values.append(buy_and_hold_value)
        else:
            done = True
    
    # Calculate metrics
    final_net_worth = env.get_portfolio_value()
    profit = final_net_worth - env.initial_balance
    profit_pct = (profit / env.initial_balance) * 100
    
    buy_and_hold_final = buy_and_hold_values[-1] if buy_and_hold_values else env.initial_balance
    buy_and_hold_profit = buy_and_hold_final - env.initial_balance
    buy_and_hold_pct = (buy_and_hold_profit / env.initial_balance) * 100
    
    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}")
    print(f"\nAgent Strategy:")
    print(f"  Initial Balance: ${env.initial_balance:,.2f}")
    print(f"  Final Net Worth: ${final_net_worth:,.2f}")
    print(f"  Profit: ${profit:,.2f} ({profit_pct:.2f}%)")
    print(f"  Total Trades: {len(env.trades)}")
    
    print(f"\nBuy & Hold Strategy:")
    print(f"  Initial Balance: ${env.initial_balance:,.2f}")
    print(f"  Final Value: ${buy_and_hold_final:,.2f}")
    print(f"  Profit: ${buy_and_hold_profit:,.2f} ({buy_and_hold_pct:.2f}%)")
    
    print(f"\nOutperformance: {profit_pct - buy_and_hold_pct:.2f}%")
    
    # Plot results
    plot_backtest_results(portfolio_values, buy_and_hold_values, env.trades, 
                          data, ticker, start_date, end_date)
    
    return {
        'agent_profit_pct': profit_pct,
        'buy_and_hold_pct': buy_and_hold_pct,
        'trades': env.trades,
        'portfolio_values': portfolio_values
    }


def plot_backtest_results(portfolio_values, buy_and_hold_values, trades, 
                          data, ticker, start_date, end_date):
    """Plot backtest results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot portfolio value comparison
    steps = range(len(portfolio_values))
    ax1.plot(steps, portfolio_values, 'b-', label='Agent Strategy', linewidth=2)
    ax1.plot(steps, buy_and_hold_values, 'g--', label='Buy & Hold', linewidth=2)
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'Portfolio Value Comparison - {ticker} ({start_date} to {end_date})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot stock price with buy/sell signals
    window = Config.WINDOW_SIZE
    price_data = data['Close'].iloc[window:window+len(portfolio_values)]
    ax2.plot(range(len(price_data)), price_data.values, 'k-', 
             label='Stock Price', linewidth=1.5, alpha=0.7)
    
    # Mark buy and sell trades
    for trade in trades:
        step = trade['step'] - window
        if step >= 0 and step < len(price_data):
            if trade['action'] == 'BUY':
                ax2.scatter(step, trade['price'], color='green', marker='^', 
                           s=100, label='Buy' if step == 0 else '')
            elif trade['action'] == 'SELL':
                ax2.scatter(step, trade['price'], color='red', marker='v', 
                           s=100, label='Sell' if step == 0 else '')
    
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Stock Price ($)')
    ax2.set_title(f'Trading Signals - {ticker}')
    
    # Remove duplicate labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    filename = f'results/{ticker}_backtest_{start_date}_{end_date}.png'
    plt.savefig(filename, dpi=150)
    print(f"\nBacktest plot saved to {filename}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest a stock trading agent')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date (default: 2023-01-01)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='End date (default: 2023-12-31)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (default: models/{ticker}_trading_agent.h5)')
    
    args = parser.parse_args()
    
    # Use default model path if not provided
    if args.model is None:
        args.model = f'models/{args.ticker}_trading_agent.h5'
    
    results = backtest_agent(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model
    )
    
    print(f"\n{'='*60}")
    print("Backtesting completed successfully!")
    print(f"{'='*60}")
