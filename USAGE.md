# Usage Guide

This guide provides detailed instructions on how to use NeuraTrade for trading US stocks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training an Agent](#training-an-agent)
4. [Backtesting](#backtesting)
5. [Advanced Usage](#advanced-usage)
6. [Configuration](#configuration)
7. [Tips and Best Practices](#tips-and-best-practices)

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy: Numerical computing
- pandas: Data manipulation
- yfinance: Stock data fetching
- tensorflow: Deep learning framework
- scikit-learn: Machine learning utilities
- matplotlib: Plotting and visualization
- ta: Technical analysis library
- gym: Reinforcement learning toolkit

## Quick Start

The fastest way to get started is to run the simple example:

```bash
python examples/simple_example.py
```

This will:
1. Fetch Apple (AAPL) stock data from 2022-2023
2. Create a trading environment
3. Train an agent for 10 episodes
4. Test the agent and display results

## Training an Agent

### Basic Training

Train on Apple stock with default settings:

```bash
python examples/train_agent.py
```

### Custom Training Options

```bash
# Train on different stock
python examples/train_agent.py --ticker GOOGL

# Custom date range
python examples/train_agent.py --ticker MSFT --start 2019-01-01 --end 2023-12-31

# More episodes for better learning
python examples/train_agent.py --ticker TSLA --episodes 100

# Combine options
python examples/train_agent.py --ticker NVDA --start 2020-01-01 --end 2023-12-31 --episodes 150
```

### Training Output

During training, you'll see:
- Episode number and progress
- Total reward per episode
- Final net worth
- Profit percentage
- Exploration rate (epsilon)

The trained model will be saved to `models/{TICKER}_trading_agent.h5`

### Training Visualization

After training, a plot will be saved showing:
- Reward progression over episodes
- Profit percentage over episodes

Location: `results/{TICKER}_training_results.png`

## Backtesting

### Basic Backtesting

Test a trained agent on new data:

```bash
python examples/backtest_agent.py --ticker AAPL
```

### Custom Backtesting

```bash
# Custom date range for backtesting
python examples/backtest_agent.py --ticker AAPL --start 2023-01-01 --end 2023-12-31

# Use specific model file
python examples/backtest_agent.py --ticker GOOGL --model models/GOOGL_trading_agent.h5

# Backtest on different period
python examples/backtest_agent.py --ticker MSFT --start 2023-06-01 --end 2023-12-31
```

### Backtesting Results

The backtesting script provides:
- Agent strategy performance
- Buy & Hold strategy comparison
- Total trades executed
- Portfolio value over time
- Visual representation of buy/sell signals

Location: `results/{TICKER}_backtest_{START}_{END}.png`

## Advanced Usage

### Custom Python Script

Create your own trading script:

```python
from neuratrade import TradingAgent, DataFetcher
from neuratrade.agent import TradingEnvironment
from neuratrade.config import Config

# Customize configuration
Config.update(
    INITIAL_BALANCE=50000,
    MAX_SHARES_PER_TRADE=50,
    EPISODES=100,
    LEARNING_RATE=0.0005
)

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_data('AAPL', '2020-01-01', '2023-12-31')

# Split into train/test
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Create environments
train_env = TradingEnvironment(train_data)
test_env = TradingEnvironment(test_data)

# Create and train agent
agent = TradingAgent()
history = agent.train(train_env, episodes=100)

# Test agent
results = agent.test(test_env)

# Save model
agent.save('my_custom_agent.h5')
```

### Using Technical Indicators

```python
from neuratrade.utils import TechnicalIndicators

# Calculate indicators
sma_20 = TechnicalIndicators.calculate_sma(data['Close'], window=20)
sma_50 = TechnicalIndicators.calculate_sma(data['Close'], window=50)

rsi = TechnicalIndicators.calculate_rsi(data['Close'], window=14)

macd, signal, histogram = TechnicalIndicators.calculate_macd(data['Close'])

upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
    data['Close'], window=20, num_std=2
)

# Use indicators for analysis
crossover_signal = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
```

### Loading a Trained Model

```python
from neuratrade import TradingAgent

# Create agent
agent = TradingAgent()

# Load weights
agent.load('models/AAPL_trading_agent.h5')

# Use for prediction
state = env.reset()
action = agent.act(state, training=False)
```

## Configuration

### Key Configuration Parameters

Edit `neuratrade/config/config.py` or use `Config.update()`:

#### Trading Parameters

```python
Config.update(
    INITIAL_BALANCE=100000,      # Starting capital in USD
    COMMISSION_FEE=0.001,        # 0.1% commission per trade
    MAX_SHARES_PER_TRADE=100     # Max shares per transaction
)
```

#### Agent Parameters

```python
Config.update(
    STATE_SIZE=10,               # State vector dimensions
    ACTION_SIZE=3,               # Number of actions (Hold, Buy, Sell)
    MEMORY_SIZE=2000,            # Replay memory capacity
    BATCH_SIZE=32,               # Training batch size
    GAMMA=0.95,                  # Discount factor
    EPSILON=1.0,                 # Initial exploration rate
    EPSILON_MIN=0.01,            # Minimum exploration rate
    EPSILON_DECAY=0.995,         # Exploration decay rate
    LEARNING_RATE=0.001          # Neural network learning rate
)
```

#### Training Parameters

```python
Config.update(
    EPISODES=100,                # Training episodes
    WINDOW_SIZE=10,              # Lookback window
    TRAIN_TEST_SPLIT=0.8         # 80% train, 20% test
)
```

## Tips and Best Practices

### Data Selection

1. **Historical Data**: Use at least 2-3 years of data for training
2. **Market Conditions**: Include various market conditions (bull, bear, sideways)
3. **Validation**: Always test on out-of-sample data

### Training

1. **Start Small**: Begin with 50-100 episodes, increase if needed
2. **Monitor Progress**: Watch for consistent improvement in rewards
3. **Multiple Runs**: Train multiple times and select best performing model
4. **Overfitting**: If test performance is much worse than training, reduce model complexity

### Hyperparameter Tuning

1. **Learning Rate**: 
   - Too high (>0.01): Unstable training
   - Too low (<0.0001): Slow convergence
   - Recommended: 0.0005 - 0.001

2. **Epsilon Decay**:
   - Too fast (>0.99): Insufficient exploration
   - Too slow (<0.98): Too much exploration
   - Recommended: 0.995

3. **Batch Size**:
   - Larger (64-128): More stable but slower
   - Smaller (16-32): Faster but more variance
   - Recommended: 32

### Stock Selection

Good candidates for training:
- **High Liquidity**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **Stable Growth**: JNJ, PG, KO
- **Tech Leaders**: NVDA, META, NFLX

Avoid:
- Very low-volume stocks
- Recently IPO'd companies with limited history
- Heavily manipulated penny stocks

### Backtesting

1. **Use Separate Data**: Never backtest on training data
2. **Transaction Costs**: Ensure commission fees are realistic
3. **Multiple Periods**: Test on different time periods
4. **Compare Baseline**: Always compare with Buy & Hold strategy

### Risk Management

1. **Position Sizing**: Consider using MAX_SHARES_PER_TRADE to limit exposure
2. **Diversification**: Train separate agents for different stocks
3. **Stop Loss**: Consider adding stop-loss logic to the environment
4. **Paper Trading**: Test thoroughly before real money

### Common Issues

**Problem**: Agent always holds and never trades
- **Solution**: Increase epsilon, adjust reward function, ensure sufficient training episodes

**Problem**: Agent trades too frequently
- **Solution**: Adjust commission fees, modify reward to penalize excessive trading

**Problem**: Poor generalization to test data
- **Solution**: Use more diverse training data, reduce model complexity, increase dropout

**Problem**: Training is very slow
- **Solution**: Reduce episodes, use GPU acceleration for TensorFlow, reduce batch size

## Performance Metrics

### Understanding Results

- **Net Worth**: Total portfolio value (cash + shares value)
- **Profit %**: Return on initial investment
- **Total Reward**: Cumulative reward (not directly interpretable as money)
- **Total Trades**: Number of buy/sell actions executed

### Good Performance Indicators

- Positive profit percentage
- Outperforms Buy & Hold on test data
- Reasonable number of trades (not too many or too few)
- Consistent performance across different test periods

## Next Steps

1. **Experiment**: Try different stocks, parameters, and date ranges
2. **Enhance**: Add more features to the state representation
3. **Extend**: Implement multi-stock portfolio management
4. **Research**: Study reinforcement learning papers for improvements
5. **Contribute**: Share your improvements with the community

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/ys2899/NeuraTrade/issues
- Documentation: README.md

## Warning

**This is for educational purposes only. Do not use for actual trading without proper testing and risk management.**
