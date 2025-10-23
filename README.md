# NeuraTrade

A Deep Reinforcement Learning Trading Agent for US Stocks

## 🚀 Overview

NeuraTrade is an intelligent trading agent that uses Deep Q-Network (DQN) reinforcement learning to trade US stocks. The agent learns optimal trading strategies by analyzing historical stock data, technical indicators, and market patterns.

## ✨ Features

- **Deep Q-Network (DQN)** reinforcement learning agent
- **Real-time data fetching** from Yahoo Finance
- **Technical indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV)
- **Configurable trading environment** with customizable parameters
- **Training and backtesting** capabilities
- **Performance visualization** with matplotlib
- **Commission fees** and realistic trading simulation
- **Buy and Hold comparison** for performance benchmarking

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ys2899/NeuraTrade.git
cd NeuraTrade
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Simple Example

Run a simple example to see the agent in action:

```bash
python examples/simple_example.py
```

### Train an Agent

Train a trading agent on historical stock data:

```bash
# Train on Apple stock (AAPL) with default parameters
python examples/train_agent.py --ticker AAPL --episodes 50

# Train on Google stock with custom date range
python examples/train_agent.py --ticker GOOGL --start 2020-01-01 --end 2023-12-31 --episodes 100
```

### Backtest an Agent

Backtest a trained agent on new data:

```bash
# Backtest with saved model
python examples/backtest_agent.py --ticker AAPL --start 2023-01-01 --end 2023-12-31

# Backtest with custom model path
python examples/backtest_agent.py --ticker AAPL --model models/AAPL_trading_agent.h5
```

## 🧠 How It Works

### Architecture

The trading agent uses a Deep Q-Network (DQN) with the following architecture:

1. **Input Layer**: State representation (10 features)
   - Normalized stock price
   - Balance and shares held
   - Technical indicators (SMA ratio, RSI, MACD)
   - Recent price changes

2. **Hidden Layers**:
   - Dense layer (64 neurons, ReLU activation)
   - Dropout (20%)
   - Dense layer (32 neurons, ReLU activation)
   - Dropout (20%)
   - Dense layer (16 neurons, ReLU activation)

3. **Output Layer**: Q-values for 3 actions (Hold, Buy, Sell)

### Training Process

1. Agent observes the current market state
2. Selects action using ε-greedy policy (exploration vs exploitation)
3. Executes action in the trading environment
4. Receives reward based on portfolio value change
5. Stores experience in replay memory
6. Samples mini-batches and updates Q-network
7. Periodically updates target network

### State Representation

The agent observes a 10-dimensional state vector:
- Normalized current price
- Normalized balance
- Normalized shares held
- SMA ratio (price/SMA)
- RSI (0-1 normalized)
- MACD (normalized)
- MACD signal (normalized)
- Recent price changes (3 timesteps)

### Actions

- **0: Hold** - Do nothing
- **1: Buy** - Purchase up to MAX_SHARES_PER_TRADE
- **2: Sell** - Sell up to MAX_SHARES_PER_TRADE

### Reward Function

Reward = (New Portfolio Value - Previous Portfolio Value) / Previous Portfolio Value

## 🔧 Configuration

Modify trading parameters in `neuratrade/config/config.py`:

```python
# Trading parameters
INITIAL_BALANCE = 100000  # Starting balance in USD
COMMISSION_FEE = 0.001  # 0.1% commission per trade
MAX_SHARES_PER_TRADE = 100  # Maximum shares per trade

# Agent parameters
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration decay rate
LEARNING_RATE = 0.001  # Neural network learning rate

# Training parameters
EPISODES = 100  # Number of training episodes
WINDOW_SIZE = 10  # Lookback window for states
```

## 📊 Technical Indicators

NeuraTrade includes a comprehensive set of technical indicators:

- **SMA** (Simple Moving Average)
- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**
- **ATR** (Average True Range)
- **OBV** (On-Balance Volume)

## 📈 Performance Metrics

The agent tracks and reports:

- Total reward per episode
- Net worth over time
- Profit/loss (absolute and percentage)
- Number of trades executed
- Comparison with Buy & Hold strategy

## 🛠️ Advanced Usage

### Custom Training Loop

```python
from neuratrade import TradingAgent, DataFetcher
from neuratrade.agent import TradingEnvironment
from neuratrade.config import Config

# Fetch data
data_fetcher = DataFetcher()
data = data_fetcher.fetch_data('AAPL', '2020-01-01', '2023-12-31')

# Create environment
env = TradingEnvironment(data)

# Create agent with custom parameters
agent = TradingAgent(state_size=10, action_size=3)

# Train
history = agent.train(env, episodes=100)

# Save model
agent.save('my_model.h5')

# Test
test_results = agent.test(env)
```

### Using Technical Indicators

```python
from neuratrade.utils import TechnicalIndicators
import pandas as pd

# Calculate indicators
sma = TechnicalIndicators.calculate_sma(data['Close'], window=20)
rsi = TechnicalIndicators.calculate_rsi(data['Close'], window=14)
macd, signal, hist = TechnicalIndicators.calculate_macd(data['Close'])
upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(data['Close'])
```

## 📝 Project Structure

```
NeuraTrade/
├── neuratrade/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── trading_agent.py    # DQN agent implementation
│   │   └── trading_env.py      # Trading environment
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_fetcher.py     # Data fetching utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   └── indicators.py       # Technical indicators
│   └── config/
│       ├── __init__.py
│       └── config.py           # Configuration settings
├── examples/
│   ├── simple_example.py       # Quick start example
│   ├── train_agent.py          # Training script
│   └── backtest_agent.py       # Backtesting script
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## 🧪 Testing

Run tests (when implemented):

```bash
python -m pytest tests/
```

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading stocks involves substantial risk of loss
- Past performance does not guarantee future results
- This agent should NOT be used for real trading without extensive testing and risk management
- The authors are not responsible for any financial losses incurred using this software
- Always consult with a financial advisor before making investment decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing stock market data
- TensorFlow/Keras for deep learning framework
- The reinforcement learning community for research and inspiration

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Trading! 📈**