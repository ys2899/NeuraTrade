"""
Configuration settings for the trading agent
"""


class Config:
    """Configuration class for trading agent parameters"""
    
    # Trading parameters
    INITIAL_BALANCE = 100000  # Starting balance in USD
    COMMISSION_FEE = 0.001  # 0.1% commission per trade
    MAX_SHARES_PER_TRADE = 100  # Maximum shares to buy/sell in one action
    
    # Agent parameters
    STATE_SIZE = 10  # Number of features in state
    ACTION_SIZE = 3  # Hold, Buy, Sell
    MEMORY_SIZE = 2000  # Replay memory size
    BATCH_SIZE = 32  # Training batch size
    GAMMA = 0.95  # Discount factor
    EPSILON = 1.0  # Initial exploration rate
    EPSILON_MIN = 0.01  # Minimum exploration rate
    EPSILON_DECAY = 0.995  # Exploration decay rate
    LEARNING_RATE = 0.001  # Neural network learning rate
    
    # Training parameters
    EPISODES = 100  # Number of training episodes
    WINDOW_SIZE = 10  # Lookback window for states
    
    # Data parameters
    DATA_START_DATE = "2020-01-01"
    DATA_END_DATE = "2023-12-31"
    TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
