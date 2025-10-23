"""
Deep Q-Network (DQN) Trading Agent
"""

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from neuratrade.config.config import Config
from neuratrade.agent.trading_env import TradingEnvironment


class TradingAgent:
    """DQN-based trading agent for stock trading"""
    
    def __init__(self, state_size=None, action_size=None):
        """
        Initialize trading agent
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
        """
        self.state_size = state_size or Config.STATE_SIZE
        self.action_size = action_size or Config.ACTION_SIZE
        
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.learning_rate = Config.LEARNING_RATE
        self.batch_size = Config.BATCH_SIZE
        
        # Create neural networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build neural network model
        
        Returns:
            keras.Model: Compiled neural network
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Update target network with weights from main network"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action based on epsilon-greedy policy
        
        Args:
            state: Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """
        Train the model using experience replay
        
        Returns:
            float: Average loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for starting states
        target = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        target_next = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train the model
        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def train(self, env, episodes=None):
        """
        Train the agent
        
        Args:
            env (TradingEnvironment): Trading environment
            episodes (int): Number of training episodes
            
        Returns:
            list: Training history with rewards per episode
        """
        episodes = episodes or Config.EPISODES
        history = []
        
        print(f"\nStarting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                action = self.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                if next_state is not None:
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                else:
                    done = True
                
                if len(self.memory) > self.batch_size:
                    self.replay()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.update_target_model()
            
            final_net_worth = env.get_portfolio_value()
            profit = final_net_worth - env.initial_balance
            profit_pct = (profit / env.initial_balance) * 100
            
            history.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'net_worth': final_net_worth,
                'profit': profit,
                'profit_pct': profit_pct,
                'epsilon': self.epsilon,
                'steps': steps
            })
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Net Worth: ${final_net_worth:.2f} | "
                      f"Profit: {profit_pct:.2f}% | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        print("\nTraining completed!")
        return history
    
    def test(self, env):
        """
        Test the agent
        
        Args:
            env (TradingEnvironment): Trading environment
            
        Returns:
            dict: Test results
        """
        print("\nTesting agent...")
        
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = self.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            if next_state is not None:
                state = next_state
                total_reward += reward
            else:
                done = True
        
        final_net_worth = env.get_portfolio_value()
        profit = final_net_worth - env.initial_balance
        profit_pct = (profit / env.initial_balance) * 100
        
        results = {
            'initial_balance': env.initial_balance,
            'final_net_worth': final_net_worth,
            'profit': profit,
            'profit_pct': profit_pct,
            'total_reward': total_reward,
            'total_trades': len(env.trades)
        }
        
        print(f"\nTest Results:")
        print(f"Initial Balance: ${results['initial_balance']:.2f}")
        print(f"Final Net Worth: ${results['final_net_worth']:.2f}")
        print(f"Profit: ${results['profit']:.2f} ({results['profit_pct']:.2f}%)")
        print(f"Total Trades: {results['total_trades']}")
        
        return results
    
    def save(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.update_target_model()
        print(f"Model loaded from {filepath}")
