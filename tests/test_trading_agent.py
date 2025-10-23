"""
Unit tests for TradingAgent
"""

import unittest
import numpy as np
from neuratrade.agent.trading_agent import TradingAgent
from neuratrade.config import Config


class TestTradingAgent(unittest.TestCase):
    """Test cases for TradingAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = TradingAgent(state_size=10, action_size=3)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.state_size, 10)
        self.assertEqual(self.agent.action_size, 3)
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)
    
    def test_model_build(self):
        """Test neural network model building"""
        model = self.agent._build_model()
        self.assertIsNotNone(model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 3))
    
    def test_act_method(self):
        """Test action selection"""
        state = np.random.rand(10)
        action = self.agent.act(state, training=False)
        
        # Action should be 0, 1, or 2
        self.assertIn(action, [0, 1, 2])
    
    def test_remember_method(self):
        """Test experience storage"""
        state = np.random.rand(10)
        next_state = np.random.rand(10)
        action = 1
        reward = 0.5
        done = False
        
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
    
    def test_update_target_model(self):
        """Test target model update"""
        # Update target model
        self.agent.update_target_model()
        
        # Check that weights are copied
        main_weights = self.agent.model.get_weights()
        target_weights = self.agent.target_model.get_weights()
        
        for main_w, target_w in zip(main_weights, target_weights):
            np.testing.assert_array_equal(main_w, target_w)


if __name__ == '__main__':
    unittest.main()
