"""
QTradeX Adapter for Combinatronix AI Trading System

This adapter integrates the Combinatronix cognitive architecture with QTradeX's BaseBot framework.

The system "makes trades that it likes" by:
1. Presenting market data as sensory input to the cognitive architecture
2. Letting it reason about trading opportunities
3. Getting confidence/preference scores for potential trades
4. Executing trades that the AI "likes" (high confidence above threshold)

Key Components:
- Cognitive architecture for reasoning and decision-making
- Field dynamics for representing market states
- Confidence-based trade selection
- Interpretable decision-making
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any

# Add reference systems to path
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference")
if REFERENCE_PATH not in sys.path:
    sys.path.insert(0, REFERENCE_PATH)

# Add Combinatronix to path
COMBINATRONIX_PATH = os.path.join(REFERENCE_PATH, "Combinatronix_AI_Cursor")
if COMBINATRONIX_PATH not in sys.path:
    sys.path.insert(0, COMBINATRONIX_PATH)

try:
    # Import Combinatronix components
    from combinatronix.organisms.integration import CognitiveArchitecture
    from combinatronix.core import NDAnalogField
    from combinatronix.organisms.reasoning import ReasoningEngine
    from combinatronix.organisms.memory import EpisodicMemory
except ImportError as e:
    print(f"Warning: Could not import Combinatronix components: {e}")
    print("Make sure Combinatronix AI is available in the reference folder.")
    raise

import qtradex as qx


class CombinatronixQTradeXBot(qx.BaseBot):
    """
    QTradeX BaseBot adapter for Combinatronix AI Trading System
    
    The AI "makes trades that it likes" by evaluating trading opportunities
    through its cognitive architecture and executing those with high confidence.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, use_memory: bool = True):
        """
        Initialize the Combinatronix trading system within QTradeX
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence score (0-1) required to execute a trade
            Higher = more selective, only trades it really "likes"
        use_memory : bool
            Whether to use episodic memory to learn from past trades
        """
        # QTradeX tuning parameters
        self.tune = {
            # Confidence threshold
            "confidence_threshold": confidence_threshold,
            
            # Market data encoding
            "price_window_size": 20,  # Bars to encode as field
            "field_size": (8, 8),     # Size of field representation
            
            # Cognitive parameters
            "reasoning_depth": 3,     # How deeply to reason about trades
            "integration_frequency": 5,  # How often to integrate subsystems
            
            # Trading parameters
            "position_size": 0.1,      # 10% of capital per trade
            "max_trades_per_day": 5,   # Limit trading frequency
        }
        
        # Optimization bounds
        self.clamps = {
            "confidence_threshold": [0.4, 0.9, 0.05],
            "price_window_size": [10, 50, 5],
            "reasoning_depth": [1, 5, 1],
        }
        
        # Store configuration
        self.use_memory = use_memory
        self.confidence_threshold = confidence_threshold
        
        # Initialize Combinatronix cognitive architecture
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': confidence_threshold,
            'max_goals': 3,
            'attention_capacity': 5,
            'max_thought_depth': self.tune['reasoning_depth'],
            'integration_frequency': self.tune['integration_frequency'],
        })
        
        # Set up trading goal
        self.mind.set_goal("make_profitable_trades")
        
        # Internal state
        self.trade_history = []
        self.confidence_scores = []
        self.market_memory = []  # Store market patterns for learning
        
        # Cached indicators
        self._market_field_cache = None
        self._confidence_cache = None
        self._preference_cache = None
    
    def indicators(self, data):
        """
        Compute market indicators using Combinatronix field representation
        
        This is called once with the full dataset at backtest start.
        """
        n = len(data['close'])
        
        # Initialize arrays for indicators
        market_field_energy = np.zeros(n)  # Energy in market field representation
        confidence_signal = np.zeros(n)    # AI confidence in trading opportunity
        preference_signal = np.zeros(n)    # AI preference/liking for trade
        
        # Process data to compute indicators
        window_size = self.tune['price_window_size']
        
        for i in range(window_size, n):
            # Get price window
            price_window = data['close'][i-window_size:i]
            volume_window = data['volume'][i-window_size:i] if len(data['volume']) > i else np.ones(window_size)
            
            # Encode market data as field
            market_field = self._encode_market_to_field(price_window, volume_window)
            
            # Store field energy
            market_field_energy[i] = np.sum(market_field.activation)
            
            # Present to cognitive architecture as sensory input
            sensory_input = self._create_sensory_input(
                price_window, volume_window, market_field, i
            )
            
            # Let the AI perceive and reason about the market
            processed = self.mind.perceive(sensory_input)
            
            # Let it think about the trading opportunity
            self.mind.think(thought_type="deliberate")
            
            # Get AI's evaluation of the trading opportunity
            # We'll create a "trade opportunity" field and see if the AI likes it
            trade_opportunity = self._create_trade_opportunity_field(
                price_window, volume_window, market_field
            )
            
            # Evaluate the trade opportunity
            confidence = self._evaluate_trade_opportunity(trade_opportunity)
            preference = self._get_trade_preference(trade_opportunity)
            
            confidence_signal[i] = confidence
            preference_signal[i] = preference
            
            # Store in memory for learning
            if self.use_memory:
                self.market_memory.append({
                    'index': i,
                    'field': market_field,
                    'confidence': confidence,
                    'preference': preference,
                    'price': data['close'][i],
                })
        
        # Cache for potential reuse
        self._market_field_cache = market_field_energy
        self._confidence_cache = confidence_signal
        self._preference_cache = preference_signal
        
        return {
            "market_field_energy": market_field_energy,
            "confidence": confidence_signal,
            "preference": preference_signal,
        }
    
    def _encode_market_to_field(self, prices: np.ndarray, volumes: np.ndarray) -> NDAnalogField:
        """
        Encode market data (prices, volumes) into a field representation
        
        This converts time series data into a spatial field that the
        cognitive architecture can reason about.
        """
        field_size = self.tune['field_size']
        field = NDAnalogField(field_size)
        
        # Normalize prices
        if len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            price_returns = np.clip(price_returns, -0.1, 0.1)  # Clip extreme moves
            price_returns = (price_returns + 0.1) / 0.2  # Normalize to 0-1
        else:
            price_returns = np.array([0.5])
        
        # Normalize volumes
        if len(volumes) > 0 and np.max(volumes) > 0:
            volumes_norm = volumes / np.max(volumes)
        else:
            volumes_norm = np.ones(len(volumes)) * 0.5
        
        # Map to field (flatten and reshape)
        data_flat = np.concatenate([price_returns, volumes_norm])
        
        # Pad or truncate to field size
        field_size_total = field_size[0] * field_size[1]
        if len(data_flat) < field_size_total:
            # Pad with mean
            padding = np.full(field_size_total - len(data_flat), np.mean(data_flat))
            data_flat = np.concatenate([data_flat, padding])
        else:
            # Truncate
            data_flat = data_flat[:field_size_total]
        
        # Reshape to field
        field.activation = data_flat.reshape(field_size)
        
        return field
    
    def _create_sensory_input(self, prices: np.ndarray, volumes: np.ndarray,
                             market_field: NDAnalogField, index: int) -> Dict[str, Any]:
        """Create sensory input for the cognitive architecture"""
        # Convert field to visual-like representation
        field_visual = market_field.activation
        
        # Normalize to 0-1 range for visual processing
        if np.max(field_visual) > 0:
            field_visual = field_visual / np.max(field_visual)
        
        return {
            'visual': field_visual,  # Market state as "visual" input
            'timestamp': index,
            'market_data': {
                'prices': prices,
                'volumes': volumes,
                'current_price': prices[-1] if len(prices) > 0 else 0,
            }
        }
    
    def _create_trade_opportunity_field(self, prices: np.ndarray, volumes: np.ndarray,
                                        market_field: NDAnalogField) -> NDAnalogField:
        """
        Create a field representing a potential trade opportunity
        
        This field encodes the "attractiveness" of making a trade right now.
        """
        opportunity_field = market_field.copy()
        
        # Enhance field based on trading signals
        # Price momentum
        if len(prices) >= 2:
            momentum = (prices[-1] - prices[-2]) / prices[-2]
            momentum_signal = np.tanh(momentum * 10)  # Normalize to -1 to 1
            momentum_signal = (momentum_signal + 1) / 2  # Normalize to 0-1
            
            # Add momentum signal to field
            opportunity_field.activation += momentum_signal * 0.2
        
        # Volume confirmation
        if len(volumes) >= 2:
            volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-10)
            volume_signal = np.tanh(volume_change * 5)
            volume_signal = (volume_signal + 1) / 2
            
            # Add volume signal to field
            opportunity_field.activation += volume_signal * 0.15
        
        # Clip to valid range
        opportunity_field.activation = np.clip(opportunity_field.activation, 0, 1)
        
        return opportunity_field
    
    def _evaluate_trade_opportunity(self, opportunity_field: NDAnalogField) -> float:
        """
        Let the AI evaluate a trade opportunity and return confidence score
        
        This is where the AI "decides if it likes" the trade.
        """
        # Simulate the trade as an action
        simulated_outcome = self.mind._simulate_action(opportunity_field)
        
        # Evaluate outcome against trading goal
        confidence = self.mind._evaluate_outcome(simulated_outcome, "make_profitable_trades")
        
        # Also check if the opportunity field resonates with the AI's current state
        # Use comparator to see how well it matches global workspace
        if hasattr(self.mind, 'integration_components'):
            global_workspace = self.mind.integration_components.get('global_workspace')
            if global_workspace:
                # Use comparator atom if available
                if 'comparator' in self.mind.integration_components:
                    similarity = self.mind.integration_components['comparator'].apply(
                        opportunity_field, global_workspace
                    )
                    similarity_score = np.mean(similarity.activation)
                    # Combine confidence with similarity
                    confidence = (confidence + similarity_score) / 2
        
        return float(np.clip(confidence, 0, 1))
    
    def _get_trade_preference(self, opportunity_field: NDAnalogField) -> float:
        """
        Get the AI's preference/liking for a trade
        
        This represents how much the AI "likes" this trade opportunity.
        """
        # Preference is based on field energy and structure
        field_energy = np.sum(opportunity_field.activation)
        field_structure = np.std(opportunity_field.activation)
        
        # Normalize
        energy_norm = field_energy / (opportunity_field.activation.size * 1.0)
        structure_norm = field_structure / 0.5  # Rough normalization
        
        # Preference combines energy and structure
        # High energy + moderate structure = high preference
        preference = (energy_norm * 0.7 + structure_norm * 0.3)
        
        return float(np.clip(preference, 0, 1))
    
    def strategy(self, state, indicators):
        """
        Use Combinatronix AI to decide on trades
        
        The AI "makes trades that it likes" based on confidence and preference scores.
        """
        price = state["close"]
        wallet = state.get("wallet")
        
        # Get indicators (scalar values for current tick)
        confidence = indicators.get("confidence", 0.0)
        preference = indicators.get("preference", 0.0)
        market_energy = indicators.get("market_field_energy", 0.0)
        
        # Get current position from wallet
        if wallet:
            asset = "BTC"  # Will be determined from data
            currency = "USDT"
            current_position = wallet.get(asset, 0.0)
        else:
            current_position = 0.0
        
        # Combined "liking" score
        # The AI "likes" a trade if both confidence and preference are high
        liking_score = (confidence * 0.6 + preference * 0.4)
        
        # Threshold for "liking" a trade
        like_threshold = self.tune.get("confidence_threshold", 0.6)
        
        # Buy decision: AI likes buying opportunity
        if liking_score > like_threshold and confidence > 0.5:
            # Check if we should buy
            if current_position <= 0:  # Not already long
                # Additional check: preference should be positive
                if preference > 0.5:
                    return qx.Buy()
        
        # Sell decision: AI doesn't like current position
        if liking_score < (like_threshold * 0.5) or confidence < 0.3:
            if current_position > 0:  # Have position
                return qx.Sell()
        
        # Hold: AI is neutral or needs more information
        # Use thresholds based on preference
        if preference > 0.4:
            buy_threshold = price * (1 - confidence * 0.01)
            sell_threshold = price * (1 + confidence * 0.01)
        else:
            buy_threshold = price * (1 - 0.02)
            sell_threshold = price * (1 + 0.02)
        
        return qx.Thresholds(
            buying=buy_threshold,
            selling=sell_threshold,
            maxvolume=self.tune.get("position_size", 0.1)
        )
    
    def execution(self, signal, indicators, wallet):
        """
        Modify signals based on AI confidence and preference
        """
        confidence = indicators.get("confidence", 0.0)
        preference = indicators.get("preference", 0.0)
        
        # Adjust position size based on how much the AI "likes" the trade
        liking_score = (confidence * 0.6 + preference * 0.4)
        
        if isinstance(signal, (qx.Buy, qx.Sell)):
            # Scale position size by liking score
            # More liking = larger position
            signal.maxvolume = signal.maxvolume * liking_score
        
        return signal
    
    def autorange(self):
        """Calculate warmup period for cognitive architecture"""
        return max(
            self.tune.get("price_window_size", 20),
            10  # Minimum for field representation
        )
    
    def reset(self):
        """Reset internal state"""
        # Reinitialize cognitive architecture
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': self.confidence_threshold,
            'max_goals': 3,
            'attention_capacity': 5,
            'max_thought_depth': self.tune['reasoning_depth'],
            'integration_frequency': self.tune['integration_frequency'],
        })
        
        # Reset goal
        self.mind.set_goal("make_profitable_trades")
        
        # Clear state
        self.trade_history = []
        self.confidence_scores = []
        self.market_memory = []
        
        # Clear caches
        self._market_field_cache = None
        self._confidence_cache = None
        self._preference_cache = None
    
    def fitness(self, states, raw_states, asset, currency):
        """Define performance metrics"""
        return [
            "roi",
            "cagr",
            "sortino",
            "maximum_drawdown",
            "trade_win_rate"
        ], {}
    
    def plot(self, data, states, indicators, block):
        """Visualize AI indicators"""
        qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            (
                ("confidence", "AI Confidence", "cyan", 1, "AI Evaluation"),
                ("preference", "AI Preference", "yellow", 1, "AI Liking"),
                ("market_field_energy", "Market Energy", "magenta", 2, "Field Representation"),
            )
        )


# Example usage
if __name__ == "__main__":
    # Load data
    data = qx.Data(
        exchange="kucoin",
        asset="BTC",
        currency="USDT",
        begin="2020-01-01",
        end="2023-01-01"
    )
    
    # Create bot - AI will make trades it "likes"
    bot = CombinatronixQTradeXBot(
        confidence_threshold=0.6,  # Only trade if AI really likes it
        use_memory=True            # Learn from past trades
    )
    
    # Run backtest
    qx.dispatch(bot, data)

