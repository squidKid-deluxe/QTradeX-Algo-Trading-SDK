"""
QTradeX Strategy Router Using Combinatronix AI

This adapter uses Combinatronix AI to route between different strategy modules
based on its perception of the market. The AI evaluates market conditions and
selects which strategy module it "likes" best for the current situation.

Key Features:
- Market perception and analysis
- Strategy module evaluation
- Dynamic routing based on AI preferences
- Multi-strategy coordination
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any, Callable

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


class StrategyModule:
    """
    Wrapper for a strategy module that can be routed to
    
    Each strategy module has:
    - A name/identifier
    - An indicators() method
    - A strategy() method
    - Optional metadata
    """
    
    def __init__(self, name: str, indicators_func: Callable, strategy_func: Callable,
                 metadata: Dict = None):
        """
        Initialize a strategy module
        
        Parameters
        ----------
        name : str
            Name/identifier for this strategy
        indicators_func : callable
            Function that computes indicators: indicators(data) -> dict
        strategy_func : callable
            Function that generates signals: strategy(state, indicators) -> signal
        metadata : dict, optional
            Additional metadata about the strategy (description, tags, etc.)
        """
        self.name = name
        self.indicators_func = indicators_func
        self.strategy_func = strategy_func
        self.metadata = metadata or {}
        
        # Track usage
        self.usage_count = 0
        self.last_used = None
        self.success_rate = 0.5  # Will be updated based on performance
    
    def compute_indicators(self, data: Dict) -> Dict:
        """Compute indicators for this strategy"""
        return self.indicators_func(data)
    
    def generate_signal(self, state: Dict, indicators: Dict):
        """Generate trading signal using this strategy"""
        return self.strategy_func(state, indicators)


class CombinatronixStrategyRouter(qx.BaseBot):
    """
    QTradeX BaseBot that routes between strategy modules using Combinatronix AI
    
    The AI perceives market conditions and routes to the strategy module it
    "likes" best for the current market state.
    """
    
    def __init__(self, strategy_modules: List[StrategyModule],
                 confidence_threshold: float = 0.5, use_memory: bool = True):
        """
        Initialize the strategy router
        
        Parameters
        ----------
        strategy_modules : list of StrategyModule
            List of strategy modules to choose from
        confidence_threshold : float
            Minimum confidence required to route to a strategy
        use_memory : bool
            Whether to use episodic memory to learn which strategies work in which conditions
        """
        if not strategy_modules:
            raise ValueError("Must provide at least one strategy module")
        
        # QTradeX tuning parameters
        self.tune = {
            # Routing parameters
            "confidence_threshold": confidence_threshold,
            "routing_window": 20,  # Bars to analyze for routing decision
            
            # Market perception
            "field_size": (8, 8),
            "perception_depth": 3,
            
            # Strategy evaluation
            "evaluation_window": 10,  # Bars to evaluate strategy performance
            "min_strategy_confidence": 0.4,
            
            # Routing behavior
            "switch_frequency": 5,  # How often to re-evaluate routing (bars)
            "min_switch_confidence_delta": 0.2,  # Min improvement to switch
        }
        
        # Store configuration
        self.strategy_modules = {module.name: module for module in strategy_modules}
        self.use_memory = use_memory
        self.confidence_threshold = confidence_threshold
        
        # Initialize Combinatronix cognitive architecture
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': confidence_threshold,
            'max_goals': len(strategy_modules),
            'attention_capacity': len(strategy_modules),
            'max_thought_depth': self.tune['perception_depth'],
        })
        
        # Set up routing goal
        self.mind.set_goal("select_best_strategy_for_market")
        
        # Routing state
        self.current_strategy = None
        self.strategy_scores = {}  # AI's evaluation scores for each strategy
        self.routing_history = []  # History of routing decisions
        self.market_perceptions = []  # Store market perceptions for learning
        
        # Cached data
        self._all_indicators = {}  # Indicators from all strategies
        self._market_field_cache = None
        self._routing_scores_cache = None
    
    def indicators(self, data):
        """
        Compute indicators for all strategy modules
        
        This allows the AI to evaluate all strategies before routing.
        """
        n = len(data['close'])
        
        # Compute indicators for all strategies
        all_indicators = {}
        for name, module in self.strategy_modules.items():
            try:
                indicators = module.compute_indicators(data)
                all_indicators[name] = indicators
            except Exception as e:
                print(f"Warning: Strategy {name} failed to compute indicators: {e}")
                all_indicators[name] = {}
        
        # Store for use in strategy()
        self._all_indicators = all_indicators
        
        # Compute market perception indicators
        market_field_energy = np.zeros(n)
        routing_scores = np.zeros(n)
        selected_strategy = np.zeros(n)
        
        window_size = self.tune['routing_window']
        
        for i in range(window_size, n):
            # Get market data window
            price_window = data['close'][i-window_size:i]
            volume_window = data['volume'][i-window_size:i] if len(data['volume']) > i else np.ones(window_size)
            
            # Encode market as field
            market_field = self._encode_market_to_field(price_window, volume_window)
            market_field_energy[i] = np.sum(market_field.activation)
            
            # Present market to AI for perception
            sensory_input = self._create_market_perception(
                price_window, volume_window, market_field, i, all_indicators
            )
            
            # AI perceives market conditions
            processed = self.mind.perceive(sensory_input)
            
            # AI thinks about which strategy is best
            self.mind.think(thought_type="deliberate")
            
            # AI evaluates all strategies for current market
            strategy_scores = self._evaluate_strategies_for_market(
                market_field, all_indicators, i
            )
            
            # Select best strategy (one AI "likes" most)
            best_strategy, best_score = self._select_best_strategy(strategy_scores)
            
            # Store routing decision
            self.strategy_scores = strategy_scores
            routing_scores[i] = best_score
            
            # Encode selected strategy as number (for indicator)
            strategy_names = list(self.strategy_modules.keys())
            if best_strategy in strategy_names:
                selected_strategy[i] = strategy_names.index(best_strategy)
            else:
                selected_strategy[i] = 0
            
            # Store in memory for learning
            if self.use_memory:
                self.market_perceptions.append({
                    'index': i,
                    'market_field': market_field,
                    'strategy_scores': strategy_scores,
                    'selected_strategy': best_strategy,
                    'market_conditions': self._extract_market_conditions(price_window, volume_window),
                })
        
        # Cache
        self._market_field_cache = market_field_energy
        self._routing_scores_cache = routing_scores
        
        return {
            "market_field_energy": market_field_energy,
            "routing_score": routing_scores,
            "selected_strategy": selected_strategy,
            # Include indicators from all strategies (for routing decision)
            **{f"{name}_indicators": self._flatten_indicators(indicators) 
               for name, indicators in all_indicators.items()}
        }
    
    def _encode_market_to_field(self, prices: np.ndarray, volumes: np.ndarray) -> NDAnalogField:
        """Encode market data into field representation"""
        field_size = self.tune['field_size']
        field = NDAnalogField(field_size)
        
        # Normalize prices
        if len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            price_returns = np.clip(price_returns, -0.1, 0.1)
            price_returns = (price_returns + 0.1) / 0.2
        else:
            price_returns = np.array([0.5])
        
        # Normalize volumes
        if len(volumes) > 0 and np.max(volumes) > 0:
            volumes_norm = volumes / np.max(volumes)
        else:
            volumes_norm = np.ones(len(volumes)) * 0.5
        
        # Map to field
        data_flat = np.concatenate([price_returns, volumes_norm])
        field_size_total = field_size[0] * field_size[1]
        
        if len(data_flat) < field_size_total:
            padding = np.full(field_size_total - len(data_flat), np.mean(data_flat))
            data_flat = np.concatenate([data_flat, padding])
        else:
            data_flat = data_flat[:field_size_total]
        
        field.activation = data_flat.reshape(field_size)
        return field
    
    def _create_market_perception(self, prices: np.ndarray, volumes: np.ndarray,
                                  market_field: NDAnalogField, index: int,
                                  all_indicators: Dict) -> Dict[str, Any]:
        """Create sensory input representing market perception"""
        # Convert field to visual-like representation
        field_visual = market_field.activation
        if np.max(field_visual) > 0:
            field_visual = field_visual / np.max(field_visual)
        
        # Include strategy indicator summaries
        strategy_summaries = {}
        for name, indicators in all_indicators.items():
            # Summarize indicators as single value
            if indicators:
                values = [v for v in indicators.values() if isinstance(v, (int, float, np.number))]
                if values:
                    strategy_summaries[name] = np.mean(values)
                else:
                    # Try to extract from arrays
                    array_values = [v for v in indicators.values() if isinstance(v, np.ndarray)]
                    if array_values and len(array_values[0]) > index:
                        strategy_summaries[name] = np.mean([arr[index] for arr in array_values if len(arr) > index])
                    else:
                        strategy_summaries[name] = 0.5
            else:
                strategy_summaries[name] = 0.5
        
        return {
            'visual': field_visual,
            'timestamp': index,
            'market_data': {
                'prices': prices,
                'volumes': volumes,
                'current_price': prices[-1] if len(prices) > 0 else 0,
            },
            'strategy_context': strategy_summaries,
        }
    
    def _evaluate_strategies_for_market(self, market_field: NDAnalogField,
                                       all_indicators: Dict, index: int) -> Dict[str, float]:
        """
        AI evaluates all strategies and returns scores for each
        
        Returns dict mapping strategy name to confidence/preference score
        """
        strategy_scores = {}
        
        for name, module in self.strategy_modules.items():
            # Get indicators for this strategy
            indicators = all_indicators.get(name, {})
            
            # Create strategy evaluation field
            strategy_field = self._create_strategy_evaluation_field(
                market_field, indicators, index
            )
            
            # AI evaluates this strategy for current market
            confidence = self._evaluate_strategy_confidence(strategy_field, name)
            preference = self._evaluate_strategy_preference(strategy_field, name)
            
            # Combined score (how much AI "likes" this strategy for current market)
            liking_score = (confidence * 0.6 + preference * 0.4)
            
            # Adjust based on historical performance if using memory
            if self.use_memory and hasattr(module, 'success_rate'):
                liking_score = liking_score * 0.7 + module.success_rate * 0.3
            
            strategy_scores[name] = float(np.clip(liking_score, 0, 1))
        
        return strategy_scores
    
    def _create_strategy_evaluation_field(self, market_field: NDAnalogField,
                                        indicators: Dict, index: int) -> NDAnalogField:
        """Create field representing a strategy's suitability for current market"""
        strategy_field = market_field.copy()
        
        # Enhance field based on strategy indicators
        if indicators:
            # Extract indicator values
            indicator_values = []
            for key, value in indicators.items():
                if isinstance(value, (int, float, np.number)):
                    indicator_values.append(float(value))
                elif isinstance(value, np.ndarray) and len(value) > index:
                    indicator_values.append(float(value[index]))
            
            if indicator_values:
                # Normalize and add to field
                indicator_signal = np.mean(indicator_values)
                indicator_signal = np.clip(indicator_signal, 0, 1)
                
                # Add indicator signal to field
                strategy_field.activation += indicator_signal * 0.3
        
        # Clip to valid range
        strategy_field.activation = np.clip(strategy_field.activation, 0, 1)
        
        return strategy_field
    
    def _evaluate_strategy_confidence(self, strategy_field: NDAnalogField,
                                     strategy_name: str) -> float:
        """AI evaluates confidence that this strategy will work for current market"""
        # Simulate using this strategy
        simulated_outcome = self.mind._simulate_action(strategy_field)
        
        # Evaluate against routing goal
        confidence = self.mind._evaluate_outcome(
            simulated_outcome, "select_best_strategy_for_market"
        )
        
        # Check resonance with global workspace
        if hasattr(self.mind, 'integration_components'):
            global_workspace = self.mind.integration_components.get('global_workspace')
            if global_workspace and 'comparator' in self.mind.integration_components:
                similarity = self.mind.integration_components['comparator'].apply(
                    strategy_field, global_workspace
                )
                similarity_score = np.mean(similarity.activation)
                confidence = (confidence + similarity_score) / 2
        
        return float(np.clip(confidence, 0, 1))
    
    def _evaluate_strategy_preference(self, strategy_field: NDAnalogField,
                                     strategy_name: str) -> float:
        """AI evaluates preference/liking for this strategy"""
        # Preference based on field energy and structure
        field_energy = np.sum(strategy_field.activation)
        field_structure = np.std(strategy_field.activation)
        
        energy_norm = field_energy / (strategy_field.activation.size * 1.0)
        structure_norm = field_structure / 0.5
        
        preference = (energy_norm * 0.7 + structure_norm * 0.3)
        
        return float(np.clip(preference, 0, 1))
    
    def _select_best_strategy(self, strategy_scores: Dict[str, float]) -> Tuple[str, float]:
        """Select the strategy the AI likes most"""
        if not strategy_scores:
            # Fallback to first strategy
            return list(self.strategy_modules.keys())[0], 0.5
        
        # Find strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        
        # Only select if above threshold
        if best_strategy[1] >= self.tune.get("min_strategy_confidence", 0.4):
            return best_strategy
        else:
            # Fallback to first strategy
            return list(self.strategy_modules.keys())[0], 0.5
    
    def _extract_market_conditions(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Extract market condition features"""
        if len(prices) < 2:
            return {}
        
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'volatility': np.std(returns),
            'trend': np.mean(returns),
            'volume_trend': np.mean(volumes[-5:]) / np.mean(volumes[:-5]) if len(volumes) > 5 else 1.0,
        }
    
    def _flatten_indicators(self, indicators: Dict) -> np.ndarray:
        """Flatten indicators dict to single array for QTradeX"""
        if not indicators:
            return np.array([0.5])
        
        values = []
        for value in indicators.values():
            if isinstance(value, np.ndarray):
                values.extend(value.tolist())
            elif isinstance(value, (int, float)):
                values.append(float(value))
        
        if not values:
            return np.array([0.5])
        
        # Return mean as single value (QTradeX expects arrays of same length)
        return np.full(1000, np.mean(values))  # Placeholder - will be trimmed
    
    def strategy(self, state, indicators):
        """
        Route to selected strategy and generate signal
        
        Uses the strategy that the AI "likes" most for current market conditions.
        """
        # Get selected strategy index
        selected_idx = int(indicators.get("selected_strategy", 0))
        strategy_names = list(self.strategy_modules.keys())
        
        if selected_idx >= len(strategy_names):
            selected_idx = 0
        
        selected_strategy_name = strategy_names[selected_idx]
        selected_module = self.strategy_modules[selected_strategy_name]
        
        # Update current strategy
        self.current_strategy = selected_strategy_name
        
        # Get indicators for selected strategy
        strategy_indicators = self._all_indicators.get(selected_strategy_name, {})
        
        # Extract scalar values for current tick
        current_indicators = {}
        for key, value in strategy_indicators.items():
            if isinstance(value, np.ndarray):
                # Get current value (assuming indicators are aligned with data)
                tick_idx = len(value) - 1 if len(value) > 0 else 0
                current_indicators[key] = float(value[tick_idx]) if tick_idx < len(value) else 0.5
            elif isinstance(value, (int, float)):
                current_indicators[key] = float(value)
            else:
                current_indicators[key] = 0.5
        
        # Generate signal using selected strategy
        try:
            signal = selected_module.generate_signal(state, current_indicators)
            
            # Track usage
            selected_module.usage_count += 1
            selected_module.last_used = state.get("unix", 0)
            
            return signal
        except Exception as e:
            print(f"Warning: Strategy {selected_strategy_name} failed: {e}")
            # Fallback: return None (hold)
            return None
    
    def execution(self, signal, indicators, wallet):
        """Modify signals based on routing confidence"""
        routing_score = indicators.get("routing_score", 0.5)
        
        # Adjust position size based on routing confidence
        if isinstance(signal, (qx.Buy, qx.Sell)):
            signal.maxvolume = signal.maxvolume * routing_score
        
        return signal
    
    def autorange(self):
        """Calculate warmup period"""
        return max(
            self.tune.get("routing_window", 20),
            10
        )
    
    def reset(self):
        """Reset internal state"""
        # Reinitialize cognitive architecture
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': self.confidence_threshold,
            'max_goals': len(self.strategy_modules),
            'attention_capacity': len(self.strategy_modules),
            'max_thought_depth': self.tune['perception_depth'],
        })
        
        self.mind.set_goal("select_best_strategy_for_market")
        
        # Reset routing state
        self.current_strategy = None
        self.strategy_scores = {}
        self.routing_history = []
        self.market_perceptions = []
        
        # Reset strategy modules
        for module in self.strategy_modules.values():
            module.usage_count = 0
            module.last_used = None
        
        # Clear caches
        self._all_indicators = {}
        self._market_field_cache = None
        self._routing_scores_cache = None
    
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
        """Visualize routing indicators"""
        qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            (
                ("routing_score", "Routing Score", "cyan", 1, "Strategy Selection"),
                ("selected_strategy", "Selected Strategy", "yellow", 1, "Active Strategy"),
                ("market_field_energy", "Market Energy", "magenta", 2, "Market Perception"),
            )
        )
    
    def get_routing_stats(self) -> Dict:
        """Get statistics about routing decisions"""
        return {
            'current_strategy': self.current_strategy,
            'strategy_scores': self.strategy_scores,
            'strategy_usage': {
                name: {
                    'usage_count': module.usage_count,
                    'last_used': module.last_used,
                    'success_rate': module.success_rate,
                }
                for name, module in self.strategy_modules.items()
            },
            'routing_decisions': len(self.routing_history),
        }


# Example: Creating strategy modules and router
if __name__ == "__main__":
    # Example strategy modules
    def trend_indicators(data):
        """Trend following indicators"""
        close = data['close']
        return {
            'ema_fast': qx.ti.ema(close, 10),
            'ema_slow': qx.ti.ema(close, 30),
        }
    
    def trend_strategy(state, indicators):
        """Trend following strategy"""
        ema_fast = indicators.get('ema_fast', state['close'])
        ema_slow = indicators.get('ema_slow', state['close'])
        
        if ema_fast > ema_slow:
            return qx.Buy()
        elif ema_fast < ema_slow:
            return qx.Sell()
        return None
    
    def mean_reversion_indicators(data):
        """Mean reversion indicators"""
        close = data['close']
        return {
            'rsi': qx.ti.rsi(close, 14),
            'bb_upper': qx.ti.bbands(close, 20)[0],
            'bb_lower': qx.ti.bbands(close, 20)[2],
        }
    
    def mean_reversion_strategy(state, indicators):
        """Mean reversion strategy"""
        rsi = indicators.get('rsi', 50)
        price = state['close']
        bb_upper = indicators.get('bb_upper', price * 1.1)
        bb_lower = indicators.get('bb_lower', price * 0.9)
        
        if rsi < 30 or price < bb_lower:
            return qx.Buy()
        elif rsi > 70 or price > bb_upper:
            return qx.Sell()
        return None
    
    # Create strategy modules
    strategies = [
        StrategyModule(
            name="trend_following",
            indicators_func=trend_indicators,
            strategy_func=trend_strategy,
            metadata={'type': 'trend', 'description': 'EMA crossover trend following'}
        ),
        StrategyModule(
            name="mean_reversion",
            indicators_func=mean_reversion_indicators,
            strategy_func=mean_reversion_strategy,
            metadata={'type': 'mean_reversion', 'description': 'RSI and Bollinger Bands mean reversion'}
        ),
    ]
    
    # Create router
    router = CombinatronixStrategyRouter(
        strategy_modules=strategies,
        confidence_threshold=0.5,
        use_memory=True
    )
    
    # Load data
    data = qx.Data(
        exchange="kucoin",
        asset="BTC",
        currency="USDT",
        begin="2020-01-01",
        end="2023-01-01"
    )
    
    # Run backtest
    qx.dispatch(router, data)
    
    # Print routing stats
    print("\nRouting Statistics:")
    stats = router.get_routing_stats()
    print(f"Current strategy: {stats['current_strategy']}")
    print(f"Strategy usage: {stats['strategy_usage']}")

