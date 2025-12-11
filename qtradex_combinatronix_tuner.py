"""
QTradeX Dynamic Parameter Tuner Using Combinatronix AI

This adapter uses Combinatronix AI to dynamically tune strategy parameters
based on its perception of the market. The AI continuously adjusts parameters
to optimize strategy performance for current market conditions.

Key Features:
- Market perception and analysis
- Parameter space exploration
- Dynamic parameter adjustment
- Real-time optimization
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass

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


@dataclass
class ParameterRange:
    """Defines a parameter's tuning range"""
    name: str
    min_value: float
    max_value: float
    default_value: float
    step: Optional[float] = None  # Optional step size for discrete parameters
    param_type: str = "continuous"  # "continuous" or "discrete"


class TunableStrategyModule:
    """
    Wrapper for a strategy module with tunable parameters
    
    The strategy function receives parameters as part of the state/indicators.
    """
    
    def __init__(self, name: str, indicators_func: Callable, strategy_func: Callable,
                 parameter_ranges: List[ParameterRange], metadata: Dict = None):
        """
        Initialize a tunable strategy module
        
        Parameters
        ----------
        name : str
            Name/identifier for this strategy
        indicators_func : callable
            Function that computes indicators: indicators(data, params) -> dict
        strategy_func : callable
            Function that generates signals: strategy(state, indicators, params) -> signal
        parameter_ranges : list of ParameterRange
            List of parameters that can be tuned
        metadata : dict, optional
            Additional metadata about the strategy
        """
        self.name = name
        self.indicators_func = indicators_func
        self.strategy_func = strategy_func
        self.parameter_ranges = {pr.name: pr for pr in parameter_ranges}
        self.metadata = metadata or {}
        
        # Current parameter values
        self.current_params = {
            pr.name: pr.default_value for pr in parameter_ranges
        }
        
        # Parameter history for learning
        self.param_history = []
        self.performance_history = []
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameter values"""
        return self.current_params.copy()
    
    def set_params(self, params: Dict[str, float]):
        """Set parameter values (clamped to valid ranges)"""
        for name, value in params.items():
            if name in self.parameter_ranges:
                pr = self.parameter_ranges[name]
                # Clamp to valid range
                clamped_value = np.clip(value, pr.min_value, pr.max_value)
                # Apply step if discrete
                if pr.step is not None:
                    clamped_value = round(clamped_value / pr.step) * pr.step
                self.current_params[name] = clamped_value
    
    def compute_indicators(self, data: Dict, params: Dict = None) -> Dict:
        """Compute indicators with given parameters"""
        if params is None:
            params = self.current_params
        return self.indicators_func(data, params)
    
    def generate_signal(self, state: Dict, indicators: Dict, params: Dict = None):
        """Generate trading signal with given parameters"""
        if params is None:
            params = self.current_params
        return self.strategy_func(state, indicators, params)


class CombinatronixParameterTuner(qx.BaseBot):
    """
    QTradeX BaseBot that dynamically tunes strategy parameters using Combinatronix AI
    
    The AI perceives market conditions and adjusts parameters to optimize
    strategy performance for current market state.
    """
    
    def __init__(self, strategy_module: TunableStrategyModule,
                 tuning_frequency: int = 10, confidence_threshold: float = 0.5,
                 use_memory: bool = True, exploration_rate: float = 0.1):
        """
        Initialize the parameter tuner
        
        Parameters
        ----------
        strategy_module : TunableStrategyModule
            Strategy module with tunable parameters
        tuning_frequency : int
            How often to re-tune parameters (in bars)
        confidence_threshold : float
            Minimum confidence required to apply parameter changes
        use_memory : bool
            Whether to use episodic memory to learn from past parameter settings
        exploration_rate : float
            How much to explore vs exploit (0-1, higher = more exploration)
        """
        # QTradeX tuning parameters
        self.tune = {
            # Tuning parameters
            "tuning_frequency": tuning_frequency,
            "confidence_threshold": confidence_threshold,
            "exploration_rate": exploration_rate,
            
            # Market perception
            "field_size": (8, 8),
            "perception_depth": 3,
            "market_window": 20,
            
            # Parameter tuning
            "tuning_window": 10,  # Bars to evaluate parameter performance
            "min_param_change": 0.01,  # Minimum change to apply
            "max_param_change": 0.1,  # Maximum change per tuning step
        }
        
        # Store configuration
        self.strategy_module = strategy_module
        self.use_memory = use_memory
        self.confidence_threshold = confidence_threshold
        self.exploration_rate = exploration_rate
        
        # Initialize Combinatronix cognitive architecture
        num_params = len(strategy_module.parameter_ranges)
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': confidence_threshold,
            'max_goals': num_params,  # One goal per parameter
            'attention_capacity': num_params,
            'max_thought_depth': self.tune['perception_depth'],
        })
        
        # Set up tuning goal
        self.mind.set_goal("optimize_strategy_parameters")
        
        # Tuning state
        self.param_scores = {}  # AI's evaluation scores for different parameter values
        self.tuning_history = []  # History of parameter adjustments
        self.market_perceptions = []  # Store market perceptions for learning
        self.last_tuning_tick = 0
        
        # Performance tracking
        self.recent_performance = []  # Track recent strategy performance
        
        # Cached data
        self._market_field_cache = None
        self._param_scores_cache = None
    
    def indicators(self, data):
        """
        Compute indicators and evaluate parameter tuning opportunities
        """
        n = len(data['close'])
        
        # Compute indicators with current parameters
        current_indicators = self.strategy_module.compute_indicators(data, self.strategy_module.get_params())
        
        # Compute market perception indicators
        market_field_energy = np.zeros(n)
        param_confidence = np.zeros(n)
        param_adjustment = np.zeros(n)
        
        window_size = self.tune['market_window']
        
        for i in range(window_size, n):
            # Get market data window
            price_window = data['close'][i-window_size:i]
            volume_window = data['volume'][i-window_size:i] if len(data['volume']) > i else np.ones(window_size)
            
            # Encode market as field
            market_field = self._encode_market_to_field(price_window, volume_window)
            market_field_energy[i] = np.sum(market_field.activation)
            
            # Check if it's time to tune parameters
            if i - self.last_tuning_tick >= self.tune['tuning_frequency']:
                # Present market to AI for perception
                sensory_input = self._create_tuning_perception(
                    price_window, volume_window, market_field, i, current_indicators
                )
                
                # AI perceives market conditions
                processed = self.mind.perceive(sensory_input)
                
                # AI thinks about optimal parameters
                self.mind.think(thought_type="deliberate")
                
                # AI evaluates parameter adjustments
                param_adjustments = self._evaluate_parameter_adjustments(
                    market_field, current_indicators, i
                )
                
                # Apply parameter adjustments if AI is confident
                if param_adjustments:
                    self._apply_parameter_adjustments(param_adjustments, i)
                    self.last_tuning_tick = i
                
                # Store confidence
                if param_adjustments:
                    avg_confidence = np.mean([adj['confidence'] for adj in param_adjustments.values()])
                    param_confidence[i] = avg_confidence
                    param_adjustment[i] = 1.0  # Indicate adjustment was made
                else:
                    param_confidence[i] = 0.5
                    param_adjustment[i] = 0.0
            else:
                # Not tuning this tick
                param_confidence[i] = param_confidence[i-1] if i > 0 else 0.5
                param_adjustment[i] = 0.0
        
        # Cache
        self._market_field_cache = market_field_energy
        self._param_scores_cache = param_confidence
        
        # Return indicators (include current parameters as indicators)
        result = {
            "market_field_energy": market_field_energy,
            "param_confidence": param_confidence,
            "param_adjustment": param_adjustment,
            **current_indicators
        }
        
        # Add parameter values as indicators
        for param_name, param_value in self.strategy_module.get_params().items():
            result[f"param_{param_name}"] = np.full(n, param_value)
        
        return result
    
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
    
    def _create_tuning_perception(self, prices: np.ndarray, volumes: np.ndarray,
                                 market_field: NDAnalogField, index: int,
                                 current_indicators: Dict) -> Dict[str, Any]:
        """Create sensory input for parameter tuning"""
        field_visual = market_field.activation
        if np.max(field_visual) > 0:
            field_visual = field_visual / np.max(field_visual)
        
        # Include current parameter values and indicator summaries
        param_context = {}
        for param_name, param_value in self.strategy_module.get_params().items():
            pr = self.strategy_module.parameter_ranges[param_name]
            # Normalize parameter to 0-1 range
            normalized = (param_value - pr.min_value) / (pr.max_value - pr.min_value + 1e-10)
            param_context[param_name] = normalized
        
        # Summarize indicators
        indicator_summary = {}
        for key, value in current_indicators.items():
            if isinstance(value, np.ndarray) and len(value) > index:
                indicator_summary[key] = float(value[index])
            elif isinstance(value, (int, float)):
                indicator_summary[key] = float(value)
        
        return {
            'visual': field_visual,
            'timestamp': index,
            'market_data': {
                'prices': prices,
                'volumes': volumes,
                'current_price': prices[-1] if len(prices) > 0 else 0,
            },
            'parameter_context': param_context,
            'indicator_context': indicator_summary,
        }
    
    def _evaluate_parameter_adjustments(self, market_field: NDAnalogField,
                                       current_indicators: Dict, index: int) -> Dict[str, Dict]:
        """
        AI evaluates parameter adjustments and returns recommended changes
        
        Returns dict mapping parameter name to adjustment info:
        {
            'param_name': {
                'adjustment': float,  # Change to apply
                'confidence': float,  # AI confidence in this adjustment
                'direction': str,     # 'increase' or 'decrease'
            }
        }
        """
        adjustments = {}
        current_params = self.strategy_module.get_params()
        
        for param_name, param_range in self.strategy_module.parameter_ranges.items():
            current_value = current_params[param_name]
            
            # Create field representing parameter adjustment
            param_field = self._create_parameter_adjustment_field(
                market_field, param_name, current_value, param_range
            )
            
            # AI evaluates this parameter adjustment
            confidence = self._evaluate_adjustment_confidence(param_field, param_name)
            preference = self._evaluate_adjustment_preference(param_field, param_name)
            
            # Combined score
            liking_score = (confidence * 0.6 + preference * 0.4)
            
            # Only suggest adjustment if above threshold
            if liking_score > self.confidence_threshold:
                # Determine adjustment direction and magnitude
                # Use exploration vs exploitation
                if np.random.random() < self.exploration_rate:
                    # Exploration: random adjustment
                    direction = np.random.choice(['increase', 'decrease'])
                    magnitude = np.random.uniform(
                        self.tune['min_param_change'],
                        self.tune['max_param_change']
                    )
                else:
                    # Exploitation: AI-guided adjustment
                    # Higher preference = increase, lower = decrease
                    if preference > 0.6:
                        direction = 'increase'
                    elif preference < 0.4:
                        direction = 'decrease'
                    else:
                        # Neutral, skip adjustment
                        continue
                    
                    # Magnitude based on confidence
                    magnitude = self.tune['min_param_change'] + (
                        (liking_score - self.confidence_threshold) * 
                        (self.tune['max_param_change'] - self.tune['min_param_change'])
                    )
                
                # Calculate adjustment
                if direction == 'increase':
                    adjustment = magnitude
                else:
                    adjustment = -magnitude
                
                # Apply step if parameter is discrete
                if param_range.step is not None:
                    adjustment = round(adjustment / param_range.step) * param_range.step
                
                adjustments[param_name] = {
                    'adjustment': adjustment,
                    'confidence': confidence,
                    'preference': preference,
                    'direction': direction,
                }
        
        return adjustments
    
    def _create_parameter_adjustment_field(self, market_field: NDAnalogField,
                                          param_name: str, current_value: float,
                                          param_range: ParameterRange) -> NDAnalogField:
        """Create field representing a parameter adjustment"""
        adjustment_field = market_field.copy()
        
        # Normalize current parameter value
        normalized_value = (current_value - param_range.min_value) / (
            param_range.max_value - param_range.min_value + 1e-10
        )
        
        # Add parameter context to field
        adjustment_field.activation += normalized_value * 0.2
        
        # Clip to valid range
        adjustment_field.activation = np.clip(adjustment_field.activation, 0, 1)
        
        return adjustment_field
    
    def _evaluate_adjustment_confidence(self, adjustment_field: NDAnalogField,
                                       param_name: str) -> float:
        """AI evaluates confidence in parameter adjustment"""
        # Simulate the adjustment
        simulated_outcome = self.mind._simulate_action(adjustment_field)
        
        # Evaluate against tuning goal
        confidence = self.mind._evaluate_outcome(
            simulated_outcome, "optimize_strategy_parameters"
        )
        
        # Check resonance with global workspace
        if hasattr(self.mind, 'integration_components'):
            global_workspace = self.mind.integration_components.get('global_workspace')
            if global_workspace and 'comparator' in self.mind.integration_components:
                similarity = self.mind.integration_components['comparator'].apply(
                    adjustment_field, global_workspace
                )
                similarity_score = np.mean(similarity.activation)
                confidence = (confidence + similarity_score) / 2
        
        return float(np.clip(confidence, 0, 1))
    
    def _evaluate_adjustment_preference(self, adjustment_field: NDAnalogField,
                                       param_name: str) -> float:
        """AI evaluates preference/liking for parameter adjustment"""
        # Preference based on field energy and structure
        field_energy = np.sum(adjustment_field.activation)
        field_structure = np.std(adjustment_field.activation)
        
        energy_norm = field_energy / (adjustment_field.activation.size * 1.0)
        structure_norm = field_structure / 0.5
        
        preference = (energy_norm * 0.7 + structure_norm * 0.3)
        
        return float(np.clip(preference, 0, 1))
    
    def _apply_parameter_adjustments(self, adjustments: Dict[str, Dict], tick: int):
        """Apply parameter adjustments to strategy module"""
        current_params = self.strategy_module.get_params()
        new_params = current_params.copy()
        
        for param_name, adj_info in adjustments.items():
            adjustment = adj_info['adjustment']
            current_value = current_params[param_name]
            param_range = self.strategy_module.parameter_ranges[param_name]
            
            # Apply adjustment
            new_value = current_value + adjustment
            
            # Clamp to valid range
            new_value = np.clip(new_value, param_range.min_value, param_range.max_value)
            
            # Apply step if discrete
            if param_range.step is not None:
                new_value = round(new_value / param_range.step) * param_range.step
            
            new_params[param_name] = new_value
        
        # Update strategy module parameters
        self.strategy_module.set_params(new_params)
        
        # Record tuning decision
        self.tuning_history.append({
            'tick': tick,
            'old_params': current_params.copy(),
            'new_params': new_params.copy(),
            'adjustments': {name: adj['adjustment'] for name, adj in adjustments.items()},
            'confidence': np.mean([adj['confidence'] for adj in adjustments.values()]),
        })
    
    def strategy(self, state, indicators):
        """
        Generate signal using strategy module with current (tuned) parameters
        """
        # Get current parameters
        current_params = self.strategy_module.get_params()
        
        # Extract scalar indicator values for current tick
        current_indicators = {}
        for key, value in indicators.items():
            if key.startswith('param_'):
                continue  # Skip parameter indicators
            if isinstance(value, np.ndarray):
                tick_idx = len(value) - 1 if len(value) > 0 else 0
                current_indicators[key] = float(value[tick_idx]) if tick_idx < len(value) else 0.5
            elif isinstance(value, (int, float)):
                current_indicators[key] = float(value)
            else:
                current_indicators[key] = 0.5
        
        # Generate signal using strategy module with current parameters
        try:
            signal = self.strategy_module.generate_signal(state, current_indicators, current_params)
            return signal
        except Exception as e:
            print(f"Warning: Strategy module failed: {e}")
            return None
    
    def execution(self, signal, indicators, wallet):
        """Modify signals based on parameter tuning confidence"""
        param_confidence = indicators.get("param_confidence", 0.5)
        
        # Adjust position size based on tuning confidence
        if isinstance(signal, (qx.Buy, qx.Sell)):
            signal.maxvolume = signal.maxvolume * param_confidence
        
        return signal
    
    def autorange(self):
        """Calculate warmup period"""
        return max(
            self.tune.get("market_window", 20),
            10
        )
    
    def reset(self):
        """Reset internal state"""
        # Reinitialize cognitive architecture
        num_params = len(self.strategy_module.parameter_ranges)
        self.mind = CognitiveArchitecture(config={
            'field_size': self.tune['field_size'],
            'action_confidence_threshold': self.confidence_threshold,
            'max_goals': num_params,
            'attention_capacity': num_params,
            'max_thought_depth': self.tune['perception_depth'],
        })
        
        self.mind.set_goal("optimize_strategy_parameters")
        
        # Reset tuning state
        self.param_scores = {}
        self.tuning_history = []
        self.market_perceptions = []
        self.last_tuning_tick = 0
        self.recent_performance = []
        
        # Reset strategy module to default parameters
        default_params = {
            pr.name: pr.default_value 
            for pr in self.strategy_module.parameter_ranges.values()
        }
        self.strategy_module.set_params(default_params)
        
        # Clear caches
        self._market_field_cache = None
        self._param_scores_cache = None
    
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
        """Visualize tuning indicators"""
        # Plot parameter values over time
        param_plots = []
        for param_name in self.strategy_module.parameter_ranges.keys():
            param_key = f"param_{param_name}"
            if param_key in indicators:
                param_plots.append((param_key, param_name, "cyan", 1, "Parameter Values"))
        
        qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            tuple([
                ("param_confidence", "Tuning Confidence", "yellow", 2, "Parameter Tuning"),
                ("param_adjustment", "Parameter Adjustment", "magenta", 2, "Tuning Events"),
                ("market_field_energy", "Market Energy", "green", 3, "Market Perception"),
            ] + param_plots)
        )
    
    def get_tuning_stats(self) -> Dict:
        """Get statistics about parameter tuning"""
        return {
            'current_parameters': self.strategy_module.get_params(),
            'tuning_decisions': len(self.tuning_history),
            'recent_tuning': self.tuning_history[-5:] if self.tuning_history else [],
            'parameter_ranges': {
                name: {
                    'min': pr.min_value,
                    'max': pr.max_value,
                    'current': self.strategy_module.get_params()[name],
                }
                for name, pr in self.strategy_module.parameter_ranges.items()
            },
        }


# Example usage
if __name__ == "__main__":
    # Define a tunable strategy
    def rsi_indicators(data, params):
        """RSI indicators with tunable period"""
        close = data['close']
        rsi_period = int(params.get('rsi_period', 14))
        return {
            'rsi': qx.ti.rsi(close, rsi_period),
        }
    
    def rsi_strategy(state, indicators, params):
        """RSI strategy with tunable thresholds"""
        rsi = indicators.get('rsi', 50)
        oversold = params.get('oversold_threshold', 30)
        overbought = params.get('overbought_threshold', 70)
        
        if rsi < oversold:
            return qx.Buy()
        elif rsi > overbought:
            return qx.Sell()
        return None
    
    # Define parameter ranges
    param_ranges = [
        ParameterRange('rsi_period', 10, 30, 14, step=1, param_type='discrete'),
        ParameterRange('oversold_threshold', 20, 40, 30, step=1, param_type='discrete'),
        ParameterRange('overbought_threshold', 60, 80, 70, step=1, param_type='discrete'),
    ]
    
    # Create tunable strategy module
    strategy_module = TunableStrategyModule(
        name="tunable_rsi",
        indicators_func=rsi_indicators,
        strategy_func=rsi_strategy,
        parameter_ranges=param_ranges,
        metadata={'type': 'momentum', 'description': 'RSI with tunable parameters'}
    )
    
    # Create parameter tuner
    tuner = CombinatronixParameterTuner(
        strategy_module=strategy_module,
        tuning_frequency=10,  # Tune every 10 bars
        confidence_threshold=0.5,
        use_memory=True,
        exploration_rate=0.1
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
    qx.dispatch(tuner, data)
    
    # Print tuning stats
    print("\nParameter Tuning Statistics:")
    stats = tuner.get_tuning_stats()
    print(f"Current parameters: {stats['current_parameters']}")
    print(f"Total tuning decisions: {stats['tuning_decisions']}")

