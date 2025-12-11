"""
QTradeX Adapter for Mythic Trading System

This adapter integrates the Mythic multi-scale, topology-aware, field-theoretic
trading system with QTradeX's BaseBot framework.

Key Components:
- Multi-scale state management (Investment, Day Trade, Scalp)
- 12 Actor consensus system with dodecahedral geometry
- Persistent homology for regime detection
- Field dynamics engine
- A* pathfinding for optimal navigation
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

# Add Mythic system to path
# Look for reference systems in the reference folder
MYTHIC_PATH = os.path.join(os.path.dirname(__file__), "reference")
if MYTHIC_PATH not in sys.path:
    sys.path.insert(0, MYTHIC_PATH)

try:
    # Import Mythic system components
    from core.multi_scale_state import MultiScaleState, MarketTick, TimeScale
    from actors.trading_actors import create_12_actors
    from geometry.dodecahedron import DodecahedralSurface, ActorConsensus
    from topology.persistent_homology import MarketRegimeClassifier
    from navigation.state_space import StateSpaceGraph, PortfolioState
    from navigation.pathfinder import RealTimePathPlanner
    from navigation.ligature import LigatureConstraints
    from dynamics.field_engine import FieldDynamicsEngine
    from dynamics.cobordism_library import CobordismLibrary
except ImportError as e:
    print(f"Warning: Could not import Mythic system components: {e}")
    print("Make sure the Mythic trading system is available at the expected path.")
    raise

import qtradex as qx


class MythicQTradeXBot(qx.BaseBot):
    """
    QTradeX BaseBot adapter for Mythic Trading System
    
    Integrates:
    - Topology-aware decisioning (persistent homology)
    - Field-state thresholds (field dynamics engine)
    - Multi-scale actor consensus
    - Optimal pathfinding navigation
    """
    
    def __init__(self):
        """Initialize the Mythic trading system within QTradeX"""
        # QTradeX tuning parameters
        self.tune = {
            # Regime detection
            "regime_check_interval": 10,  # Check regime every N ticks
            "topology_lookback": 50,       # Days for topology analysis
            
            # Field dynamics
            "field_momentum_decay": 0.95,
            "field_reversion_strength": 0.05,
            "field_volatility_decay": 0.98,
            
            # Actor consensus
            "consensus_method": "smooth",  # 'smooth' or 'weighted'
            "min_confidence": 0.3,         # Minimum actor confidence
            
            # Pathfinding
            "pathfinding_timeout": 0.05,   # 50ms timeout for A*
            "position_quantum": 10,        # Position quantization
            
            # Risk management
            "max_position_size": 0.1,      # 10% of capital per trade
            "stop_loss_pct": 0.05,          # 5% stop loss
        }
        
        # Optimization bounds
        self.clamps = {
            "regime_check_interval": [5, 20, 1],
            "topology_lookback": [30, 100, 5],
            "field_momentum_decay": [0.9, 0.99, 0.01],
            "min_confidence": [0.2, 0.5, 0.05],
        }
        
        # Initialize Mythic system components
        self._initialize_mythic_components()
        
        # Internal state
        self.tick_count = 0
        self.last_regime_check = 0
        self.current_regime = "unknown"
        self.last_consensus = {}
        self.current_position = 0.0
        
        # Cached indicators (computed once per backtest)
        self._topology_signal_cache = None
        self._field_state_cache = None
        self._regime_cache = None
        self._consensus_cache = None
    
    def _initialize_mythic_components(self):
        """Initialize all Mythic system components"""
        # Multi-scale state (will be populated with data)
        self.market_state = None  # Initialized in indicators()
        
        # Dodecahedral surface for actor consensus
        self.surface = DodecahedralSurface()
        
        # 12 Trading actors
        self.actors = create_12_actors(self.surface.vertices)
        
        # Regime classifier (topology-aware)
        self.regime_classifier = MarketRegimeClassifier()
        
        # State space graph for navigation
        self.state_graph = StateSpaceGraph(
            max_investment_pos=100,
            max_day_pos=30,
            max_scalp_pos=10,
            position_quantum=self.tune.get("position_quantum", 10)
        )
        
        # Path planner
        self.path_planner = RealTimePathPlanner(self.state_graph)
        
        # Constraints
        self.constraints = LigatureConstraints()
        
        # Field dynamics engine
        self.field_engine = FieldDynamicsEngine()
        
        # Cobordism library (learning)
        self.cobordism_library = CobordismLibrary()
    
    def indicators(self, data):
        """
        Compute topology-aware and field-state indicators
        
        This is called once with the full dataset at backtest start.
        """
        # Initialize multi-scale state with historical data
        self.market_state = MultiScaleState(
            symbol="BTC",  # Will be set from data
            cash=10000.0
        )
        
        # Convert QTradeX data format to Mythic format
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        unix = data["unix"]
        
        n = len(close)
        
        # Initialize arrays for indicators
        topology_signal = np.zeros(n)
        field_state = np.zeros(n)
        regime_signal = np.zeros(n)
        consensus_signal = np.zeros(n)
        
        # Process data in chunks to build up multi-scale state
        # We need to simulate the tick-by-tick ingestion
        for i in range(n):
            # Create market tick
            timestamp = pd.Timestamp.fromtimestamp(unix[i])
            tick = MarketTick(
                timestamp=timestamp,
                symbol="BTC",
                price=close[i],
                volume=volume[i] if volume[i] > 0 else 1e-6,
                bid=low[i],
                ask=high[i]
            )
            
            # Ingest tick into multi-scale state
            self.market_state.ingest_tick(tick)
            
            # Evolve fields if we have enough data
            if i > 1 and self.field_engine:
                prices = self.market_state.get_scale_state(TimeScale.SCALP).prices
                if len(prices) > 1:
                    price_return = (prices[-1] - prices[-2]) / prices[-2]
                    
                    for scale in TimeScale:
                        self.field_engine.evolve_field(
                            scale, price_return, tick.volume, 0
                        )
                    
                    self.field_engine.cross_scale_coupling()
            
            # Compute indicators once we have enough data
            if i >= self.tune["topology_lookback"]:
                # Topology-aware signal (regime detection)
                day_state = self.market_state.get_scale_state(TimeScale.DAY_TRADE)
                if len(day_state.prices) >= 30:
                    regime_info = self.regime_classifier.classify_regime(day_state.prices)
                    regime = regime_info.get('regime', 'unknown')
                    confidence = regime_info.get('confidence', 0.0)
                    
                    # Map regime to signal value
                    regime_map = {
                        'trending': 0.7,
                        'choppy': 0.3,
                        'range_bound': 0.5,
                        'unknown': 0.5
                    }
                    topology_signal[i] = regime_map.get(regime, 0.5) * confidence
                    regime_signal[i] = regime_map.get(regime, 0.5)
                else:
                    topology_signal[i] = 0.5
                    regime_signal[i] = 0.5
                
                # Field-state threshold (from field dynamics)
                if self.field_engine:
                    # Get field state from investment scale
                    inv_state = self.market_state.get_scale_state(TimeScale.INVESTMENT)
                    field_vec = self.field_engine.get_field_state(TimeScale.INVESTMENT)
                    
                    if field_vec is not None:
                        # Combine momentum, mean reversion, and volatility
                        momentum = field_vec.momentum
                        reversion = field_vec.mean_reversion
                        volatility = field_vec.volatility
                        
                        # Field-state as weighted combination
                        field_state[i] = (
                            abs(momentum) * 0.4 +
                            abs(reversion) * 0.3 +
                            volatility * 0.3
                        ) * 100  # Scale to reasonable range
                    else:
                        field_state[i] = 0.0
                else:
                    field_state[i] = 0.0
                
                # Actor consensus signal
                recommendations = []
                for actor in self.actors:
                    try:
                        rec = actor.analyze(self.market_state)
                        # Adjust for regime
                        if regime_signal[i] < 0.4:  # Choppy regime
                            if actor.actor_type.value == 'trend_rider':
                                rec.confidence *= 0.5
                        recommendations.append(rec)
                    except Exception:
                        continue
                
                if recommendations:
                    # Build consensus
                    consensus = self._build_consensus(recommendations)
                    
                    # Aggregate consensus across scales
                    consensus_values = []
                    for scale in TimeScale:
                        if scale in consensus:
                            consensus_values.append(
                                consensus[scale].aggregated_value
                            )
                    
                    if consensus_values:
                        consensus_signal[i] = np.mean(consensus_values)
                    else:
                        consensus_signal[i] = 0.0
                else:
                    consensus_signal[i] = 0.0
            else:
                # Not enough data yet
                topology_signal[i] = 0.5
                field_state[i] = 0.0
                regime_signal[i] = 0.5
                consensus_signal[i] = 0.0
        
        # Cache for potential reuse
        self._topology_signal_cache = topology_signal
        self._field_state_cache = field_state
        self._regime_cache = regime_signal
        self._consensus_cache = consensus_signal
        
        return {
            "topology_signal": topology_signal,
            "field_state": field_state,
            "regime": regime_signal,
            "consensus": consensus_signal,
        }
    
    def _build_consensus(self, recommendations) -> Dict:
        """Build consensus from actor recommendations"""
        consensus = {}
        
        for scale in TimeScale:
            scale_recs = [r for r in recommendations if r.scale == scale]
            if not scale_recs:
                continue
            
            values = np.array([r.target_position for r in scale_recs])
            confidences = np.array([r.confidence for r in scale_recs])
            
            # Filter by minimum confidence
            min_conf = self.tune.get("min_confidence", 0.3)
            mask = confidences >= min_conf
            if not np.any(mask):
                continue
            
            values = values[mask]
            confidences = confidences[mask]
            
            # Pad to 12 if needed
            if len(values) < 12:
                pad = 12 - len(values)
                values = np.concatenate([values, np.zeros(pad)])
                confidences = np.concatenate([confidences, np.zeros(pad)])
            
            # Aggregate using dodecahedral surface
            method = self.tune.get("consensus_method", "smooth")
            aggregated = self.surface.weighted_aggregate(values, confidences, method=method)
            smoothed = self.surface.smooth_field(values)
            
            consensus[scale] = ActorConsensus(
                aggregated_value=aggregated,
                individual_values=values,
                confidences=confidences,
                smoothed_values=smoothed,
                method=method
            )
        
        return consensus
    
    def strategy(self, state, indicators):
        """
        Use topology-aware and field-state indicators for trading decisions
        
        Combines:
        - Topology-aware regime detection
        - Field-state thresholds
        - Actor consensus
        - Pathfinding navigation
        """
        price = state["close"]
        wallet = state.get("wallet")
        
        # Get indicators (scalar values for current tick)
        topology = indicators.get("topology_signal", 0.5)
        field_state = indicators.get("field_state", 0.0)
        regime = indicators.get("regime", 0.5)
        consensus = indicators.get("consensus", 0.0)
        
        # Get current position from wallet
        if wallet:
            asset = "BTC"  # Will be determined from data
            currency = "USDT"
            current_position = wallet.get(asset, 0.0)
        else:
            current_position = 0.0
        
        # Topology-aware decisioning
        # High topology signal + low field state = buy opportunity
        if topology > 0.6 and field_state < 50:
            if current_position <= 0:  # Not already long
                return qx.Buy()
        
        # Field-state threshold logic
        # High field state = sell signal (volatility/uncertainty)
        if field_state > 100:
            if current_position > 0:
                return qx.Sell()
        
        # Consensus-based decisioning
        # Strong positive consensus = buy
        if consensus > 20:
            if current_position <= 0:
                return qx.Buy()
        
        # Strong negative consensus = sell
        if consensus < -20:
            if current_position > 0:
                return qx.Sell()
        
        # Regime-adjusted thresholds
        # In trending regime, use momentum-based thresholds
        if regime > 0.6:  # Trending
            buy_threshold = price * (1 - topology * 0.01)
            sell_threshold = price * (1 + topology * 0.01)
        else:  # Choppy/range-bound
            buy_threshold = price * (1 - field_state * 0.001)
            sell_threshold = price * (1 + field_state * 0.001)
        
        return qx.Thresholds(
            buying=buy_threshold,
            selling=sell_threshold,
            maxvolume=self.tune.get("max_position_size", 0.1)
        )
    
    def execution(self, signal, indicators, wallet):
        """
        Modify signals based on field-state and constraints
        """
        field_state = indicators.get("field_state", 0.0)
        
        # Reduce position size in high field-state (volatile conditions)
        if field_state > 80:
            if isinstance(signal, (qx.Buy, qx.Sell)):
                signal.maxvolume = signal.maxvolume * 0.5
        
        # Apply stop loss
        if isinstance(signal, qx.Buy) and hasattr(self, 'last_buy_price'):
            if self.last_buy_price:
                stop_loss = self.last_buy_price * (1 - self.tune.get("stop_loss_pct", 0.05))
                # This would need to be handled in strategy() for proper stop loss
        
        return signal
    
    def autorange(self):
        """Calculate warmup period for custom indicators"""
        return max(
            self.tune.get("topology_lookback", 50),
            30  # Minimum for regime detection
        )
    
    def reset(self):
        """Reset internal state"""
        self.tick_count = 0
        self.last_regime_check = 0
        self.current_regime = "unknown"
        self.last_consensus = {}
        self.current_position = 0.0
        
        # Reset Mythic components
        if self.market_state:
            self.market_state = None
        if self.field_engine:
            self.field_engine = FieldDynamicsEngine()
        if self.cobordism_library:
            self.cobordism_library = CobordismLibrary()
        
        # Clear caches
        self._topology_signal_cache = None
        self._field_state_cache = None
        self._regime_cache = None
        self._consensus_cache = None
    
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
        """Visualize custom indicators"""
        qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            (
                ("topology_signal", "Topology Signal", "cyan", 1, "Topology-Aware"),
                ("field_state", "Field State", "magenta", 2, "Field-State Thresholds"),
                ("regime", "Regime", "yellow", 1, "Market Regime"),
                ("consensus", "Actor Consensus", "green", 3, "Consensus Signal"),
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
    
    # Create bot
    bot = MythicQTradeXBot()
    
    # Run backtest
    qx.dispatch(bot, data)

