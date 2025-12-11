"""
QTradeX Adapter for Channel Trading System

This adapter integrates the Channel Algebra trading system with QTradeX's BaseBot framework.

Key Components:
- Channel algebra state encoding (EMPTY, DELTA, PHI, PSI)
- Topology-aware adaptive thresholds
- Multi-scale regime detection
- Intelligent feature scoring
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
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

# Add channelpy_Cursor to path
CHANNELPY_PATH = os.path.join(REFERENCE_PATH, "channelpy_Cursor")
if CHANNELPY_PATH not in sys.path:
    sys.path.insert(0, CHANNELPY_PATH)

try:
    # Import channel trading system components
    # Note: channel_trading.py uses relative imports, so we need to handle this carefully
    import importlib.util
    
    # Load channel_trading module
    channel_trading_path = os.path.join(REFERENCE_PATH, "channel_trading.py")
    if os.path.exists(channel_trading_path):
        spec = importlib.util.spec_from_file_location("channel_trading", channel_trading_path)
        channel_trading = importlib.util.module_from_spec(spec)
        
        # We need to set up the module path for relative imports
        # The file expects to be in channelpy_Cursor/applications/
        sys.modules['channelpy_Cursor'] = __import__('channelpy_Cursor', fromlist=[''])
        sys.modules['channelpy_Cursor.applications'] = __import__('channelpy_Cursor.applications', fromlist=[''])
        
        spec.loader.exec_module(channel_trading)
        
        TechnicalIndicators = channel_trading.TechnicalIndicators
        TradingSignalEncoder = channel_trading.TradingSignalEncoder
        TradingChannelSystem = channel_trading.TradingChannelSystem
        SimpleChannelStrategy = channel_trading.SimpleChannelStrategy
        AdaptiveMomentumStrategy = channel_trading.AdaptiveMomentumStrategy
    else:
        raise ImportError(f"channel_trading.py not found at {channel_trading_path}")
        
except ImportError as e:
    print(f"Warning: Could not import Channel trading system components: {e}")
    print("Make sure the Channel trading system is available in the reference folder.")
    raise

import qtradex as qx


class ChannelQTradeXBot(qx.BaseBot):
    """
    QTradeX BaseBot adapter for Channel Algebra Trading System
    
    Integrates:
    - Channel algebra state encoding (EMPTY, DELTA, PHI, PSI)
    - Topology-aware adaptive thresholds
    - Multi-scale regime detection
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    """
    
    def __init__(self, strategy_type: str = 'simple', use_advanced_adaptive: bool = True):
        """
        Initialize the Channel trading system within QTradeX
        
        Parameters
        ----------
        strategy_type : str
            Strategy type: 'simple' or 'adaptive'
        use_advanced_adaptive : bool
            Whether to use advanced adaptive components (topology-aware, multi-scale, scoring)
        """
        # QTradeX tuning parameters
        self.tune = {
            # Technical indicators
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            
            # Adaptive thresholds
            "adaptive_window": 200,
            "adaptation_rate": 0.01,
            "topology_update_interval": 50,
            
            # Multi-scale regime detection
            "fast_window": 50,
            "medium_window": 200,
            "slow_window": 1000,
            
            # Strategy parameters
            "min_confidence": 0.5,
            "position_size": 0.1,  # 10% of capital per trade
        }
        
        # Optimization bounds
        self.clamps = {
            "rsi_period": [10, 30, 1],
            "macd_fast": [8, 20, 1],
            "macd_slow": [20, 40, 1],
            "adaptive_window": [100, 500, 50],
            "adaptation_rate": [0.005, 0.02, 0.005],
        }
        
        # Store configuration
        self.strategy_type = strategy_type
        self.use_advanced_adaptive = use_advanced_adaptive
        
        # Initialize Channel trading system
        self.channel_system = TradingChannelSystem(
            strategy=strategy_type,
            use_advanced_adaptive=use_advanced_adaptive
        )
        
        # Internal state
        self.is_fitted = False
        self.current_channels = {}
        self.regime_info = {}
        
        # Cached indicators
        self._channel_states_cache = None
        self._regime_cache = None
        self._topology_cache = None
    
    def indicators(self, data):
        """
        Compute channel algebra indicators
        
        This is called once with the full dataset at backtest start.
        """
        # Convert QTradeX data format to pandas DataFrame
        df = pd.DataFrame({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
        })
        
        # Fit the channel system on historical data
        self.channel_system.fit(df)
        self.is_fitted = True
        
        n = len(data['close'])
        
        # Initialize arrays for indicators
        price_channel = np.zeros(n, dtype=int)  # Channel state encoding
        volume_channel = np.zeros(n, dtype=int)
        volatility_channel = np.zeros(n, dtype=int)
        regime_signal = np.zeros(n)
        topology_signal = np.zeros(n)
        
        # Process data to compute channel states
        for i in range(n):
            # Get single bar
            row = df.iloc[i:i+1]
            
            # Process bar to get channel states
            signal = self.channel_system.process_bar(row.iloc[0])
            
            # Get channel states from encoder
            channels = self.channel_system.encoder.encode_all_channels(row)
            
            # Extract channel states (convert State objects to numeric)
            # State encoding: EMPTY=0, DELTA=1, PHI=2, PSI=3
            if len(channels['price']) > 0:
                price_state = channels['price'][-1]
                price_channel[i] = self._state_to_int(price_state)
            
            if len(channels['volume']) > 0:
                volume_state = channels['volume'][-1]
                volume_channel[i] = self._state_to_int(volume_state)
            
            if len(channels['volatility']) > 0:
                vol_state = channels['volatility'][-1]
                volatility_channel[i] = self._state_to_int(vol_state)
            
            # Get regime information if using advanced adaptive
            if self.use_advanced_adaptive:
                regime_info = self.channel_system.encoder.get_regime_info()
                
                # Map regime to signal value
                regime_map = {
                    'TRENDING': 0.7,
                    'VOLATILE': 0.3,
                    'STABLE': 0.5,
                    'UNKNOWN': 0.5
                }
                current_regime = regime_info.get('current_regime', 'UNKNOWN')
                regime_confidence = regime_info.get('regime_confidence', 0.5)
                regime_signal[i] = regime_map.get(current_regime, 0.5) * regime_confidence
                
                # Get topology information
                topology_info = self.channel_system.encoder.get_topology_info()
                if topology_info:
                    # Aggregate topology signals
                    topology_values = []
                    for channel_name, topo in topology_info.items():
                        if topo and 'persistence' in topo:
                            topology_values.append(topo['persistence'])
                    if topology_values:
                        topology_signal[i] = np.mean(topology_values)
                    else:
                        topology_signal[i] = 0.5
                else:
                    topology_signal[i] = 0.5
            else:
                regime_signal[i] = 0.5
                topology_signal[i] = 0.5
        
        # Cache for potential reuse
        self._channel_states_cache = {
            'price': price_channel,
            'volume': volume_channel,
            'volatility': volatility_channel
        }
        self._regime_cache = regime_signal
        self._topology_cache = topology_signal
        
        return {
            "price_channel": price_channel.astype(float),  # Convert to float for QTradeX
            "volume_channel": volume_channel.astype(float),
            "volatility_channel": volatility_channel.astype(float),
            "regime": regime_signal,
            "topology": topology_signal,
        }
    
    def _state_to_int(self, state) -> int:
        """Convert channel state to integer"""
        # Try to access state attributes
        if hasattr(state, 'i') and hasattr(state, 'q'):
            # State has i and q bits
            i_bit = int(state.i) if hasattr(state.i, '__bool__') else (1 if state.i else 0)
            q_bit = int(state.q) if hasattr(state.q, '__bool__') else (1 if state.q else 0)
            
            # Map to channel states: EMPTY=0, DELTA=1, PHI=2, PSI=3
            if i_bit == 0 and q_bit == 0:
                return 0  # EMPTY
            elif i_bit == 1 and q_bit == 0:
                return 1  # DELTA
            elif i_bit == 0 and q_bit == 1:
                return 2  # PHI
            elif i_bit == 1 and q_bit == 1:
                return 3  # PSI
        elif isinstance(state, (int, np.integer)):
            return int(state)
        elif hasattr(state, '__int__'):
            return int(state)
        
        # Default fallback
        return 0
    
    def strategy(self, state, indicators):
        """
        Use channel algebra indicators for trading decisions
        
        Channel states:
        - 0 (EMPTY): No signal, sell
        - 1 (DELTA): Weak signal, hold
        - 2 (PHI): Potential signal, watch
        - 3 (PSI): Strong signal, buy
        """
        price = state["close"]
        wallet = state.get("wallet")
        
        # Get indicators (scalar values for current tick)
        price_channel = indicators.get("price_channel", 0.0)
        volume_channel = indicators.get("volume_channel", 0.0)
        volatility_channel = indicators.get("volatility_channel", 0.0)
        regime = indicators.get("regime", 0.5)
        topology = indicators.get("topology", 0.5)
        
        # Get current position from wallet
        if wallet:
            asset = "BTC"  # Will be determined from data
            currency = "USDT"
            current_position = wallet.get(asset, 0.0)
        else:
            current_position = 0.0
        
        # Channel algebra decision logic
        # PSI (3) = strong buy signal
        # EMPTY (0) = sell signal
        
        # Strong buy: All channels in PSI state
        if price_channel >= 3.0 and volume_channel >= 3.0:
            if current_position <= 0:
                return qx.Buy()
        
        # Buy: Price and volume in PSI
        if price_channel >= 3.0 and volume_channel >= 2.0:
            if current_position <= 0:
                return qx.Buy()
        
        # Sell: Any channel in EMPTY state
        if price_channel <= 0.5 or volume_channel <= 0.5:
            if current_position > 0:
                return qx.Sell()
        
        # Regime-aware adjustments
        # In trending regime, be more aggressive
        if regime > 0.6:  # Trending
            if price_channel >= 2.0 and current_position <= 0:
                return qx.Buy()
        
        # In volatile regime, be more conservative
        if regime < 0.4:  # Volatile
            if price_channel <= 1.0 and current_position > 0:
                return qx.Sell()
        
        # Topology-aware thresholds
        # High topology signal = strong structure, use tighter thresholds
        if topology > 0.6:
            buy_threshold = price * (1 - 0.01)
            sell_threshold = price * (1 + 0.01)
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
        Modify signals based on channel states and regime
        """
        price_channel = indicators.get("price_channel", 0.0)
        regime = indicators.get("regime", 0.5)
        
        # Reduce position size in volatile regimes
        if regime < 0.4:
            if isinstance(signal, (qx.Buy, qx.Sell)):
                signal.maxvolume = signal.maxvolume * 0.5
        
        # Increase position size when all channels are PSI
        if price_channel >= 3.0:
            if isinstance(signal, qx.Buy):
                signal.maxvolume = signal.maxvolume * 1.2
        
        return signal
    
    def autorange(self):
        """Calculate warmup period for channel indicators"""
        return max(
            self.tune.get("adaptive_window", 200) // 4,  # Quarter of adaptive window
            50  # Minimum for regime detection
        )
    
    def reset(self):
        """Reset internal state"""
        self.is_fitted = False
        self.current_channels = {}
        self.regime_info = {}
        
        # Reinitialize channel system
        self.channel_system = TradingChannelSystem(
            strategy=self.strategy_type,
            use_advanced_adaptive=self.use_advanced_adaptive
        )
        
        # Clear caches
        self._channel_states_cache = None
        self._regime_cache = None
        self._topology_cache = None
    
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
        """Visualize channel indicators"""
        qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            (
                ("price_channel", "Price Channel", "cyan", 1, "Channel States"),
                ("volume_channel", "Volume Channel", "yellow", 1, "Channel States"),
                ("volatility_channel", "Volatility Channel", "magenta", 1, "Channel States"),
                ("regime", "Regime", "green", 2, "Market Regime"),
                ("topology", "Topology", "red", 3, "Topology Signal"),
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
    
    # Create bot with simple strategy
    bot = ChannelQTradeXBot(strategy_type='simple', use_advanced_adaptive=True)
    
    # Run backtest
    qx.dispatch(bot, data)

