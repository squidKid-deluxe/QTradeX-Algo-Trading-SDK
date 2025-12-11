"""
Trading application using channel algebra

Complete trading system with technical indicators, signal generation, and backtesting
Now enhanced with advanced adaptive components: topology-aware thresholds, multi-scale regime detection, and intelligent feature scoring
"""
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.parallel import ParallelChannels
from ..adaptive.streaming import StreamingAdaptiveThreshold
from ..adaptive.topology_adaptive import TopologyAdaptiveThreshold, TopologyAnalyzer
from ..adaptive.multiscale import MultiScaleAdaptiveThreshold, RegimeType, RegimeChange
from ..adaptive.scoring import FeatureScorer, create_trading_scorer
from ..pipeline.encoders import DualFeatureEncoder
from ..pipeline.interpreters import RuleBasedInterpreter, FSMInterpreter


class TechnicalIndicators:
    """
    Technical analysis indicators for trading
    
    Examples
    --------
    >>> indicators = TechnicalIndicators()
    >>> df['rsi'] = indicators.rsi(df['close'], period=14)
    >>> df['macd'], df['signal'] = indicators.macd(df['close'])
    """
    
    @staticmethod
    def sma(prices: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Returns values from 0 to 100
        >70 = overbought, <30 = oversold
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Returns
        -------
        macd_line : pd.Series
            MACD line
        signal_line : pd.Series
            Signal line
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                       num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Returns
        -------
        upper : pd.Series
            Upper band
        middle : pd.Series
            Middle band (SMA)
        lower : pd.Series
            Lower band
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range (volatility measure)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv


class TradingSignalEncoder:
    """
    Encode trading signals into channel states
    
    Uses multiple technical indicators to determine i and q bits
    Now enhanced with topology-aware adaptive thresholds, multi-scale regime detection, and intelligent feature scoring
    
    Examples
    --------
    >>> encoder = TradingSignalEncoder()
    >>> encoder.fit(df)
    >>> states = encoder.encode(df)
    """
    
    def __init__(self, use_advanced_adaptive: bool = True):
        """
        Initialize trading signal encoder
        
        Parameters
        ----------
        use_advanced_adaptive : bool
            Whether to use advanced adaptive components (topology-aware, multi-scale, scoring)
        """
        self.use_advanced_adaptive = use_advanced_adaptive
        self.indicators = TechnicalIndicators()
        self.is_fitted = False
        
        if use_advanced_adaptive:
            # Create intelligent feature scorer for trading
            self.scorer = create_trading_scorer()
            
            # Create topology-aware adaptive thresholds
            self.price_threshold = TopologyAdaptiveThreshold(
                window_size=200,
                adaptation_rate=0.01,
                topology_update_interval=50,
                feature_scorer=self.scorer
            )
            self.volume_threshold = TopologyAdaptiveThreshold(
                window_size=200,
                adaptation_rate=0.01,
                topology_update_interval=50,
                feature_scorer=self.scorer
            )
            self.volatility_threshold = TopologyAdaptiveThreshold(
                window_size=200,
                adaptation_rate=0.01,
                topology_update_interval=50,
                feature_scorer=self.scorer
            )
            
            # Create multi-scale regime detector
            self.regime_detector = MultiScaleAdaptiveThreshold(
                use_topology=True,
                fast_window=50,    # ~1 month
                medium_window=200, # ~4 months
                slow_window=1000   # ~2 years
            )
            
            # Track regime changes
            self.regime_changes = []
            self.current_regime = RegimeType.UNKNOWN
        else:
            # Fallback to basic adaptive thresholds
            self.price_threshold = StreamingAdaptiveThreshold(window_size=50)
            self.volume_threshold = StreamingAdaptiveThreshold(window_size=50)
            self.volatility_threshold = StreamingAdaptiveThreshold(window_size=50)
            self.regime_detector = None
            self.regime_changes = []
            self.current_regime = None
    
    def fit(self, df: pd.DataFrame):
        """
        Initialize with historical data
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: open, high, low, close, volume
        """
        if self.use_advanced_adaptive:
            # Use advanced adaptive components
            for i, row in df.iterrows():
                # Update topology-aware thresholds
                self.price_threshold.update(row['close'])
                self.volume_threshold.update(row['volume'])
                
                # Compute volatility
                if len(self.price_threshold.window) >= 20:
                    volatility = self._compute_volatility()
                    self.volatility_threshold.update(volatility)
                
                # Update regime detector
                self.regime_detector.update(row['close'])
                
                # Check for regime changes
                if self.regime_detector.regime_changed():
                    change = self.regime_detector.get_last_regime_change()
                    self.regime_changes.append(change)
                    self.current_regime = change.to_regime
                
                # Score features for intelligent adaptation
                if i > 100:  # Need enough history for scoring
                    context = self._create_trading_context(row, df, i)
                    self.scorer.record_score(row['close'], context, outcome=row['close'])
        else:
            # Use basic adaptive thresholds
            for _, row in df.iterrows():
                self.price_threshold.update(row['close'])
                self.volume_threshold.update(row['volume'])
                
                # Compute volatility
                if len(self.price_threshold.window) >= 20:
                    volatility = self._compute_volatility()
                    self.volatility_threshold.update(volatility)
        
        self.is_fitted = True
        return self
    
    def _create_trading_context(self, row: pd.Series, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Create trading context for feature scoring"""
        # Get historical data for context
        start_idx = max(0, index - 100)
        historical_data = df.iloc[start_idx:index]
        
        if len(historical_data) == 0:
            return {
                'historical_values': [row['close']],
                'historical_outcomes': [row['close']],
                'sample_size': 1,
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.1
            }
        
        # Calculate returns for outcomes
        returns = historical_data['close'].pct_change().dropna()
        
        return {
            'historical_values': historical_data['close'].tolist(),
            'historical_outcomes': returns.tolist(),
            'sample_size': len(historical_data),
            'age_seconds': 0,  # Real-time data
            'missing_rate': 0.01,
            'noise_level': returns.std() if len(returns) > 0 else 0.1
        }
    
    def encode_price_channel(self, df: pd.DataFrame) -> StateArray:
        """
        Encode price momentum channel with advanced adaptive thresholds
        
        i-bit: Price above adaptive threshold
        q-bit: Strong trend (RSI not neutral) with regime awareness
        """
        if self.use_advanced_adaptive:
            # Use topology-aware adaptive thresholds
            states = []
            for _, row in df.iterrows():
                # Update thresholds
                self.price_threshold.update(row['close'])
                
                # Encode with adaptive threshold
                state = self.price_threshold.encode(row['close'])
                states.append(state)
            
            return StateArray(states)
        else:
            # Use basic encoding
            sma_20 = self.indicators.sma(df['close'], 20)
            rsi = self.indicators.rsi(df['close'], 14)
            
            i_bits = (df['close'] > sma_20).astype(int)
            q_bits = ((rsi > 60) | (rsi < 40)).astype(int)
            
            return StateArray(i=i_bits.values, q=q_bits.values)
    
    def encode_volume_channel(self, df: pd.DataFrame) -> StateArray:
        """
        Encode volume channel with advanced adaptive thresholds
        
        i-bit: Volume above adaptive threshold
        q-bit: Volume surge with regime awareness
        """
        if self.use_advanced_adaptive:
            # Use topology-aware adaptive thresholds
            states = []
            for _, row in df.iterrows():
                # Update thresholds
                self.volume_threshold.update(row['volume'])
                
                # Encode with adaptive threshold
                state = self.volume_threshold.encode(row['volume'])
                states.append(state)
            
            return StateArray(states)
        else:
            # Use basic encoding
            vol_ma = df['volume'].rolling(50).mean()
            
            i_bits = (df['volume'] > vol_ma).astype(int)
            q_bits = (df['volume'] > vol_ma * 1.5).astype(int)
            
            return StateArray(i=i_bits.values, q=q_bits.values)
    
    def encode_volatility_channel(self, df: pd.DataFrame) -> StateArray:
        """
        Encode volatility channel with advanced adaptive thresholds
        
        i-bit: Above adaptive volatility threshold
        q-bit: High volatility regime with topology awareness
        """
        if self.use_advanced_adaptive:
            # Use topology-aware adaptive thresholds
            states = []
            for _, row in df.iterrows():
                # Compute volatility
                volatility = self._compute_volatility()
                
                # Update thresholds
                self.volatility_threshold.update(volatility)
                
                # Encode with adaptive threshold
                state = self.volatility_threshold.encode(volatility)
                states.append(state)
            
            return StateArray(states)
        else:
            # Use basic encoding
            atr = self.indicators.atr(df['high'], df['low'], df['close'], 14)
            
            vol_median = atr.median()
            vol_75 = atr.quantile(0.75)
            
            i_bits = (atr > vol_median).astype(int)
            q_bits = (atr > vol_75).astype(int)
            
            return StateArray(i=i_bits.values, q=q_bits.values)
    
    def encode_all_channels(self, df: pd.DataFrame) -> Dict[str, StateArray]:
        """
        Encode all trading channels with regime awareness
        
        Returns
        -------
        channels : dict
            Dictionary of channel name -> StateArray
        """
        return {
            'price': self.encode_price_channel(df),
            'volume': self.encode_volume_channel(df),
            'volatility': self.encode_volatility_channel(df)
        }
    
    def get_regime_info(self) -> Dict[str, Any]:
        """
        Get current regime information
        
        Returns
        -------
        regime_info : dict
            Current regime information including type, confidence, and changes
        """
        if not self.use_advanced_adaptive or self.regime_detector is None:
            return {
                'current_regime': 'UNKNOWN',
                'regime_confidence': 0.5,
                'regime_changes': 0,
                'last_change': None
            }
        
        regime_info = self.regime_detector.get_regime_info()
        return {
            'current_regime': regime_info['current_regime'],
            'regime_confidence': regime_info['confidence'],
            'regime_changes': len(self.regime_changes),
            'last_change': self.regime_changes[-1] if self.regime_changes else None
        }
    
    def get_topology_info(self) -> Dict[str, Any]:
        """
        Get topology information for all channels
        
        Returns
        -------
        topology_info : dict
            Topology information for each channel
        """
        if not self.use_advanced_adaptive:
            return {}
        
        return {
            'price': self.price_threshold.get_topology(),
            'volume': self.volume_threshold.get_topology(),
            'volatility': self.volatility_threshold.get_topology()
        }
    
    def _compute_volatility(self):
        """Compute current volatility from price window"""
        if len(self.price_threshold.window) < 20:
            return 0.0
        
        prices = np.array(self.price_threshold.window[-20:])
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # Annualized


class TradingStrategy:
    """
    Base class for trading strategies
    
    Subclass this to implement custom strategies
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions = []  # List of (timestamp, action, price, size)
        self.current_position = 0.0
    
    def generate_signal(self, channels: Dict[str, State]) -> Dict:
        """
        Generate trading signal from channel states
        
        Parameters
        ----------
        channels : dict
            Dictionary of channel states
            
        Returns
        -------
        signal : dict
            Trading signal with 'action', 'size', 'confidence'
        """
        raise NotImplementedError
    
    def execute_signal(self, signal: Dict, price: float, timestamp):
        """Record trade execution"""
        if signal['action'] != 'HOLD':
            self.positions.append({
                'timestamp': timestamp,
                'action': signal['action'],
                'price': price,
                'size': signal.get('size', 1.0),
                'confidence': signal.get('confidence', 0.5)
            })
            
            if signal['action'] == 'BUY':
                self.current_position += signal.get('size', 1.0)
            elif signal['action'] == 'SELL':
                self.current_position -= signal.get('size', 1.0)


class SimpleChannelStrategy(TradingStrategy):
    """
    Simple strategy based on channel states with regime awareness
    
    Rules:
    - All channels ψ → Strong buy
    - Price ψ + Volume ψ → Buy
    - Any channel ∅ → Sell
    - Otherwise → Hold
    
    Now enhanced with regime-aware decision making
    
    Examples
    --------
    >>> strategy = SimpleChannelStrategy()
    >>> signal = strategy.generate_signal({
    ...     'price': PSI,
    ...     'volume': PSI,
    ...     'volatility': DELTA
    ... })
    >>> print(signal)
    {'action': 'BUY', 'size': 1.0, 'confidence': 0.8}
    """
    
    def __init__(self, use_regime_awareness: bool = True):
        super().__init__(name="SimpleChannel")
        self.use_regime_awareness = use_regime_awareness
        self.interpreter = RuleBasedInterpreter()
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup interpretation rules"""
        # Strong buy: all good
        self.interpreter.add_rule(
            lambda c: all(s == PSI for s in c.values()),
            {'action': 'BUY', 'size': 1.0, 'confidence': 1.0}
        )
        
        # Buy: price and volume good
        self.interpreter.add_rule(
            lambda c: c.get('price') == PSI and c.get('volume') == PSI,
            {'action': 'BUY', 'size': 0.8, 'confidence': 0.8}
        )
        
        # Sell: any channel empty
        self.interpreter.add_rule(
            lambda c: any(s == EMPTY for s in c.values()),
            {'action': 'SELL', 'size': 1.0, 'confidence': 0.9}
        )
        
        # Default: hold
        self.interpreter.set_default({'action': 'HOLD', 'size': 0.0, 'confidence': 0.5})
    
    def generate_signal(self, channels: Dict[str, State], regime_info: Dict[str, Any] = None) -> Dict:
        """
        Generate signal using rule-based interpreter with regime awareness
        
        Parameters
        ----------
        channels : dict
            Dictionary of channel states
        regime_info : dict, optional
            Regime information for enhanced decision making
            
        Returns
        -------
        signal : dict
            Trading signal with regime-aware adjustments
        """
        # Get base signal
        signal = self.interpreter.interpret(channels)
        
        # Apply regime-aware adjustments
        if self.use_regime_awareness and regime_info is not None:
            signal = self._apply_regime_adjustments(signal, regime_info)
        
        return signal
    
    def _apply_regime_adjustments(self, signal: Dict, regime_info: Dict[str, Any]) -> Dict:
        """Apply regime-aware adjustments to trading signal"""
        current_regime = regime_info.get('current_regime', 'UNKNOWN')
        regime_confidence = regime_info.get('regime_confidence', 0.5)
        
        # Adjust confidence based on regime
        if current_regime == 'VOLATILE':
            # Reduce confidence in volatile markets
            signal['confidence'] *= 0.8
        elif current_regime == 'STABLE':
            # Increase confidence in stable markets
            signal['confidence'] *= 1.1
        elif current_regime == 'TRENDING':
            # Adjust for trending markets
            if signal['action'] == 'BUY':
                signal['confidence'] *= 1.05
            elif signal['action'] == 'SELL':
                signal['confidence'] *= 0.95
        
        # Adjust position size based on regime confidence
        signal['size'] *= regime_confidence
        
        return signal


class AdaptiveMomentumStrategy(TradingStrategy):
    """
    Adaptive momentum strategy using FSM with regime awareness
    
    Modes: WAITING, ACCUMULATING, HOLDING, DISTRIBUTING
    Transitions based on channel states and market regimes
    
    Examples
    --------
    >>> strategy = AdaptiveMomentumStrategy()
    >>> for row in df.itertuples():
    ...     channels = encoder.encode_row(row)
    ...     signal = strategy.generate_signal(channels)
    ...     strategy.execute_signal(signal, row.close, row.Index)
    """
    
    def __init__(self, use_regime_awareness: bool = True):
        super().__init__(name="AdaptiveMomentum")
        self.use_regime_awareness = use_regime_awareness
        self.fsm = FSMInterpreter(initial_mode='WAITING')
        self._setup_fsm()
    
    def _setup_fsm(self):
        """Setup finite state machine"""
        
        # From WAITING
        self.fsm.add_transition('WAITING', PSI, 'ACCUMULATING', 
                               {'action': 'BUY', 'size': 0.3, 'confidence': 0.7})
        self.fsm.add_transition('WAITING', DELTA, 'WAITING',
                               {'action': 'HOLD', 'size': 0.0, 'confidence': 0.5})
        
        # From ACCUMULATING
        self.fsm.add_transition('ACCUMULATING', PSI, 'ACCUMULATING',
                               {'action': 'BUY', 'size': 0.3, 'confidence': 0.8})
        self.fsm.add_transition('ACCUMULATING', DELTA, 'HOLDING',
                               {'action': 'HOLD', 'size': 0.0, 'confidence': 0.6})
        self.fsm.add_transition('ACCUMULATING', EMPTY, 'DISTRIBUTING',
                               {'action': 'SELL', 'size': 0.5, 'confidence': 0.7})
        
        # From HOLDING
        self.fsm.add_transition('HOLDING', PSI, 'ACCUMULATING',
                               {'action': 'BUY', 'size': 0.2, 'confidence': 0.6})
        self.fsm.add_transition('HOLDING', EMPTY, 'DISTRIBUTING',
                               {'action': 'SELL', 'size': 0.5, 'confidence': 0.8})
        self.fsm.add_transition('HOLDING', DELTA, 'HOLDING',
                               {'action': 'HOLD', 'size': 0.0, 'confidence': 0.5})
        
        # From DISTRIBUTING
        self.fsm.add_transition('DISTRIBUTING', EMPTY, 'DISTRIBUTING',
                               {'action': 'SELL', 'size': 0.5, 'confidence': 0.9})
        self.fsm.add_transition('DISTRIBUTING', PHI, 'WAITING',
                               {'action': 'HOLD', 'size': 0.0, 'confidence': 0.5})
    
    def generate_signal(self, channels: Dict[str, State], regime_info: Dict[str, Any] = None) -> Dict:
        """
        Generate signal using FSM with regime awareness
        
        Parameters
        ----------
        channels : dict
            Dictionary of channel states
        regime_info : dict, optional
            Regime information for enhanced decision making
            
        Returns
        -------
        signal : dict
            Trading signal with regime-aware adjustments
        """
        # Use price channel as primary signal
        primary_state = channels.get('price', EMPTY)
        signal = self.fsm.process(primary_state)
        
        # Apply regime-aware adjustments
        if self.use_regime_awareness and regime_info is not None:
            signal = self._apply_regime_adjustments(signal, regime_info)
        
        return signal
    
    def _apply_regime_adjustments(self, signal: Dict, regime_info: Dict[str, Any]) -> Dict:
        """Apply regime-aware adjustments to trading signal"""
        current_regime = regime_info.get('current_regime', 'UNKNOWN')
        regime_confidence = regime_info.get('regime_confidence', 0.5)
        
        # Adjust confidence based on regime
        if current_regime == 'VOLATILE':
            # Reduce confidence in volatile markets
            signal['confidence'] *= 0.8
        elif current_regime == 'STABLE':
            # Increase confidence in stable markets
            signal['confidence'] *= 1.1
        elif current_regime == 'TRENDING':
            # Adjust for trending markets
            if signal['action'] == 'BUY':
                signal['confidence'] *= 1.05
            elif signal['action'] == 'SELL':
                signal['confidence'] *= 0.95
        
        # Adjust position size based on regime confidence
        signal['size'] *= regime_confidence
        
        return signal


class TradingChannelSystem:
    """
    Complete trading system combining encoder, strategy, and backtester
    Now enhanced with advanced adaptive components: topology-aware thresholds, multi-scale regime detection, and intelligent feature scoring
    
    Examples
    --------
    >>> system = TradingChannelSystem(strategy='simple')
    >>> system.fit(train_df)
    >>> results = system.backtest(test_df)
    >>> print(f"Total return: {results['total_return']:.2%}")
    """
    
    def __init__(self, strategy: str = 'simple', use_advanced_adaptive: bool = True):
        """
        Parameters
        ----------
        strategy : str
            Strategy type: 'simple', 'adaptive', or custom
        use_advanced_adaptive : bool
            Whether to use advanced adaptive components
        """
        self.encoder = TradingSignalEncoder(use_advanced_adaptive=use_advanced_adaptive)
        self.use_advanced_adaptive = use_advanced_adaptive
        
        if strategy == 'simple':
            self.strategy = SimpleChannelStrategy(use_regime_awareness=use_advanced_adaptive)
        elif strategy == 'adaptive':
            self.strategy = AdaptiveMomentumStrategy(use_regime_awareness=use_advanced_adaptive)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit system on historical data
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV data
        """
        self.encoder.fit(df)
        self.is_fitted = True
        return self
    
    def process_bar(self, row: pd.Series) -> Dict:
        """
        Process single bar of data with advanced adaptive components
        
        Parameters
        ----------
        row : pd.Series
            OHLCV data for one bar
            
        Returns
        -------
        signal : dict
            Trading signal with regime awareness
        """
        # Update adaptive thresholds
        self.encoder.price_threshold.update(row['close'])
        self.encoder.volume_threshold.update(row['volume'])
        
        # Update regime detector if using advanced adaptive
        if self.use_advanced_adaptive:
            self.encoder.regime_detector.update(row['close'])
        
        # Encode to states (need to pass as DataFrame)
        df_single = pd.DataFrame([row])
        channels = self.encoder.encode_all_channels(df_single)
        
        # Get states for this bar (last index)
        current_channels = {
            name: states[-1] 
            for name, states in channels.items()
        }
        
        # Get regime information if using advanced adaptive
        regime_info = None
        if self.use_advanced_adaptive:
            regime_info = self.encoder.get_regime_info()
        
        # Generate signal with regime awareness
        signal = self.strategy.generate_signal(current_channels, regime_info)
        
        return signal
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        
        Returns
        -------
        system_info : dict
            System information including regime, topology, and performance metrics
        """
        info = {
            'use_advanced_adaptive': self.use_advanced_adaptive,
            'strategy_name': self.strategy.name,
            'is_fitted': self.is_fitted
        }
        
        if self.use_advanced_adaptive:
            # Add regime information
            info['regime_info'] = self.encoder.get_regime_info()
            
            # Add topology information
            info['topology_info'] = self.encoder.get_topology_info()
            
            # Add feature scorer statistics
            if hasattr(self.encoder, 'scorer'):
                info['scorer_stats'] = {
                    'total_scores': len(self.encoder.scorer.score_history),
                    'dimensions': len(self.encoder.scorer.dimensions)
                }
        
        return info
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                commission: float = 0.001) -> Dict:
        """
        Backtest strategy on historical data
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV data
        initial_capital : float
            Starting capital
        commission : float
            Commission rate (0.001 = 0.1%)
            
        Returns
        -------
        results : dict
            Backtest results with metrics
        """
        if not self.is_fitted:
            raise RuntimeError("System not fitted. Call fit() first.")
        
        capital = initial_capital
        shares = 0
        equity_curve = []
        trades = []
        
        for idx, row in df.iterrows():
            # Generate signal
            signal = self.process_bar(row)
            
            # Execute trade
            if signal['action'] == 'BUY' and capital > 0:
                # Buy shares
                cost = row['close'] * signal['size']
                commission_cost = cost * commission
                
                if capital >= cost + commission_cost:
                    shares_bought = signal['size']
                    capital -= (cost + commission_cost)
                    shares += shares_bought
                    
                    trades.append({
                        'timestamp': idx,
                        'action': 'BUY',
                        'price': row['close'],
                        'shares': shares_bought,
                        'capital': capital
                    })
            
            elif signal['action'] == 'SELL' and shares > 0:
                # Sell shares
                shares_to_sell = min(shares, signal['size'])
                proceeds = row['close'] * shares_to_sell
                commission_cost = proceeds * commission
                
                capital += (proceeds - commission_cost)
                shares -= shares_to_sell
                
                trades.append({
                    'timestamp': idx,
                    'action': 'SELL',
                    'price': row['close'],
                    'shares': shares_to_sell,
                    'capital': capital
                })
            
            # Record equity
            equity = capital + shares * row['close']
            equity_curve.append(equity)
        
        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'final_shares': shares,
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
            'trades': trades,
            'num_trades': len(trades),
            'equity_curve': equity_curve,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': self._max_drawdown(equity_curve)
        }
        
        return results
    
    def _max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)


class LiveTradingSystem(TradingChannelSystem):
    """
    Live trading system for real-time data with advanced adaptive components
    
    Now enhanced with topology-aware thresholds, multi-scale regime detection, and intelligent feature scoring
    
    Examples
    --------
    >>> system = LiveTradingSystem()
    >>> system.fit(historical_data)
    >>> 
    >>> # In live trading loop
    >>> for bar in live_data_stream:
    ...     signal = system.process_bar(bar)
    ...     if signal['action'] != 'HOLD':
    ...         execute_order(signal)
    """
    
    def __init__(self, strategy: str = 'simple', risk_per_trade: float = 0.02, 
                 use_advanced_adaptive: bool = True):
        """
        Parameters
        ----------
        strategy : str
            Strategy type
        risk_per_trade : float
            Maximum risk per trade as fraction of capital
        use_advanced_adaptive : bool
            Whether to use advanced adaptive components
        """
        super().__init__(strategy=strategy, use_advanced_adaptive=use_advanced_adaptive)
        self.risk_per_trade = risk_per_trade
        self.live_positions = {}
    
    def process_bar_live(self, bar: Dict, current_capital: float) -> Dict:
        """
        Process bar in live trading with risk management and regime awareness
        
        Parameters
        ----------
        bar : dict
            Real-time bar data
        current_capital : float
            Current account capital
            
        Returns
        -------
        signal : dict
            Trading signal with position sizing and regime awareness
        """
        # Get base signal with regime awareness
        signal = self.process_bar(pd.Series(bar))
        
        # Apply risk management with regime awareness
        if signal['action'] == 'BUY':
            # Calculate position size based on risk
            max_risk_amount = current_capital * self.risk_per_trade
            
            # Adjust risk based on regime if using advanced adaptive
            if self.use_advanced_adaptive:
                regime_info = self.encoder.get_regime_info()
                current_regime = regime_info.get('current_regime', 'UNKNOWN')
                regime_confidence = regime_info.get('regime_confidence', 0.5)
                
                # Adjust risk based on regime
                if current_regime == 'VOLATILE':
                    max_risk_amount *= 0.5  # Reduce risk in volatile markets
                elif current_regime == 'STABLE':
                    max_risk_amount *= 1.2  # Increase risk in stable markets
                
                # Adjust based on regime confidence
                max_risk_amount *= regime_confidence
            
            # Calculate position size
            shares = max_risk_amount / bar['close']
            signal['size'] = shares
            
            # Add regime information to signal
            if self.use_advanced_adaptive:
                signal['regime_info'] = self.encoder.get_regime_info()
        
        return signal
