"""
Production Trading System - Phase 4

Integrates all phases with real exchange connectivity:
- Phase 1: Foundation (actors, consensus)
- Phase 2: Navigation (pathfinding, constraints)
- Phase 3: Dynamics (fields, learning)
- Phase 4: Production (exchange, monitoring)
"""

import time
import json
from dataclasses import dataclass
from typing import Dict, Optional

# Phase 1-3 components
from core.multi_scale_state import MultiScaleState, MarketTick, TimeScale
from actors.trading_actors import create_12_actors
from geometry.dodecahedron import DodecahedralSurface, ActorConsensus
from topology.persistent_homology import MarketRegimeClassifier
from navigation.state_space import StateSpaceGraph, PortfolioState
from navigation.pathfinder import RealTimePathPlanner
from navigation.ligature import LigatureConstraints
from dynamics.field_engine import FieldDynamicsEngine
from dynamics.cobordism_library import CobordismLibrary, Cobordism
from optimization.performance import OptimizationManager

# Phase 4 components
from exchange import (
    ExchangeAPI, 
    DataStream, 
    StreamConfig,
    OrderManagementSystem, 
    OrderRequest,
    OrderSide, 
    OrderType
)
import numpy as np

# Utilities
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class TradingConfig:
    """Production trading configuration"""
    # Exchange
    symbol: str = "BTCUSDT"
    exchange_name: str = "binance"
    testnet: bool = True
    
    # Trading parameters
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of capital per trade
    min_order_size: float = 0.001  # Minimum order size
    
    # Risk management
    max_drawdown: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    
    # System parameters
    update_interval_ms: int = 1000  # 1 second updates
    regime_check_interval: int = 10  # Check regime every 10 ticks
    
    # Performance
    enable_profiling: bool = True
    enable_learning: bool = True
    enable_field_dynamics: bool = True


class ProductionTradingSystem:
    """
    Complete production trading system
    
    Connects all phases to real exchange
    """
    
    def __init__(self, 
                 exchange: ExchangeAPI,
                 data_stream: DataStream,
                 config: TradingConfig):
        """
        Args:
            exchange: Connected exchange API
            data_stream: Real-time data stream
            config: Trading configuration
        """
        self.config = config
        self.exchange = exchange
        self.data_stream = data_stream
        
        # Core system components (Phases 1-3)
        self.market_state = MultiScaleState(symbol=config.symbol, cash=config.initial_capital)
        self.surface = DodecahedralSurface()
        self.actors = create_12_actors(self.surface.vertices)
        self.regime_classifier = MarketRegimeClassifier()
        self.state_graph = StateSpaceGraph(
            max_investment_pos=100,
            max_day_pos=30,
            max_scalp_pos=10,
            position_quantum=10
        )
        self.path_planner = RealTimePathPlanner(self.state_graph)
        self.constraints = LigatureConstraints()
        self.field_engine = FieldDynamicsEngine() if config.enable_field_dynamics else None
        self.cobordism_library = CobordismLibrary() if config.enable_learning else None
        self.optimizer = OptimizationManager() if config.enable_profiling else None
        
        # Production components (Phase 4)
        self.oms = OrderManagementSystem(exchange)
        
        # State
        self.running = False
        self.tick_count = 0
        self.last_regime_check = 0
        self.current_regime = "unknown"
        
        # Statistics
        self.decisions_made = 0
        self.orders_placed = 0
        self.trades_executed = 0
        
        logger.info(f"Production Trading System initialized for {config.symbol}")
        logger.info(f"  Exchange: {config.exchange_name} ({'testnet' if config.testnet else 'LIVE'})")
        logger.info(f"  Initial capital: ${config.initial_capital:,.2f}")
    
    def start(self):
        """Start the production system"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("\nStarting production trading system...")
        
        # Start OMS
        self.oms.start()
        
        # Set up data stream callback
        self.data_stream.callback = self.on_tick
        
        # Start data stream
        self.data_stream.start()
        
        self.running = True
        logger.info("✓ System started - trading live!\n")
    
    def stop(self):
        """Stop the production system"""
        if not self.running:
            return
        
        logger.info("\nStopping production trading system...")
        
        self.running = False
        
        # Stop data stream
        self.data_stream.stop()
        
        # Stop OMS
        self.oms.stop()
        
        # Close all positions (optional)
        # self._close_all_positions()
        
        logger.info("✓ System stopped")
        
        # Print final statistics
        self._print_statistics()
    
    def on_tick(self, tick: MarketTick):
        """
        Main callback for each market tick
        
        This is where the magic happens!
        """
        if not self.running:
            return
        
        try:
            # Ingest tick into market state
            self.market_state.ingest_tick(tick)
            self.tick_count += 1
            
            # Evolve fields
            if self.field_engine and len(self.market_state.get_scale_state(TimeScale.SCALP).prices) > 1:
                prices = self.market_state.get_scale_state(TimeScale.SCALP).prices
                price_return = (prices[-1] - prices[-2]) / prices[-2]
                
                for scale in TimeScale:
                    self.field_engine.evolve_field(scale, price_return, tick.volume, 0)
                
                self.field_engine.cross_scale_coupling()
            
            # Check regime periodically
            if self.tick_count - self.last_regime_check >= self.config.regime_check_interval:
                regime_info = self._detect_regime()
                self.current_regime = regime_info['regime']
                self.last_regime_check = self.tick_count
            
            # Make trading decision every N ticks
            if self.tick_count % 5 == 0:  # Every 5 ticks
                self._trading_decision()
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}", exc_info=True)
    
    def _detect_regime(self) -> Dict:
        """Detect market regime"""
        day_state = self.market_state.get_scale_state(TimeScale.DAY_TRADE)
        
        if len(day_state.prices) < 30:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        return self.regime_classifier.classify_regime(day_state.prices)
    
    def _trading_decision(self):
        """Make a trading decision"""
        try:
            self.decisions_made += 1
            
            # Get actor recommendations
            recommendations = []
            for actor in self.actors:
                rec = actor.analyze(self.market_state)
                # Adjust for regime
                if self.current_regime == 'choppy' and actor.actor_type.value == 'trend_rider':
                    rec.confidence *= 0.5
                recommendations.append(rec)
            
            # Build consensus
            consensus = self._build_consensus(recommendations)
            
            # Get target state
            target_state = self._build_target_state(consensus)
            
            # Get current state
            current_state = self._get_current_state()
            
            # Plan path
            self.path_planner.update_goal(target_state, current_state)
            next_state = self.path_planner.get_next_action(current_state)
            
            if next_state is None:
                return  # Already at target
            
            # Check constraints
            is_allowed, violations = self.constraints.check_transition(
                current_state, next_state, self.market_state
            )
            
            if not is_allowed:
                logger.warning(f"Trade blocked by constraints: {len(violations)} violations")
                return
            
            # Execute transition
            self._execute_transition(current_state, next_state)
            
        except Exception as e:
            logger.error(f"Decision error: {e}", exc_info=True)
    
    def _build_consensus(self, recommendations) -> Dict:
        """Build consensus from actor recommendations"""
        consensus = {}
        
        for scale in TimeScale:
            scale_recs = [r for r in recommendations if r.scale == scale]
            if not scale_recs:
                continue
            
            values = np.array([r.target_position for r in scale_recs])
            confidences = np.array([r.confidence for r in scale_recs])
            
            if len(values) < 12:
                pad = 12 - len(values)
                values = np.concatenate([values, np.zeros(pad)])
                confidences = np.concatenate([confidences, np.zeros(pad)])
            
            aggregated = self.surface.weighted_aggregate(values, confidences, method='smooth')
            smoothed = self.surface.smooth_field(values)
            
            consensus[scale] = ActorConsensus(
                aggregated_value=aggregated,
                individual_values=values,
                confidences=confidences,
                smoothed_values=smoothed,
                method='smooth'
            )
        
        return consensus
    
    def _build_target_state(self, consensus: Dict) -> PortfolioState:
        """Build target portfolio state"""
        inv_target = 0
        day_target = 0
        scalp_target = 0
        
        if TimeScale.INVESTMENT in consensus:
            inv_target = int(consensus[TimeScale.INVESTMENT].aggregated_value / 10) * 10
        
        if TimeScale.DAY_TRADE in consensus:
            day_target = int(consensus[TimeScale.DAY_TRADE].aggregated_value / 10) * 10
        
        if TimeScale.SCALP in consensus:
            scalp_target = int(consensus[TimeScale.SCALP].aggregated_value / 10) * 10
        
        target = self.state_graph.find_closest_state((inv_target, day_target, scalp_target))
        return target
    
    def _get_current_state(self) -> PortfolioState:
        """Get current portfolio state from exchange"""
        # Get position from OMS
        position = self.oms.get_position(self.config.symbol)
        
        if position:
            quantity = position.quantity
        else:
            quantity = 0.0
        
        # Quantize
        quantized = int(quantity / 10) * 10
        return PortfolioState(quantized, 0, 0)
    
    def _execute_transition(self, current: PortfolioState, next: PortfolioState):
        """Execute state transition via order"""
        # Calculate required trade
        current_quantity = current.total_position()
        target_quantity = next.total_position()
        delta = target_quantity - current_quantity
        
        if abs(delta) < self.config.min_order_size:
            return  # Too small to trade
        
        # Determine side
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        quantity = abs(delta)
        
        # Create order request
        request = OrderRequest(
            symbol=self.config.symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reason=f"Transition {current} -> {next}"
        )
        
        # Submit order
        order_id = self.oms.submit_order(request)
        
        if order_id:
            self.orders_placed += 1
            logger.info(f"Order placed: {side.value} {quantity} {self.config.symbol}")
            
            # Record cobordism
            if self.cobordism_library:
                cobordism = Cobordism(
                    boundary_in=current,
                    boundary_out=next,
                    interior=[current, next],
                    cost=quantity * 0.001,  # Estimated
                    duration=1,
                    slippage=0.0,
                    market_impact=0.0,
                    regime=self.current_regime,
                    volatility=0.01,
                    timestamp=float(self.tick_count),
                    success=True
                )
                self.cobordism_library.add_cobordism(cobordism)
    
    def _print_statistics(self):
        """Print system statistics"""
        logger.info("\n" + "="*70)
        logger.info("PRODUCTION SYSTEM STATISTICS")
        logger.info("="*70)
        
        logger.info(f"\nTrading:")
        logger.info(f"  Ticks processed: {self.tick_count}")
        logger.info(f"  Decisions made: {self.decisions_made}")
        logger.info(f"  Orders placed: {self.orders_placed}")
        
        # OMS stats
        oms_stats = self.oms.get_statistics()
        logger.info(f"\nOrder Management:")
        for key, value in oms_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Positions
        positions = self.oms.get_all_positions()
        logger.info(f"\nPositions:")
        for symbol, position in positions.items():
            logger.info(f"  {symbol}: {position.quantity:.4f} @ ${position.avg_entry_price:.2f}")
        
        # Cobordisms
        if self.cobordism_library:
            cob_stats = self.cobordism_library.get_statistics()
            logger.info(f"\nLearning (Cobordisms):")
            for key, value in cob_stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("")


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PRODUCTION TRADING SYSTEM - PHASE 4")
    print("="*70)
    print()
    print("This is the production system integrating all phases.")
    print("Requires actual exchange API keys to run.")
    print()
    print("Usage pattern:")
    print("""
from exchange import BinanceAPI, BinanceDataStream, StreamConfig

# Configure
config = TradingConfig(
    symbol="BTCUSDT",
    exchange_name="binance",
    testnet=True,  # ALWAYS start with testnet!
    initial_capital=10000.0
)

# Initialize exchange
exchange = BinanceAPI(api_key, api_secret, testnet=True)
if not exchange.connect():
    print("Failed to connect to exchange")
    exit(1)

# Initialize data stream
stream_config = StreamConfig(symbol="BTCUSDT", testnet=True)
data_stream = BinanceDataStream(stream_config.symbol, testnet=True, callback=None)

# Create system
system = ProductionTradingSystem(exchange, data_stream, config)

# Start trading!
system.start()

# Run for a while
time.sleep(300)  # 5 minutes

# Stop
system.stop()
    """)
    
    print("\n✓ Production trading system ready!")
    print("\nIMPORTANT: Always test on testnet first!")
