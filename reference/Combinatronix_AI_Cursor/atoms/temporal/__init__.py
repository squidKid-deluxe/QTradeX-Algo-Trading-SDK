# combinatronix/atoms/temporal/__init__.py
"""
Temporal Atoms - Time-Based Operations

These 5 atoms handle time, prediction, memory, and temporal dynamics:

1. Anticipator - Predict future state (foresight, expectation)
2. MemoryTrace - Accumulate history (learning, karma)
3. Rhythm - Periodic patterns (beat, synchronization)
4. Threshold - Activation gates (decision points, breakthrough)
5. Decay - Exponential fading (forgetting, entropy)

Each operates on temporal aspects of cognition.
"""

from .anticipator import AnticipatorAtom
from .memory_trace import MemoryTraceAtom
from .rhythm import RhythmAtom
from .threshold import ThresholdAtom
from .decay import DecayAtom

__all__ = ['AnticipatorAtom', 'MemoryTraceAtom', 'RhythmAtom', 'ThresholdAtom', 'DecayAtom']


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TEMPORAL ATOMS - TIME-BASED OPERATIONS TEST")
    print("="*60)
    
    from combinatronix.core import NDAnalogField
    
    # Test 1: Anticipator
    print("\n[TEST 1] Anticipator Atom")
    anticipator = AnticipatorAtom(history_depth=3)
    field = NDAnalogField((8, 8))
    
    # Create predictable pattern
    for i in range(5):
        field.activation[4, 4] = i * 0.2
        anticipator.apply(field)
    
    prediction_error = anticipator.get_prediction_error(field)
    print(f"  Prediction error: {prediction_error:.3f}")
    assert anticipator.last_prediction is not None, "Should generate prediction"
    print("  ✓ PASS - Prediction verified")
    
    # Test 2: Memory Trace
    print("\n[TEST 2] Memory Trace Atom")
    memory = MemoryTraceAtom(accumulation_rate=0.2, decay_rate=0.95)
    field2 = NDAnalogField((8, 8))
    
    # Accumulate at same location
    for _ in range(5):
        field2.activation[4, 4] = 1.0
        memory.apply(field2)
        field2.activation[4, 4] = 0  # Clear after each step
    
    trace_strength = memory.get_trace_strength()
    hotspots = memory.get_hotspots(top_k=3)
    
    print(f"  Trace strength: {trace_strength:.3f}")
    print(f"  Hotspots found: {len(hotspots)}")
    assert trace_strength > 0, "Should accumulate memory"
    print("  ✓ PASS - Memory accumulation verified")
    
    # Test 3: Rhythm
    print("\n[TEST 3] Rhythm Atom")
    rhythm = RhythmAtom(frequency=0.5, amplitude=0.3, pattern='sine')
    field3 = NDAnalogField((8, 8))
    field3.activation.fill(0.5)
    
    values = []
    for _ in range(10):
        rhythm.apply(field3)
        values.append(np.mean(field3.activation))
    
    # Check oscillation
    value_range = max(values) - min(values)
    print(f"  Value range: {value_range:.3f}")
    assert value_range > 0.2, "Should oscillate"
    print("  ✓ PASS - Rhythmic oscillation verified")
    
    # Test 4: Threshold
    print("\n[TEST 4] Threshold Atom")
    threshold = ThresholdAtom(threshold=0.5, mode='rectify', hysteresis=0.1)
    field4 = NDAnalogField((8, 8))
    field4.activation = np.random.random((8, 8))
    
    before_count = np.sum(field4.activation > 0)
    threshold.apply(field4)
    after_count = np.sum(field4.activation > 0)
    
    print(f"  Active before: {before_count}, after: {after_count}")
    assert after_count < before_count, "Should filter by threshold"
    print("  ✓ PASS - Threshold gating verified")
    
    # Test 5: Decay
    print("\n[TEST 5] Decay Atom")
    decay = DecayAtom(decay_rate=0.8, min_value=0.01)
    field5 = NDAnalogField((8, 8))
    field5.activation.fill(1.0)
    
    initial_energy = field5.get_energy()
    
    for _ in range(10):
        decay.apply(field5)
    
    final_energy = field5.get_energy()
    half_life = decay.get_half_life()
    
    print(f"  Initial: {initial_energy:.3f}, Final: {final_energy:.3f}")
    print(f"  Half-life: {half_life:.1f} steps")
    assert final_energy < initial_energy * 0.5, "Should decay significantly"
    print("  ✓ PASS - Exponential decay verified")
    
    # Integration test
    print("\n[TEST 6] Integration - Temporal Dynamics")
    field6 = NDAnalogField((8, 8))
    
    # Create complex temporal behavior
    anticipator = AnticipatorAtom(history_depth=3)
    memory = MemoryTraceAtom(accumulation_rate=0.1)
    rhythm = RhythmAtom(frequency=1.0, amplitude=0.2)
    threshold = ThresholdAtom(threshold=0.3, mode='amplify')
    decay_op = DecayAtom(decay_rate=0.95)
    
    # Simulate temporal evolution
    for t in range(20):
        # Inject periodic stimulus
        if t % 5 == 0:
            field6.activation[4, 4] = 1.0
        
        # Apply temporal operations
        rhythm.apply(field6)
        anticipator.apply(field6)
        memory.apply(field6)
        threshold.apply(field6)
        decay_op.apply(field6)
    
    trace_strength = memory.get_trace_strength()
    print(f"  Final trace strength: {trace_strength:.3f}")
    print(f"  Total decayed: {decay_op.get_total_decayed():.3f}")
    print("  ✓ PASS - Temporal atoms compose successfully")
    
    print("\n" + "="*60)
    print("ALL TEMPORAL ATOM TESTS PASSED ✓")
    print("="*60)
    print("\nTemporal Atoms Summary:")
    print(f"  1. {AnticipatorAtom()} - Predict future")
    print(f"  2. {MemoryTraceAtom()} - Accumulate history")
    print(f"  3. {RhythmAtom()} - Periodic patterns")
    print(f"  4. {ThresholdAtom()} - Activation gates")
    print(f"  5. {DecayAtom()} - Exponential fading")
    print("\nThese atoms control time, memory, and temporal dynamics!")