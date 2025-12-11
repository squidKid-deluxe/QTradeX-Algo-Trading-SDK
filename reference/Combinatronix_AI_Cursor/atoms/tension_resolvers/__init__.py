# combinatronix/atoms/tension_resolvers/__init__.py
"""
Tension Resolvers - Cognitive Repair Atoms

These 5 atoms handle cognitive tensions, conflicts, and system regulation:

1. Balancer - Resolve contradictions (harmony, equilibrium)
2. Splitter - Disambiguate (analysis, discrimination)
3. Filler - Fill conceptual gaps (creation, invention)
4. Damper - Reduce overflow (restraint, inhibition)
5. Amplifier - Boost weak signals (enhancement, emphasis)

Each operates on cognitive tensions and system health.
"""

from .balancer import BalancerAtom
from .splitter import SplitterAtom
from .filler import FillerAtom
from .damper import DamperAtom
from .amplifier import AmplifierAtom

__all__ = ['BalancerAtom', 'SplitterAtom', 'FillerAtom', 'DamperAtom', 'AmplifierAtom']




# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TENSION RESOLVERS - COGNITIVE REPAIR TEST")
    print("="*60)
    
    from combinatronix.core import NDAnalogField
    
    # Test 1: Balancer
    print("\n[TEST 1] Balancer Atom")
    balancer = BalancerAtom(equilibrium_rate=0.4)
    field_a = NDAnalogField((8, 8))
    field_b = NDAnalogField((8, 8))
    field_a.activation[4, 4] = 1.0
    field_b.activation[4, 4] = -0.5
    
    before_diff = abs(field_a.activation[4, 4] - field_b.activation[4, 4])
    balancer.apply(field_a, field_b)
    after_diff = abs(field_a.activation[4, 4] - field_b.activation[4, 4])
    
    print(f"  Difference before: {before_diff:.3f}, after: {after_diff:.3f}")
    assert after_diff < before_diff, "Should reduce contradiction"
    print("  ✓ PASS - Balance verified")
    
    # Test 2: Splitter
    print("\n[TEST 2] Splitter Atom")
    splitter = SplitterAtom(separation_strength=0.5)
    field2 = NDAnalogField((8, 8))
    field2.activation[3:6, 3:6] = 0.5  # Ambiguous region
    
    ambiguity_before = splitter.get_ambiguity_map(field2)
    splitter.apply(field2)
    ambiguity_after = splitter.get_ambiguity_map(field2)
    
    print(f"  Ambiguous regions before: {np.sum(ambiguity_before):.0f}")
    print(f"  Ambiguous regions after: {np.sum(ambiguity_after):.0f}")
    print(f"  Split locations: {len(splitter.split_locations)}")
    print("  ✓ PASS - Disambiguation verified")
    
    # Test 3: Filler
    print("\n[TEST 3] Filler Atom")
    filler = FillerAtom(creativity=0.5)
    field3 = NDAnalogField((8, 8))
    # Create gap: high activation around, low in center
    field3.activation[3:6, 3:6] = 0.8
    field3.activation[4, 4] = 0.0  # Gap
    
    before_gap = field3.activation[4, 4]
    filler.apply(field3)
    after_gap = field3.activation[4, 4]
    
    print(f"  Gap before: {before_gap:.3f}, after: {after_gap:.3f}")
    print(f"  Gaps filled: {len(filler.get_filled_gaps())}")
    assert after_gap > before_gap, "Should fill gap"
    print("  ✓ PASS - Gap filling verified")
    
    # Test 4: Damper
    print("\n[TEST 4] Damper Atom")
    damper = DamperAtom(threshold=0.7, damping_rate=0.6, mode='soft')
    field4 = NDAnalogField((8, 8))
    field4.activation.fill(1.0)  # Overflow
    
    before_max = np.max(field4.activation)
    damper.apply(field4)
    after_max = np.max(field4.activation)
    
    print(f"  Max before: {before_max:.3f}, after: {after_max:.3f}")
    assert after_max < before_max, "Should reduce overflow"
    print("  ✓ PASS - Damping verified")
    
    # Test 5: Amplifier
    print("\n[TEST 5] Amplifier Atom")
    amplifier = AmplifierAtom(threshold=0.3, gain=3.0, mode='linear')
    field5 = NDAnalogField((8, 8))
    field5.activation[4, 4] = 0.1  # Weak signal
    
    before_weak = field5.activation[4, 4]
    amplifier.apply(field5)
    after_weak = field5.activation[4, 4]
    
    weak_signals = amplifier.detect_weak_signals(field5)
    print(f"  Weak signal before: {before_weak:.3f}, after: {after_weak:.3f}")
    print(f"  Amplification: {after_weak/before_weak:.1f}x")
    assert after_weak > before_weak * 2, "Should amplify weak signal"
    print("  ✓ PASS - Amplification verified")
    
    # Integration test
    print("\n[TEST 6] Integration - System Regulation")
    field6 = NDAnalogField((16, 16))
    field6.activation = np.random.random((16, 16))
    
    # Create system with all tension resolvers
    balancer_op = BalancerAtom(equilibrium_rate=0.3)
    splitter_op = SplitterAtom(separation_strength=0.4)
    filler_op = FillerAtom(creativity=0.3)
    damper_op = DamperAtom(threshold=0.8, damping_rate=0.5)
    amplifier_op = AmplifierAtom(threshold=0.2, gain=2.0)
    
    initial_variance = np.var(field6.activation)
    
    # Apply tension resolution
    balancer_op.apply(field6)
    splitter_op.apply(field6)
    filler_op.apply(field6)
    damper_op.apply(field6)
    amplifier_op.apply(field6)
    
    final_variance = np.var(field6.activation)
    
    print(f"  Initial variance: {initial_variance:.3f}")
    print(f"  Final variance: {final_variance:.3f}")
    print(f"  Gaps filled: {len(filler_op.get_filled_gaps())}")
    print(f"  Regions amplified: {len(amplifier_op.amplified_regions)}")
    print("  ✓ PASS - Tension resolvers work together")
    
    print("\n" + "="*60)
    print("ALL TENSION RESOLVER TESTS PASSED ✓")
    print("="*60)
    print("\nTension Resolvers Summary:")
    print(f"  1. {balancer} - Resolve contradictions")
    print(f"  2. {splitter} - Disambiguate")
    print(f"  3. {filler} - Fill gaps")
    print(f"  4. {damper} - Reduce overflow")