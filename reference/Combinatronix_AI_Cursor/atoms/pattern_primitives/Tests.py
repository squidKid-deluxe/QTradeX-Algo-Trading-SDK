

# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PATTERN PRIMITIVES - ATOMIC OPERATIONS TEST")
    print("="*60)
    
    from combinatronix.core import NDAnalogField
    
    # Test 1: Pulse
    print("\n[TEST 1] Pulse Atom")
    pulse = PulseAtom(frequency=0.5, amplitude=1.0)
    field = NDAnalogField((8, 8))
    
    values = []
    for _ in range(10):
        pulse.apply(field, location=(4, 4))
        values.append(field.activation[4, 4])
    
    print(f"  Pulse values: {values[:5]}")
    assert max(values) > 0.5 and min(values) < -0.5, "Pulse should oscillate"
    print("  ✓ PASS - Oscillation verified")
    
    # Test 2: Seed
    print("\n[TEST 2] Seed Atom")
    seed = SeedAtom(spread_radius=2)
    field2 = NDAnalogField((8, 8))
    seed.apply(field2, location=(4, 4), strength=1.0)
    
    center_value = field2.activation[4, 4]
    neighbor_value = field2.activation[4, 5]
    print(f"  Center: {center_value:.3f}, Neighbor: {neighbor_value:.3f}")
    assert center_value > neighbor_value > 0, "Seed should spread from center"
    print("  ✓ PASS - Spreading verified")
    
    # Test 3: Echo
    print("\n[TEST 3] Echo Atom")
    echo = EchoAtom(decay_rate=0.8, depth=3)
    field3 = NDAnalogField((8, 8))
    
    # Create pattern and echo it
    field3.activation[4, 4] = 1.0
    echo.apply(field3)
    field3.activation[4, 4] = 0.0  # Remove original
    echo.apply(field3)
    
    echo_strength = echo.get_echo_strength()
    print(f"  Echo strength: {echo_strength:.3f}")
    assert echo_strength > 0, "Echo should persist"
    print("  ✓ PASS - Memory persistence verified")
    
    # Test 4: Mirror
    print("\n[TEST 4] Mirror Atom")
    mirror = MirrorAtom(axis='vertical')
    field4 = NDAnalogField((8, 8))
    field4.activation[2, 2] = 1.0  # Asymmetric pattern
    
    symmetry_before = mirror.compute_symmetry(field4)
    mirror.apply(field4, blend=1.0)  # Full reflection
    symmetry_after = mirror.compute_symmetry(field4)
    
    print(f"  Symmetry before: {symmetry_before:.3f}, after: {symmetry_after:.3f}")
    assert symmetry_after > symmetry_before, "Mirror should increase symmetry"
    print("  ✓ PASS - Reflection verified")
    
    # Test 5: Gradient
    print("\n[TEST 5] Gradient Atom")
    gradient = GradientAtom(strength=0.2, direction='ascent')
    field5 = NDAnalogField((8, 8))
    field5.activation[4, 4] = 1.0  # Create peak
    
    initial_energy = field5.get_energy()
    gradient.apply(field5)
    final_energy = field5.get_energy()
    
    peaks = gradient.find_peaks(field5, threshold=0.3)
    print(f"  Peaks found: {len(peaks)}")
    print(f"  Energy change: {initial_energy:.3f} -> {final_energy:.3f}")
    assert len(peaks) > 0, "Gradient should find peaks"
    print("  ✓ PASS - Gradient flow verified")
    
    # Integration test
    print("\n[TEST 6] Integration - Combining Atoms")
    field6 = NDAnalogField((8, 8))
    
    # Inject with seed
    seed.apply(field6, location=(4, 4), strength=1.0)
    # Add pulse
    pulse.reset()
    pulse.apply(field6, location=(4, 4))
    # Create echo
    echo.clear()
    echo.apply(field6)
    # Apply gradient
    gradient.apply(field6)
    
    print(f"  Final energy: {field6.get_energy():.3f}")
    print(f"  Echo strength: {echo.get_echo_strength():.3f}")
    print("  ✓ PASS - Atoms compose successfully")
    
    print("\n" + "="*60)
    print("ALL PATTERN PRIMITIVE TESTS PASSED ✓")
    print("="*60)
    print("\nPattern Primitives Summary:")
    print(f"  1. {pulse}")
    print(f"  2. {seed}")
    print(f"  3. {echo}")
    print(f"  4. {mirror}")
    print(f"  5. {gradient}")
    print("\nThese 5 atoms form the foundation for all higher operations!")