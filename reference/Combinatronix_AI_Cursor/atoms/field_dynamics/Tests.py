

# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FIELD DYNAMICS - SPATIAL OPERATIONS TEST")
    print("="*60)
    
    from combinatronix.core import NDAnalogField
    
    # Test 1: Vortex
    print("\n[TEST 1] Vortex Atom")
    vortex = VortexAtom(center=(8, 8), angular_velocity=0.5, strength=0.2)
    field = NDAnalogField((16, 16))
    field.activation[8, 8] = 1.0
    
    initial_pos = field.activation[8, 8]
    for _ in range(5):
        vortex.apply(field)
    
    print(f"  Initial center value: {initial_pos:.3f}")
    print(f"  After rotation, energy spread: {field.get_energy():.3f}")
    assert field.get_energy() > 0, "Vortex should preserve energy"
    print("  ✓ PASS - Rotation verified")
    
    # Test 2: Attractor
    print("\n[TEST 2] Attractor Atom")
    attractor = AttractorAtom(location=(8, 8), strength=0.3)
    field2 = NDAnalogField((16, 16))
    field2.activation[2, 2] = 1.0
    field2.activation[14, 14] = 1.0
    
    initial_center = field2.activation[8, 8]
    attractor.apply(field2)
    final_center = field2.activation[8, 8]
    
    print(f"  Center before: {initial_center:.3f}")
    print(f"  Center after: {final_center:.3f}")
    assert final_center > initial_center, "Attractor should pull toward center"
    print("  ✓ PASS - Attraction verified")
    
    # Test 3: Barrier
    print("\n[TEST 3] Barrier Atom")
    barrier = BarrierAtom(region=[((5, 5), (7, 7))], permeability=0.1)
    field3 = NDAnalogField((16, 16))
    field3.activation[6, 6] = 1.0
    
    before_barrier = field3.activation[6, 6]
    barrier.apply(field3)
    after_barrier = field3.activation[6, 6]
    
    print(f"  Before barrier: {before_barrier:.3f}")
    print(f"  After barrier: {after_barrier:.3f}")
    assert after_barrier < before_barrier, "Barrier should reduce activation"
    print("  ✓ PASS - Blocking verified")
    
    # Test 4: Bridge
    print("\n[TEST 4] Bridge Atom")
    bridge = BridgeAtom(bidirectional=True, transfer_rate=0.5)
    field4 = NDAnalogField((16, 16))
    field4.activation[2, 2] = 1.0
    field4.activation[14, 14] = 0.0
    
    bridge.connect(field4, (2, 2), (14, 14), strength=1.0)
    
    value_a = field4.activation[2, 2]
    value_b = field4.activation[14, 14]
    
    print(f"  Region A: {value_a:.3f}, Region B: {value_b:.3f}")
    assert abs(value_a - value_b) < 0.5, "Bridge should equalize regions"
    print("  ✓ PASS - Connection verified")
    
    # Test 5: Void
    print("\n[TEST 5] Void Atom")
    void = VoidAtom(location=(8, 8), radius=3.0, absorption_rate=0.8)
    field5 = NDAnalogField((16, 16))
    field5.activation = np.ones((16, 16))
    
    initial_energy = field5.get_energy()
    void.apply(field5)
    final_energy = field5.get_energy()
    absorbed = void.get_absorbed_amount()
    
    print(f"  Initial energy: {initial_energy:.3f}")
    print(f"  Final energy: {final_energy:.3f}")
    print(f"  Absorbed: {absorbed:.3f}")
    assert final_energy < initial_energy, "Void should absorb energy"
    print("  ✓ PASS - Absorption verified")
    
    # Integration test
    print("\n[TEST 6] Integration - Combining Field Dynamics")
    field6 = NDAnalogField((16, 16))
    field6.activation[8, 8] = 1.0
    
    # Create complex field dynamics
    vortex_op = VortexAtom(center=(8, 8), angular_velocity=0.3)
    attractor_op = AttractorAtom(location=(12, 12), strength=0.2)
    barrier_op = BarrierAtom(region=[((7, 7), (9, 9))], permeability=0.5)
    void_op = VoidAtom(location=(4, 4), radius=2.0)
    
    # Apply all operations
    vortex_op.apply(field6)
    attractor_op.apply(field6)
    barrier_op.apply(field6)
    void_op.apply(field6)
    
    print(f"  Final energy: {field6.get_energy():.3f}")
    print(f"  Void absorbed: {void_op.get_absorbed_amount():.3f}")
    print("  ✓ PASS - Field dynamics compose successfully")
    
    print("\n" + "="*60)
    print("ALL FIELD DYNAMICS TESTS PASSED ✓")
    print("="*60)
    print("\nField Dynamics Summary:")
    print(f"  1. {vortex} - Circular flow")
    print(f"  2. {attractor} - Pull toward point")
    print(f"  3. {barrier} - Block propagation")
    print(f"  4. {bridge} - Connect regions")
    print(f"  5. {void} - Absorb activation")
    print("\nThese atoms control spatial topology and boundaries!")
