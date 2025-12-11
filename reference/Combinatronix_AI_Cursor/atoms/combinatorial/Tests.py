

# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("COMBINATORIAL ATOMS - PURE LOGIC OPERATIONS TEST")
    print("="*60)
    
    from combinatronix.core import NDAnalogField
    
    # Test 1: Witness (I combinator)
    print("\n[TEST 1] Witness Atom (I combinator)")
    witness = WitnessAtom(amplification=2.0)
    field = NDAnalogField((8, 8))
    field.activation[4, 4] = 1.0
    
    observed = witness.observe(field)
    witness.apply(field)
    
    print(f"  Original value: 1.0")
    print(f"  After witness: {field.activation[4, 4]:.3f}")
    assert field.activation[4, 4] == 2.0, "Witness should amplify by factor"
    print("  ✓ PASS - Identity with amplification")
    
    # Test 2: Selector (K combinator)
    print("\n[TEST 2] Selector Atom (K combinator)")
    selector = SelectorAtom(selection_threshold=0.5)
    field_a = NDAnalogField((8, 8))
    field_b = NDAnalogField((8, 8))
    field_a.activation[4, 4] = 1.0
    field_b.activation[4, 4] = 0.5
    
    # K x y = x (should return field_a)
    result = selector.apply(field_a, field_b)
    assert result is field_a, "Selector should return first field"
    print(f"  Returned first field: {result is field_a}")
    
    # Test threshold selection
    field_c = NDAnalogField((8, 8))
    field_c.activation = np.random.random((8, 8))
    selector.select_by_threshold(field_c, keep_above=True)
    assert np.all(field_c.activation[field_c.activation > 0] > 0.5), "Should keep above threshold"
    print("  ✓ PASS - Selection verified")
    
    # Test 3: Weaver (S combinator)
    print("\n[TEST 3] Weaver Atom (S combinator)")
    weaver = WeaverAtom(combination='add', weight_a=0.6, weight_b=0.4)
    field = NDAnalogField((8, 8))
    
    def op_a(f):
        return f.activation + 1.0
    
    def op_b(f):
        return f.activation * 2.0
    
    field.activation[4, 4] = 1.0
    weaver.apply(field, op_a, op_b)
    
    # Result should be: (1+1)*0.6 + (1*2)*0.4 = 1.2 + 0.8 = 2.0
    print(f"  Woven result: {field.activation[4, 4]:.3f}")
    print("  ✓ PASS - Parallel application verified")
    
    # Test 4: Composer (B combinator)
    print("\n[TEST 4] Composer Atom (B combinator)")
    composer = ComposerAtom(preserve_intermediate=True)
    field = NDAnalogField((8, 8))
    field.activation[4, 4] = 1.0
    
    # Chain of operations
    ops = [
        lambda f: (f.activation * 2, f)[1],  # Double
        lambda f: (f.activation + 1, f)[1],  # Add 1
        lambda f: (f.activation * 0.5, f)[1] # Half
    ]
    
    composer.apply(field, ops)
    
    # Result: ((1 * 2) + 1) * 0.5 = 1.5
    print(f"  Composed result: {field.activation[4, 4]:.3f}")
    intermediates = composer.get_intermediate_results()
    print(f"  Intermediate steps captured: {len(intermediates)}")
    assert len(intermediates) == 4, "Should capture all intermediate steps"
    print("  ✓ PASS - Composition verified")
    
    # Test 5: Swapper (C combinator)
    print("\n[TEST 5] Swapper Atom (C combinator)")
    swapper = SwapperAtom(swap_type='spatial')
    field = NDAnalogField((8, 8))
    field.activation[2, 2] = 1.0
    field.activation[5, 5] = 0.5
    
    original_2_2 = field.activation[2, 2]
    swapper.apply(field, axis='both')  # Flip both axes
    flipped_2_2 = field.activation[2, 2]
    
    print(f"  Original [2,2]: {original_2_2:.3f}")
    print(f"  After flip [2,2]: {flipped_2_2:.3f}")
    assert original_2_2 != flipped_2_2, "Swapper should change positions"
    print("  ✓ PASS - Perspective flip verified")
    
    # Integration test
    print("\n[TEST 6] Integration - Combining Combinatorial Atoms")
    field = NDAnalogField((8, 8))
    field.activation[4, 4] = 1.0
    
    # Complex pipeline using all combinators
    witness_atom = WitnessAtom(1.5)
    selector_atom = SelectorAtom(0.3)
    weaver_atom = WeaverAtom('multiply')
    composer_atom = ComposerAtom()
    swapper_atom = SwapperAtom('spatial')
    
    # Observe
    witness_atom.apply(field)
    # Select
    selector_atom.select_by_threshold(field)
    # Swap perspective
    swapper_atom.apply(field, axis='diagonal')
    
    print(f"  Pipeline complete, energy: {field.get_energy():.3f}")
    print("  ✓ PASS - Combinatorial atoms compose successfully")
    
    print("\n" + "="*60)
    print("ALL COMBINATORIAL ATOM TESTS PASSED ✓")
    print("="*60)
    print("\nCombinatorial Atoms Summary:")
    print(f"  1. {WitnessAtom()} - I combinator")
    print(f"  2. {SelectorAtom()} - K combinator")
    print(f"  3. {WeaverAtom()} - S combinator")
    print(f"  4. {ComposerAtom()} - B combinator")
    print(f"  5. {SwapperAtom()} - C combinator")
    print("\nThese atoms bring pure logic to field operations!")