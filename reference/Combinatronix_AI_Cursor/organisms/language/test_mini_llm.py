# ============================================================================
# MiniLLM Test Suite
# ============================================================================

"""
Test suite for MiniLLM organism
"""

import numpy as np
import unittest

try:
    from .mini_llm import MiniLLM, WordEcho
except ImportError:
    from combinatronix.organisms.language.mini_llm import MiniLLM, WordEcho


class TestMiniLLM(unittest.TestCase):
    """Test MiniLLM organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.llm = MiniLLM({'enable_visualization': False})
    
    def test_initialization(self):
        """Test LLM initialization"""
        self.assertIsNotNone(self.llm.atoms)
        self.assertIsNotNone(self.llm.molecules)
        self.assertIsNotNone(self.llm.fields)
        self.assertEqual(len(self.llm.atoms), 6)
        self.assertEqual(len(self.llm.molecules), 3)
        self.assertGreater(len(self.llm.vocabulary), 0)
        self.assertIn('articles', self.llm.word_categories)
        self.assertIn('nouns', self.llm.word_categories)
    
    def test_process_word(self):
        """Test word processing"""
        # Test valid word
        result = self.llm.process_word("cat")
        self.assertTrue(result)
        self.assertEqual(len(self.llm.state['word_echoes']), 1)
        self.assertEqual(self.llm.state['word_echoes'][0].word, "cat")
        
        # Test invalid word
        result = self.llm.process_word("unknown")
        self.assertFalse(result)
        self.assertEqual(len(self.llm.state['word_echoes']), 1)  # Still 1 from previous
    
    def test_word_patterns(self):
        """Test word pattern generation"""
        # Test that patterns are created for all words
        self.assertEqual(len(self.llm.word_patterns), len(self.llm.vocabulary))
        
        # Test that patterns are unique
        patterns = list(self.llm.word_patterns.values())
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns):
                if i != j:
                    # Patterns should be different
                    self.assertFalse(np.array_equal(pattern1.activation, pattern2.activation))
    
    def test_grammar_patterns(self):
        """Test grammar pattern creation"""
        self.assertIn('article_noun', self.llm.grammar_patterns)
        self.assertIn('adj_noun', self.llm.grammar_patterns)
        self.assertIn('noun_verb', self.llm.grammar_patterns)
        self.assertIn('verb_adv', self.llm.grammar_patterns)
        
        # Test that grammar patterns have correct shapes
        for pattern in self.llm.grammar_patterns.values():
            self.assertEqual(pattern.shape, self.llm.config['field_size'])
    
    def test_predict_next_word(self):
        """Test next word prediction"""
        # Test with no context
        word, confidence = self.llm.predict_next_word()
        self.assertIn(word, self.llm.vocabulary)
        self.assertGreaterEqual(confidence, 0)
        
        # Test with context
        self.llm.process_word("the")
        word, confidence = self.llm.predict_next_word()
        self.assertIn(word, self.llm.vocabulary)
        self.assertGreaterEqual(confidence, 0)
    
    def test_generate_sentence(self):
        """Test sentence generation"""
        sentence = self.llm.generate_sentence(max_length=5, temperature=0.5)
        
        self.assertIsInstance(sentence, list)
        self.assertGreater(len(sentence), 0)
        self.assertLessEqual(len(sentence), 5)
        
        # All words should be in vocabulary
        for word in sentence:
            self.assertIn(word, self.llm.vocabulary)
    
    def test_seeded_generation(self):
        """Test generation with seed word"""
        seed_word = "cat"
        sentence = self.llm.generate_sentence(seed_word=seed_word, max_length=4)
        
        self.assertEqual(sentence[0], seed_word)
        self.assertGreater(len(sentence), 1)
    
    def test_context_evolution(self):
        """Test context evolution over time"""
        # Process some words
        self.llm.process_word("the")
        self.llm.process_word("big")
        
        initial_echoes = len(self.llm.state['word_echoes'])
        initial_context = np.sum(self.llm.fields['context_field'].activation)
        
        # Let it evolve
        for _ in range(5):
            self.llm.tick()
        
        # Should have some evolution
        final_echoes = len(self.llm.state['word_echoes'])
        final_context = np.sum(self.llm.fields['context_field'].activation)
        
        # Context should decay
        self.assertLess(final_context, initial_context)
    
    def test_echo_fading(self):
        """Test that echoes fade over time"""
        # Process a word
        self.llm.process_word("cat")
        initial_strength = self.llm.state['word_echoes'][0].strength
        
        # Let it fade
        for _ in range(50):  # Many ticks to ensure fading
            self.llm.tick()
        
        if self.llm.state['word_echoes']:
            final_strength = self.llm.state['word_echoes'][0].strength
            self.assertLess(final_strength, initial_strength)
    
    def test_grammar_bonus(self):
        """Test grammar bonus computation"""
        # Test article -> noun bonus
        self.llm.process_word("the")
        bonus = self.llm._get_grammar_bonus("cat")
        self.assertGreater(bonus, 0)
        
        # Test adjective -> noun bonus
        self.llm.reset_context()
        self.llm.process_word("big")
        bonus = self.llm._get_grammar_bonus("cat")
        self.assertGreater(bonus, 0)
        
        # Test noun -> verb bonus
        self.llm.reset_context()
        self.llm.process_word("cat")
        bonus = self.llm._get_grammar_bonus("runs")
        self.assertGreater(bonus, 0)
    
    def test_word_resonances(self):
        """Test word resonance computation"""
        # Build some context
        self.llm.process_word("the")
        self.llm.process_word("big")
        
        # Create expectation field
        expectation_field = self.llm._create_expectation_field()
        
        # Compute resonances
        resonances = self.llm._compute_word_resonances(expectation_field)
        
        # Should have resonances for all words
        self.assertEqual(len(resonances), len(self.llm.vocabulary))
        
        # All resonances should be non-negative
        for word, resonance in resonances.items():
            self.assertGreaterEqual(resonance, 0)
    
    def test_temperature_sampling(self):
        """Test temperature-based word selection"""
        word_scores = {"cat": 0.8, "dog": 0.6, "bird": 0.4}
        
        # Test greedy selection (temperature = 0)
        word, confidence = self.llm._select_word_with_temperature(word_scores, 0.0)
        self.assertEqual(word, "cat")  # Highest score
        self.assertEqual(confidence, 0.8)
        
        # Test temperature sampling
        word, confidence = self.llm._select_word_with_temperature(word_scores, 1.0)
        self.assertIn(word, word_scores.keys())
        self.assertGreaterEqual(confidence, 0)
    
    def test_get_state_summary(self):
        """Test state summary generation"""
        # Process some words
        self.llm.process_word("the")
        self.llm.process_word("cat")
        
        summary = self.llm.get_state_summary()
        
        self.assertIn('active_echoes', summary)
        self.assertIn('recent_words', summary)
        self.assertIn('context_energy', summary)
        self.assertIn('vocabulary_size', summary)
        self.assertEqual(summary['active_echoes'], 2)
        self.assertEqual(summary['vocabulary_size'], len(self.llm.vocabulary))
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.llm.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset_context(self):
        """Test context reset"""
        # Process some words
        self.llm.process_word("the")
        self.llm.process_word("cat")
        
        # Reset
        self.llm.reset_context()
        
        # Should be empty
        self.assertEqual(len(self.llm.state['word_echoes']), 0)
        self.assertEqual(len(self.llm.state['sequence_memory']), 0)
        self.assertEqual(self.llm.state['position_counter'], 0)
    
    def test_export_model(self):
        """Test model export"""
        # Process some words
        self.llm.process_word("the")
        self.llm.process_word("cat")
        
        # Export should complete without error
        try:
            self.llm.export_model("test_mini_llm.json")
            # Clean up
            import os
            if os.path.exists("test_mini_llm.json"):
                os.remove("test_mini_llm.json")
        except Exception as e:
            self.fail(f"Export failed with error: {e}")
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_size': (6, 6), 'max_echoes': 5},
            {'field_size': (10, 10), 'max_echoes': 8},
            {'temperature': 0.3, 'prediction_threshold': 0.2}
        ]
        
        for config in configs:
            llm = MiniLLM(config)
            
            # Should work with different configs
            result = llm.process_word("cat")
            self.assertTrue(result)
            
            sentence = llm.generate_sentence(max_length=3)
            self.assertIsInstance(sentence, list)
            self.assertGreater(len(sentence), 0)
    
    def test_grammar_boost(self):
        """Test grammar boost application"""
        # Build context
        self.llm.process_word("the")
        
        # Create expectation field
        expectation_field = self.llm._create_expectation_field()
        
        # Apply grammar boost
        boosted_field = self.llm._apply_grammar_boost(expectation_field)
        
        # Should be different from original
        self.assertFalse(np.array_equal(expectation_field.activation, boosted_field.activation))
    
    def test_molecular_processing(self):
        """Test molecular processing of words"""
        # Process a word
        self.llm.process_word("cat")
        
        echo = self.llm.state['word_echoes'][0]
        original_pattern = echo.pattern.activation.copy()
        
        # Process word molecularly
        processed = self.llm._process_word_molecular(echo.pattern)
        
        # Should be different due to molecular processing
        self.assertFalse(np.array_equal(processed.activation, original_pattern))
    
    def test_context_field_update(self):
        """Test context field updates"""
        initial_context = np.sum(self.llm.fields['context_field'].activation)
        
        # Process a word
        self.llm.process_word("cat")
        
        final_context = np.sum(self.llm.fields['context_field'].activation)
        
        # Context should have changed
        self.assertNotEqual(final_context, initial_context)
    
    def test_echo_field_update(self):
        """Test echo field updates"""
        # Process some words
        self.llm.process_word("the")
        self.llm.process_word("cat")
        
        # Update echo field
        self.llm._update_echo_field()
        
        # Echo field should have content
        echo_energy = np.sum(self.llm.fields['echo_field'].activation)
        self.assertGreater(echo_energy, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

