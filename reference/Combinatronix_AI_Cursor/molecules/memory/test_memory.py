# ============================================================================
# Memory Molecules Tests
# ============================================================================

"""
Comprehensive tests for memory molecules:
- ShortTermMemory
- LongTermMemory
- AssociativeMemory
- WorkingMemory
"""

import numpy as np
import unittest
from unittest.mock import Mock

try:
    from ...core import NDAnalogField
    from .short_term_memory import ShortTermMemory
    from .long_term_memory import LongTermMemory
    from .associative_memory import AssociativeMemory
    from .working_memory import WorkingMemory
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.molecules.memory.short_term_memory import ShortTermMemory
    from combinatronix.molecules.memory.long_term_memory import LongTermMemory
    from combinatronix.molecules.memory.associative_memory import AssociativeMemory
    from combinatronix.molecules.memory.working_memory import WorkingMemory


class TestShortTermMemory(unittest.TestCase):
    """Test ShortTermMemory molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((6, 6))
        self.stm = ShortTermMemory(capacity=3, decay_rate=0.9)
    
    def test_initialization(self):
        """Test STM initialization"""
        self.assertEqual(self.stm.capacity, 3)
        self.assertEqual(self.stm.echo.decay_rate, 0.9)
        self.assertEqual(len(self.stm.memory_items), 0)
    
    def test_store_and_recall(self):
        """Test storing and recalling items"""
        # Store first item
        self.field.activation[2:4, 2:4] = 1.0
        result1 = self.stm.store(self.field)
        
        self.assertEqual(len(self.stm.memory_items), 1)
        self.assertEqual(self.stm.total_stored, 1)
        
        # Store second item
        self.field.activation[1:3, 1:3] = 0.8
        result2 = self.stm.store(self.field)
        
        self.assertEqual(len(self.stm.memory_items), 2)
        
        # Recall most recent
        recalled = self.stm.recall()
        self.assertIsNotNone(recalled)
        self.assertEqual(recalled.shape, self.field.shape)
    
    def test_capacity_limit(self):
        """Test capacity limit enforcement"""
        # Fill beyond capacity
        for i in range(5):
            self.field.activation.fill(0)
            self.field.activation[i, i] = 1.0
            self.stm.store(self.field)
        
        # Should not exceed capacity
        self.assertEqual(len(self.stm.memory_items), 3)
        self.assertEqual(self.stm.capacity, 3)
    
    def test_memory_statistics(self):
        """Test memory statistics computation"""
        # Store some items
        for i in range(2):
            self.field.activation.fill(0)
            self.field.activation[i, i] = 1.0
            self.stm.store(self.field)
        
        stats = self.stm.get_memory_statistics()
        
        self.assertIn('item_count', stats)
        self.assertIn('capacity', stats)
        self.assertIn('utilization', stats)
        self.assertEqual(stats['item_count'], 2)
        self.assertEqual(stats['capacity'], 3)
    
    def test_similar_items(self):
        """Test finding similar items"""
        # Store a pattern
        self.field.activation[2:4, 2:4] = 1.0
        self.stm.store(self.field)
        
        # Create similar pattern
        similar_pattern = np.zeros((6, 6))
        similar_pattern[2:4, 2:4] = 0.9
        
        similar = self.stm.find_similar_items(similar_pattern, similarity_threshold=0.5)
        
        self.assertGreater(len(similar), 0)
        self.assertGreater(similar[0]['similarity'], 0.5)
    
    def test_consolidation(self):
        """Test memory consolidation"""
        # Store and access items
        self.field.activation[2:4, 2:4] = 1.0
        self.stm.store(self.field)
        self.stm.recall()  # Access the item
        
        original_energy = np.sum(np.abs(self.stm.memory_items[0]['activation']))
        self.stm.consolidate_memory(consolidation_factor=1.2)
        consolidated_energy = np.sum(np.abs(self.stm.memory_items[0]['activation']))
        
        self.assertGreater(consolidated_energy, original_energy)
    
    def test_reset(self):
        """Test STM reset"""
        self.field.activation[2:4, 2:4] = 1.0
        self.stm.store(self.field)
        
        self.stm.reset()
        
        self.assertEqual(len(self.stm.memory_items), 0)
        self.assertEqual(self.stm.total_stored, 0)


class TestLongTermMemory(unittest.TestCase):
    """Test LongTermMemory molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((6, 6))
        self.ltm = LongTermMemory(consolidation_threshold=0.8, importance_threshold=0.5)
    
    def test_initialization(self):
        """Test LTM initialization"""
        self.assertEqual(self.ltm.consolidation_threshold, 0.8)
        self.assertEqual(self.ltm.importance_threshold, 0.5)
        self.assertEqual(len(self.ltm.memories), 0)
    
    def test_store_memory(self):
        """Test storing memories"""
        self.field.activation[2:4, 2:4] = 1.0
        
        memory_id = self.ltm.store(self.field, importance=0.9, tags=["important"])
        
        self.assertIsNotNone(memory_id)
        self.assertEqual(len(self.ltm.memories), 1)
        self.assertIn("important", self.ltm.tags_index)
    
    def test_retrieve_by_id(self):
        """Test retrieving memory by ID"""
        self.field.activation[2:4, 2:4] = 1.0
        memory_id = self.ltm.store(self.field, importance=0.9)
        
        retrieved = self.ltm.retrieve(memory_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, self.field.shape)
        self.assertEqual(self.ltm.memories[memory_id]['access_count'], 1)
    
    def test_retrieve_by_tags(self):
        """Test retrieving memories by tags"""
        # Store memories with tags
        self.field.activation[2:4, 2:4] = 1.0
        self.ltm.store(self.field, importance=0.9, tags=["pattern", "important"])
        
        self.field.activation[1:3, 1:3] = 0.8
        self.ltm.store(self.field, importance=0.7, tags=["pattern"])
        
        # Retrieve by tags
        results = self.ltm.retrieve_by_tags(["pattern"])
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 10)  # Should respect max_results
    
    def test_consolidation(self):
        """Test memory consolidation"""
        # Store and access memory multiple times
        self.field.activation[2:4, 2:4] = 1.0
        memory_id = self.ltm.store(self.field, importance=0.9)
        
        # Access multiple times
        for _ in range(3):
            self.ltm.retrieve(memory_id)
        
        # Consolidate
        consolidated_count = self.ltm.consolidate_memories()
        
        self.assertGreater(consolidated_count, 0)
        self.assertGreater(self.ltm.consolidation_count, 0)
    
    def test_association(self):
        """Test creating associations between memories"""
        # Store two memories
        self.field.activation[2:4, 2:4] = 1.0
        memory_id_a = self.ltm.store(self.field, importance=0.8)
        
        self.field.activation[1:3, 1:3] = 0.8
        memory_id_b = self.ltm.store(self.field, importance=0.7)
        
        # Create association
        success = self.ltm.create_association(memory_id_a, memory_id_b, strength=0.9)
        
        self.assertTrue(success)
        self.assertIn('associations', self.ltm.memories[memory_id_a]['metadata'])
    
    def test_forget_old_memories(self):
        """Test forgetting old memories"""
        # Store memory
        self.field.activation[2:4, 2:4] = 1.0
        memory_id = self.ltm.store(self.field, importance=0.3)  # Low importance
        
        # Simulate time passing
        self.ltm.memory_counter += 150
        
        # Forget old memories
        forgotten_count = self.ltm.forget_old_memories(age_threshold=100)
        
        self.assertGreater(forgotten_count, 0)
        self.assertNotIn(memory_id, self.ltm.memories)
    
    def test_reset(self):
        """Test LTM reset"""
        self.field.activation[2:4, 2:4] = 1.0
        self.ltm.store(self.field, importance=0.9)
        
        self.ltm.reset()
        
        self.assertEqual(len(self.ltm.memories), 0)
        self.assertEqual(self.ltm.memory_counter, 0)


class TestAssociativeMemory(unittest.TestCase):
    """Test AssociativeMemory molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pattern_a = np.zeros((4, 4))
        self.pattern_a[1:3, 1:3] = 1.0
        
        self.pattern_b = np.zeros((4, 4))
        self.pattern_b[2:4, 2:4] = 0.8
        
        self.am = AssociativeMemory(binding_strength=0.8, resonance_threshold=0.6)
    
    def test_initialization(self):
        """Test AM initialization"""
        self.assertEqual(self.am.binder.binding_strength, 0.8)
        self.assertEqual(self.am.resonator.threshold, 0.6)
        self.assertEqual(len(self.am.patterns), 0)
    
    def test_store_pattern(self):
        """Test storing patterns"""
        pattern_id = self.am.store_pattern(self.pattern_a, name="test_pattern")
        
        self.assertIsNotNone(pattern_id)
        self.assertEqual(len(self.am.patterns), 1)
        self.assertEqual(self.am.patterns[pattern_id]['name'], "test_pattern")
    
    def test_create_association(self):
        """Test creating associations"""
        # Store patterns
        pattern_id_a = self.am.store_pattern(self.pattern_a)
        pattern_id_b = self.am.store_pattern(self.pattern_b)
        
        # Create association
        success = self.am.associate(pattern_id_a, pattern_id_b, strength=0.9)
        
        self.assertTrue(success)
        self.assertEqual(len(self.am.associations), 1)
    
    def test_retrieve_by_resonance(self):
        """Test retrieving patterns by resonance"""
        # Store pattern
        pattern_id = self.am.store_pattern(self.pattern_a)
        
        # Create query pattern (similar to stored)
        query_pattern = np.zeros((4, 4))
        query_pattern[1:3, 1:3] = 0.9
        
        # Retrieve by resonance
        results = self.am.retrieve_by_resonance(query_pattern)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], pattern_id)  # Should match stored pattern
        self.assertGreater(results[0][2], 0.6)  # Should exceed resonance threshold
    
    def test_pattern_completion(self):
        """Test pattern completion"""
        # Store complete pattern
        pattern_id = self.am.store_pattern(self.pattern_a)
        
        # Create partial pattern
        partial_pattern = np.zeros((4, 4))
        partial_pattern[1:2, 1:2] = 0.8  # Only part of the original
        
        # Complete pattern
        completed = self.am.pattern_completion(partial_pattern)
        
        self.assertIsNotNone(completed)
        self.assertEqual(completed.shape, partial_pattern.shape)
    
    def test_network_statistics(self):
        """Test network statistics"""
        # Store patterns and create associations
        pattern_id_a = self.am.store_pattern(self.pattern_a)
        pattern_id_b = self.am.store_pattern(self.pattern_b)
        self.am.associate(pattern_id_a, pattern_id_b)
        
        stats = self.am.get_network_statistics()
        
        self.assertEqual(stats['pattern_count'], 2)
        self.assertEqual(stats['association_count'], 1)
        self.assertGreater(stats['network_density'], 0)
    
    def test_strongly_connected_patterns(self):
        """Test finding strongly connected patterns"""
        # Store multiple patterns and create associations
        pattern_ids = []
        for i in range(3):
            pattern = np.zeros((4, 4))
            pattern[i:i+2, i:i+2] = 1.0
            pattern_ids.append(self.am.store_pattern(pattern))
        
        # Create associations
        self.am.associate(pattern_ids[0], pattern_ids[1])
        self.am.associate(pattern_ids[0], pattern_ids[2])
        self.am.associate(pattern_ids[1], pattern_ids[2])
        
        # Find strongly connected patterns
        connected = self.am.find_strongly_connected_patterns(min_connections=2)
        
        self.assertGreater(len(connected), 0)
    
    def test_reset(self):
        """Test AM reset"""
        self.am.store_pattern(self.pattern_a)
        self.am.store_pattern(self.pattern_b)
        
        self.am.reset()
        
        self.assertEqual(len(self.am.patterns), 0)
        self.assertEqual(len(self.am.associations), 0)


class TestWorkingMemory(unittest.TestCase):
    """Test WorkingMemory molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((6, 6))
        self.wm = WorkingMemory(capacity=3, focus_strength=0.3)
    
    def test_initialization(self):
        """Test WM initialization"""
        self.assertEqual(self.wm.capacity, 3)
        self.assertEqual(self.wm.attractor.strength, 0.3)
        self.assertEqual(len(self.wm.items), 0)
    
    def test_add_item(self):
        """Test adding items to working memory"""
        self.field.activation[2:4, 2:4] = 1.0
        
        item_id = self.wm.add_item(self.field, importance=0.9, focus_location=(3, 3))
        
        self.assertIsNotNone(item_id)
        self.assertEqual(len(self.wm.items), 1)
        self.assertEqual(self.wm.items[0]['importance'], 0.9)
    
    def test_update_focus(self):
        """Test updating focus location"""
        self.wm.update_focus((4, 4), radius=2.0)
        
        self.assertEqual(self.wm.focus_location, (4, 4))
        self.assertEqual(self.wm.focus_radius, 2.0)
        self.assertEqual(self.wm.attractor.location, (4, 4))
    
    def test_get_active_items(self):
        """Test getting active items"""
        # Add items with different importance
        self.field.activation[2:4, 2:4] = 1.0
        self.wm.add_item(self.field, importance=0.9, focus_location=(3, 3))
        
        self.field.activation[1:3, 1:3] = 0.8
        self.wm.add_item(self.field, importance=0.5, focus_location=(2, 2))
        
        # Get active items
        active = self.wm.get_active_items(min_importance=0.4)
        
        self.assertGreater(len(active), 0)
        self.assertGreater(active[0]['activity'], active[1]['activity'])  # Sorted by activity
    
    def test_retrieve_item(self):
        """Test retrieving specific item"""
        self.field.activation[2:4, 2:4] = 1.0
        item_id = self.wm.add_item(self.field, importance=0.9)
        
        retrieved = self.wm.retrieve_item(item_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, self.field.shape)
        self.assertEqual(self.wm.items[0]['access_count'], 1)
    
    def test_remove_item(self):
        """Test removing items"""
        self.field.activation[2:4, 2:4] = 1.0
        item_id = self.wm.add_item(self.field, importance=0.9)
        
        success = self.wm.remove_item(item_id)
        
        self.assertTrue(success)
        self.assertEqual(len(self.wm.items), 0)
    
    def test_memory_load(self):
        """Test memory load statistics"""
        # Add items
        for i in range(2):
            self.field.activation.fill(0)
            self.field.activation[i, i] = 1.0
            self.wm.add_item(self.field, importance=0.8)
        
        load = self.wm.get_memory_load()
        
        self.assertEqual(load['item_count'], 2)
        self.assertEqual(load['capacity'], 3)
        self.assertGreater(load['utilization'], 0)
    
    def test_focus_statistics(self):
        """Test focus statistics"""
        # Set focus
        self.wm.update_focus((3, 3), radius=2.0)
        
        # Add item at focus location
        self.field.activation[2:4, 2:4] = 1.0
        self.wm.add_item(self.field, importance=0.9, focus_location=(3, 3))
        
        stats = self.wm.get_focus_statistics()
        
        self.assertTrue(stats['focus_active'])
        self.assertEqual(stats['focus_location'], (3, 3))
        self.assertGreater(stats['focused_items'], 0)
    
    def test_consolidation(self):
        """Test consolidating important items"""
        # Add important item
        self.field.activation[2:4, 2:4] = 1.0
        self.wm.add_item(self.field, importance=0.9)
        
        original_importance = self.wm.items[0]['importance']
        consolidated_count = self.wm.consolidate_important_items(importance_threshold=0.8)
        
        self.assertGreater(consolidated_count, 0)
        self.assertGreater(self.wm.items[0]['importance'], original_importance)
    
    def test_working_field(self):
        """Test getting working field representation"""
        # Add items
        self.field.activation[2:4, 2:4] = 1.0
        self.wm.add_item(self.field, importance=0.9)
        
        working_field = self.wm.get_working_field()
        
        self.assertIsNotNone(working_field)
        self.assertEqual(working_field.shape, self.field.shape)
    
    def test_reset(self):
        """Test WM reset"""
        self.field.activation[2:4, 2:4] = 1.0
        self.wm.add_item(self.field, importance=0.9)
        
        self.wm.reset()
        
        self.assertEqual(len(self.wm.items), 0)
        self.assertIsNone(self.wm.focus_location)


class TestMemoryIntegration(unittest.TestCase):
    """Test integration between memory molecules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.stm = ShortTermMemory(capacity=3)
        self.ltm = LongTermMemory(importance_threshold=0.5)
        self.am = AssociativeMemory()
        self.wm = WorkingMemory(capacity=5)
    
    def test_memory_pipeline(self):
        """Test complete memory pipeline"""
        # Create pattern
        self.field.activation[3:5, 3:5] = 1.0
        
        # Store in STM
        self.stm.store(self.field)
        
        # Store important item in LTM
        memory_id = self.ltm.store(self.field, importance=0.9, tags=["important"])
        
        # Store pattern in AM
        pattern_id = self.am.store_pattern(self.field.activation, name="test")
        
        # Add to WM
        item_id = self.wm.add_item(self.field, importance=0.8)
        
        # All should complete without error
        self.assertIsNotNone(memory_id)
        self.assertIsNotNone(pattern_id)
        self.assertIsNotNone(item_id)
    
    def test_memory_transfer(self):
        """Test transferring from STM to LTM"""
        # Store in STM
        self.field.activation[2:4, 2:4] = 1.0
        self.stm.store(self.field)
        
        # Retrieve from STM
        stm_retrieved = self.stm.recall()
        
        # Store in LTM
        memory_id = self.ltm.store(stm_retrieved, importance=0.8)
        
        # Retrieve from LTM
        ltm_retrieved = self.ltm.retrieve(memory_id)
        
        # Should be similar
        self.assertIsNotNone(ltm_retrieved)
        self.assertEqual(ltm_retrieved.shape, self.field.shape)
    
    def test_associative_retrieval(self):
        """Test associative retrieval across memory systems"""
        # Store patterns in AM
        pattern_a = np.zeros((4, 4))
        pattern_a[1:3, 1:3] = 1.0
        pattern_id_a = self.am.store_pattern(pattern_a, name="pattern_a")
        
        pattern_b = np.zeros((4, 4))
        pattern_b[2:4, 2:4] = 0.8
        pattern_id_b = self.am.store_pattern(pattern_b, name="pattern_b")
        
        # Create association
        self.am.associate(pattern_id_a, pattern_id_b)
        
        # Retrieve by resonance
        query = np.zeros((4, 4))
        query[1:3, 1:3] = 0.9
        results = self.am.retrieve_by_resonance(query)
        
        self.assertGreater(len(results), 0)
        self.assertIn(pattern_id_a, [r[0] for r in results])


def create_test_field_with_pattern():
    """Helper function to create test field with pattern"""
    field = NDAnalogField((6, 6))
    field.activation[2:4, 2:4] = 1.0
    return field


def create_test_patterns():
    """Helper function to create test patterns"""
    pattern_a = np.zeros((4, 4))
    pattern_a[1:3, 1:3] = 1.0
    
    pattern_b = np.zeros((4, 4))
    pattern_b[2:4, 2:4] = 0.8
    
    return pattern_a, pattern_b


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

