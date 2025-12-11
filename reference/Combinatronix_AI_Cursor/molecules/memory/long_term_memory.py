# ============================================================================
# LongTermMemory - Persistent Memory with Consolidation
# ============================================================================

"""
LongTermMemory - Persistent memory with consolidation and retrieval

Composition: MemoryTrace + Threshold + Binder
Category: Memory
Complexity: Molecule (50-200 lines)

Maintains long-term memory traces with consolidation, threshold-based
storage, and associative binding. This enables persistent storage of
important information, memory consolidation, and associative retrieval.

Example:
    >>> ltm = LongTermMemory(consolidation_threshold=0.8, binding_strength=0.9)
    >>> ltm.store(field, importance=0.9, tags=["important", "pattern"])
    >>> retrieved = ltm.retrieve_by_tags(["pattern"])
    >>> consolidated = ltm.consolidate_memories()
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from ...atoms.multi_field import BinderAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from combinatronix.atoms.multi_field import BinderAtom
    from combinatronix.core import NDAnalogField


class LongTermMemory:
    """Long-term memory with consolidation and associative retrieval"""
    
    def __init__(self, consolidation_threshold: float = 0.8, 
                 binding_strength: float = 0.9, decay_rate: float = 0.999,
                 importance_threshold: float = 0.5):
        """
        Args:
            consolidation_threshold: Threshold for memory consolidation
            binding_strength: Strength of associative bindings
            decay_rate: How fast memories decay (very slow for LTM)
            importance_threshold: Minimum importance to store
        """
        self.memory_trace = MemoryTraceAtom(
            accumulation_rate=0.1,  # Slow accumulation
            decay_rate=decay_rate,
            threshold=importance_threshold
        )
        self.threshold = ThresholdAtom(
            threshold=consolidation_threshold,
            mode='amplify'  # Amplify important memories
        )
        self.binder = BinderAtom(binding_strength=binding_strength)
        
        # Memory storage
        self.memories = {}  # memory_id -> memory_data
        self.memory_counter = 0
        self.tags_index = defaultdict(list)  # tag -> [memory_ids]
        self.importance_index = defaultdict(list)  # importance_level -> [memory_ids]
        self.consolidation_count = 0
        self.total_stored = 0
        self.total_retrieved = 0
    
    def store(self, field: NDAnalogField, importance: float = 0.5, 
              tags: list = None, metadata: dict = None) -> str:
        """Store information in long-term memory
        
        Args:
            field: Field to store
            importance: Importance level (0.0-1.0)
            tags: List of tags for retrieval
            metadata: Additional metadata
            
        Returns:
            Memory ID for the stored item
        """
        if importance < self.importance_threshold:
            return None  # Not important enough to store
        
        self.memory_counter += 1
        memory_id = f"ltm_{self.memory_counter}"
        self.total_stored += 1
        
        # Create memory entry
        memory_data = {
            'id': memory_id,
            'activation': field.activation.copy(),
            'shape': field.shape,
            'importance': importance,
            'tags': tags or [],
            'metadata': metadata or {},
            'creation_time': self.memory_counter,
            'access_count': 0,
            'last_accessed': self.memory_counter,
            'consolidation_level': 0.0
        }
        
        # Store in memory
        self.memories[memory_id] = memory_data
        
        # Update indices
        for tag in memory_data['tags']:
            self.tags_index[tag].append(memory_id)
        
        importance_level = int(importance * 10)  # 0-10 scale
        self.importance_index[importance_level].append(memory_id)
        
        # Apply memory trace
        self.memory_trace.apply(field)
        
        # Apply threshold for consolidation
        self.threshold.apply(field)
        
        return memory_id
    
    def retrieve(self, memory_id: str) -> NDAnalogField:
        """Retrieve specific memory by ID
        
        Args:
            memory_id: ID of memory to retrieve
            
        Returns:
            Field with retrieved memory, or None if not found
        """
        if memory_id not in self.memories:
            return None
        
        memory_data = self.memories[memory_id]
        memory_data['access_count'] += 1
        memory_data['last_accessed'] = self.memory_counter
        self.total_retrieved += 1
        
        # Create field with retrieved memory
        field = NDAnalogField(memory_data['shape'])
        field.activation = memory_data['activation'].copy()
        
        # Apply memory trace for context
        self.memory_trace.apply(field)
        
        return field
    
    def retrieve_by_tags(self, tags: list, max_results: int = 10) -> list:
        """Retrieve memories by tags
        
        Args:
            tags: List of tags to search for
            max_results: Maximum number of results
            
        Returns:
            List of (memory_id, field) tuples
        """
        # Find memories that match any of the tags
        matching_ids = set()
        for tag in tags:
            if tag in self.tags_index:
                matching_ids.update(self.tags_index[tag])
        
        # Sort by importance and access count
        scored_memories = []
        for memory_id in matching_ids:
            memory_data = self.memories[memory_id]
            score = (memory_data['importance'] * 0.7 + 
                    memory_data['access_count'] * 0.3)
            scored_memories.append((score, memory_id))
        
        scored_memories.sort(reverse=True)
        
        # Retrieve top results
        results = []
        for _, memory_id in scored_memories[:max_results]:
            field = self.retrieve(memory_id)
            if field is not None:
                results.append((memory_id, field))
        
        return results
    
    def retrieve_by_importance(self, min_importance: float = 0.7, 
                              max_results: int = 10) -> list:
        """Retrieve memories by importance level
        
        Args:
            min_importance: Minimum importance threshold
            max_results: Maximum number of results
            
        Returns:
            List of (memory_id, field) tuples
        """
        # Find memories above importance threshold
        matching_memories = []
        for memory_id, memory_data in self.memories.items():
            if memory_data['importance'] >= min_importance:
                score = (memory_data['importance'] * 0.8 + 
                        memory_data['access_count'] * 0.2)
                matching_memories.append((score, memory_id))
        
        matching_memories.sort(reverse=True)
        
        # Retrieve top results
        results = []
        for _, memory_id in matching_memories[:max_results]:
            field = self.retrieve(memory_id)
            if field is not None:
                results.append((memory_id, field))
        
        return results
    
    def consolidate_memories(self, consolidation_factor: float = 1.5) -> int:
        """Consolidate frequently accessed memories
        
        Args:
            consolidation_factor: How much to strengthen consolidated memories
            
        Returns:
            Number of memories consolidated
        """
        consolidated_count = 0
        
        for memory_id, memory_data in self.memories.items():
            # Consolidate if accessed multiple times
            if memory_data['access_count'] > 2:
                # Strengthen the memory
                memory_data['activation'] *= consolidation_factor
                memory_data['activation'] = np.clip(memory_data['activation'], 0, 1)
                
                # Increase consolidation level
                memory_data['consolidation_level'] = min(
                    memory_data['consolidation_level'] + 0.1, 1.0
                )
                
                consolidated_count += 1
        
        self.consolidation_count += 1
        return consolidated_count
    
    def create_association(self, memory_id_a: str, memory_id_b: str, 
                          strength: float = None) -> bool:
        """Create associative link between two memories
        
        Args:
            memory_id_a: First memory ID
            memory_id_b: Second memory ID
            strength: Association strength (uses default if None)
            
        Returns:
            True if association created successfully
        """
        if memory_id_a not in self.memories or memory_id_b not in self.memories:
            return False
        
        # Retrieve memories
        field_a = self.retrieve(memory_id_a)
        field_b = self.retrieve(memory_id_b)
        
        if field_a is None or field_b is None:
            return False
        
        # Create binding
        self.binder.bind(field_a, field_b)
        
        # Store association in metadata
        if 'associations' not in self.memories[memory_id_a]['metadata']:
            self.memories[memory_id_a]['metadata']['associations'] = []
        
        self.memories[memory_id_a]['metadata']['associations'].append({
            'memory_id': memory_id_b,
            'strength': strength or self.binder.binding_strength,
            'created_time': self.memory_counter
        })
        
        return True
    
    def retrieve_by_association(self, memory_id: str, max_results: int = 5) -> list:
        """Retrieve memories associated with given memory
        
        Args:
            memory_id: Memory to find associations for
            max_results: Maximum number of results
            
        Returns:
            List of (memory_id, field) tuples
        """
        if memory_id not in self.memories:
            return []
        
        memory_data = self.memories[memory_id]
        associations = memory_data['metadata'].get('associations', [])
        
        # Sort by strength
        associations.sort(key=lambda x: x['strength'], reverse=True)
        
        # Retrieve associated memories
        results = []
        for assoc in associations[:max_results]:
            assoc_id = assoc['memory_id']
            if assoc_id in self.memories:
                field = self.retrieve(assoc_id)
                if field is not None:
                    results.append((assoc_id, field))
        
        return results
    
    def forget_old_memories(self, age_threshold: int = 100, 
                           importance_threshold: float = 0.3) -> int:
        """Remove old, unimportant memories
        
        Args:
            age_threshold: Maximum age to keep
            importance_threshold: Minimum importance to keep
            
        Returns:
            Number of memories removed
        """
        current_time = self.memory_counter
        memories_to_remove = []
        
        for memory_id, memory_data in self.memories.items():
            age = current_time - memory_data['creation_time']
            
            # Remove if old and unimportant
            if (age > age_threshold and 
                memory_data['importance'] < importance_threshold):
                memories_to_remove.append(memory_id)
        
        # Remove memories
        for memory_id in memories_to_remove:
            self._remove_memory(memory_id)
        
        return len(memories_to_remove)
    
    def _remove_memory(self, memory_id: str):
        """Remove memory and update indices"""
        if memory_id not in self.memories:
            return
        
        memory_data = self.memories[memory_id]
        
        # Remove from tag index
        for tag in memory_data['tags']:
            if tag in self.tags_index and memory_id in self.tags_index[tag]:
                self.tags_index[tag].remove(memory_id)
        
        # Remove from importance index
        importance_level = int(memory_data['importance'] * 10)
        if importance_level in self.importance_index:
            if memory_id in self.importance_index[importance_level]:
                self.importance_index[importance_level].remove(memory_id)
        
        # Remove from memories
        del self.memories[memory_id]
    
    def get_memory_statistics(self) -> dict:
        """Get statistics about long-term memory"""
        if not self.memories:
            return {
                'memory_count': 0,
                'total_stored': self.total_stored,
                'total_retrieved': self.total_retrieved,
                'consolidation_count': self.consolidation_count,
                'average_importance': 0.0,
                'memory_trace_strength': 0.0
            }
        
        importances = [mem['importance'] for mem in self.memories.values()]
        access_counts = [mem['access_count'] for mem in self.memories.values()]
        
        return {
            'memory_count': len(self.memories),
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'consolidation_count': self.consolidation_count,
            'average_importance': np.mean(importances),
            'average_access_count': np.mean(access_counts),
            'memory_trace_strength': self.memory_trace.get_trace_strength(),
            'binding_count': self.binder.get_binding_count()
        }
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'memories': {mem_id: {
                'importance': mem['importance'],
                'tags': mem['tags'],
                'access_count': mem['access_count'],
                'consolidation_level': mem['consolidation_level']
            } for mem_id, mem in self.memories.items()},
            'memory_counter': self.memory_counter,
            'consolidation_count': self.consolidation_count,
            'memory_trace_strength': self.memory_trace.get_trace_strength(),
            'binding_count': self.binder.get_binding_count()
        }
    
    def reset(self):
        """Reset long-term memory"""
        self.memories.clear()
        self.tags_index.clear()
        self.importance_index.clear()
        self.memory_counter = 0
        self.consolidation_count = 0
        self.total_stored = 0
        self.total_retrieved = 0
        self.memory_trace.clear()
        self.binder.clear_bindings()
    
    def __repr__(self):
        return f"LongTermMemory(memories={len(self.memories)}, consolidated={self.consolidation_count})"