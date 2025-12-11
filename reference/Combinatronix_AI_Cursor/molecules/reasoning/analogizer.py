# ============================================================================
# Analogizer - Find Analogies Between Patterns
# ============================================================================

"""
Analogizer - Find analogies between patterns using bridge, comparison, and translation

Composition: Bridge + Comparator + Translator
Category: Reasoning
Complexity: Molecule (50-200 lines)

Finds analogies between patterns by comparing them, creating bridges between
similar regions, and translating patterns across different contexts. This
enables analogical reasoning, metaphor understanding, and cross-domain thinking.

Example:
    >>> analogizer = Analogizer(similarity_threshold=0.7, bridge_strength=0.8)
    >>> analogy = analogizer.find_analogy(pattern_a, pattern_b)
    >>> analogizer.create_metaphor(source_field, target_field)
    >>> analogies = analogizer.get_analogy_network()
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.field_dynamics import BridgeAtom
    from ...atoms.multi_field import ComparatorAtom, TranslatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.field_dynamics import BridgeAtom
    from combinatronix.atoms.multi_field import ComparatorAtom, TranslatorAtom
    from combinatronix.core import NDAnalogField


class Analogizer:
    """Find analogies between patterns using bridge, comparison, and translation"""
    
    def __init__(self, similarity_threshold: float = 0.7, bridge_strength: float = 0.8,
                 translation_strength: float = 0.6, comparison_metric: str = 'correlation'):
        """
        Args:
            similarity_threshold: Minimum similarity for analogy detection
            bridge_strength: Strength of bridges between analogous regions
            translation_strength: Strength of pattern translation
            comparison_metric: Metric for comparing patterns
        """
        self.bridge = BridgeAtom(bidirectional=True, transfer_rate=bridge_strength)
        self.comparator = ComparatorAtom(metric=comparison_metric, normalize=True)
        self.translator = TranslatorAtom(
            scale_factor=1.0,
            rotation=0.0,
            transformation='linear'
        )
        
        # Analogy state
        self.similarity_threshold = similarity_threshold
        self.translation_strength = translation_strength
        self.analogies = []  # List of analogy records
        self.analogy_network = defaultdict(list)  # pattern_id -> [analogous_patterns]
        self.metaphor_mappings = {}  # metaphor_id -> mapping_data
        self.analogy_count = 0
        self.total_processed = 0
    
    def find_analogy(self, pattern_a: np.ndarray, pattern_b: np.ndarray,
                    context_a: str = None, context_b: str = None) -> dict:
        """Find analogy between two patterns
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            context_a: Context for first pattern
            context_b: Context for second pattern
            
        Returns:
            Dictionary with analogy information
        """
        self.total_processed += 1
        
        # Create fields from patterns
        field_a = type('Field', (), {
            'activation': pattern_a,
            'shape': pattern_a.shape
        })()
        
        field_b = type('Field', (), {
            'activation': pattern_b,
            'shape': pattern_b.shape
        })()
        
        # Compare patterns
        similarity = self.comparator.compare(field_a, field_b)
        
        if similarity >= self.similarity_threshold:
            # Found an analogy
            self.analogy_count += 1
            
            analogy_record = {
                'id': f"analogy_{self.analogy_count}",
                'pattern_a': pattern_a.copy(),
                'pattern_b': pattern_b.copy(),
                'similarity': similarity,
                'context_a': context_a,
                'context_b': context_b,
                'strength': similarity,
                'created_time': self.total_processed,
                'usage_count': 0
            }
            
            self.analogies.append(analogy_record)
            
            # Update analogy network
            pattern_a_id = f"pattern_{id(pattern_a)}"
            pattern_b_id = f"pattern_{id(pattern_b)}"
            self.analogy_network[pattern_a_id].append(pattern_b_id)
            self.analogy_network[pattern_b_id].append(pattern_a_id)
            
            return analogy_record
        else:
            return {
                'id': None,
                'similarity': similarity,
                'is_analogy': False
            }
    
    def create_metaphor(self, source_field: NDAnalogField, target_field: NDAnalogField,
                       metaphor_name: str = None, transformation: str = 'linear') -> str:
        """Create metaphor by translating pattern from source to target
        
        Args:
            source_field: Source field with pattern to translate
            target_field: Target field to receive translated pattern
            metaphor_name: Name for the metaphor
            transformation: Type of transformation to apply
            
        Returns:
            Metaphor ID
        """
        if metaphor_name is None:
            metaphor_name = f"metaphor_{len(self.metaphor_mappings) + 1}"
        
        # Set up translator
        self.translator.transformation = transformation
        
        # Create metaphor mapping
        metaphor_id = f"metaphor_{id(source_field)}_{id(target_field)}"
        
        metaphor_data = {
            'id': metaphor_id,
            'name': metaphor_name,
            'source_field': source_field,
            'target_field': target_field,
            'transformation': transformation,
            'strength': self.translation_strength,
            'created_time': self.total_processed,
            'usage_count': 0
        }
        
        self.metaphor_mappings[metaphor_id] = metaphor_data
        
        # Apply translation
        self.translator.translate(source_field, target_field, self.translation_strength)
        
        return metaphor_id
    
    def apply_metaphor(self, metaphor_id: str, strength: float = None) -> bool:
        """Apply existing metaphor
        
        Args:
            metaphor_id: ID of metaphor to apply
            strength: Override strength (uses default if None)
            
        Returns:
            True if metaphor applied successfully
        """
        if metaphor_id not in self.metaphor_mappings:
            return False
        
        metaphor = self.metaphor_mappings[metaphor_id]
        metaphor['usage_count'] += 1
        
        # Apply translation with specified strength
        apply_strength = strength if strength is not None else metaphor['strength']
        self.translator.translate(metaphor['source_field'], metaphor['target_field'], apply_strength)
        
        return True
    
    def find_structural_analogy(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> dict:
        """Find structural analogy (shape/structure similarity)
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            
        Returns:
            Dictionary with structural analogy information
        """
        # Resize patterns to same size for comparison
        if pattern_a.shape != pattern_b.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern_a.shape, pattern_b.shape))
            if len(min_shape) == 2:
                pattern_a_resized = pattern_a[:min_shape[0], :min_shape[1]]
                pattern_b_resized = pattern_b[:min_shape[0], :min_shape[1]]
            else:
                pattern_a_resized = pattern_a
                pattern_b_resized = pattern_b
        else:
            pattern_a_resized = pattern_a
            pattern_b_resized = pattern_b
        
        # Create fields
        field_a = type('Field', (), {
            'activation': pattern_a_resized,
            'shape': pattern_a_resized.shape
        })()
        
        field_b = type('Field', (), {
            'activation': pattern_b_resized,
            'shape': pattern_b_resized.shape
        })()
        
        # Compare structural similarity
        similarity = self.comparator.compare(field_a, field_b)
        
        # Find matching regions
        matching_regions = self.comparator.find_matching_regions(field_a, field_b, 
                                                                threshold=self.similarity_threshold)
        
        return {
            'similarity': similarity,
            'matching_regions': matching_regions,
            'is_structural_analogy': similarity >= self.similarity_threshold,
            'pattern_a_shape': pattern_a.shape,
            'pattern_b_shape': pattern_b.shape
        }
    
    def create_analogical_bridge(self, field_a: NDAnalogField, field_b: NDAnalogField,
                               region_a: tuple, region_b: tuple, strength: float = None) -> bool:
        """Create bridge between analogous regions in two fields
        
        Args:
            field_a: First field
            field_b: Second field
            region_a: Region in first field
            region_b: Region in second field
            strength: Bridge strength (uses default if None)
            
        Returns:
            True if bridge created successfully
        """
        # Check if regions are analogous
        pattern_a = field_a.activation[region_a] if isinstance(region_a, tuple) else field_a.activation
        pattern_b = field_b.activation[region_b] if isinstance(region_b, tuple) else field_b.activation
        
        analogy = self.find_analogy(pattern_a, pattern_b)
        
        if analogy.get('is_analogy', False):
            # Create bridge
            bridge_strength = strength if strength is not None else self.bridge.transfer_rate
            self.bridge.connect(field_a, region_a, region_b, bridge_strength)
            return True
        
        return False
    
    def get_analogy_network(self) -> dict:
        """Get complete analogy network
        
        Returns:
            Dictionary representing the analogy network
        """
        return {
            'analogies': [{
                'id': a['id'],
                'similarity': a['similarity'],
                'context_a': a['context_a'],
                'context_b': a['context_b'],
                'strength': a['strength'],
                'usage_count': a['usage_count']
            } for a in self.analogies],
            'network_connections': dict(self.analogy_network),
            'metaphors': [{
                'id': m['id'],
                'name': m['name'],
                'transformation': m['transformation'],
                'strength': m['strength'],
                'usage_count': m['usage_count']
            } for m in self.metaphor_mappings.values()],
            'statistics': self.get_analogy_statistics()
        }
    
    def get_analogy_statistics(self) -> dict:
        """Get statistics about analogies and metaphors"""
        if not self.analogies:
            return {
                'analogy_count': 0,
                'metaphor_count': 0,
                'average_similarity': 0.0,
                'network_density': 0.0
            }
        
        similarities = [a['similarity'] for a in self.analogies]
        
        # Calculate network density
        total_possible_connections = len(self.analogies) * (len(self.analogies) - 1) // 2
        actual_connections = sum(len(connections) for connections in self.analogy_network.values()) // 2
        network_density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        return {
            'analogy_count': len(self.analogies),
            'metaphor_count': len(self.metaphor_mappings),
            'average_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'network_density': network_density,
            'total_processed': self.total_processed
        }
    
    def find_analogous_patterns(self, query_pattern: np.ndarray, 
                               max_results: int = 5) -> list:
        """Find patterns analogous to query pattern
        
        Args:
            query_pattern: Pattern to find analogies for
            max_results: Maximum number of results
            
        Returns:
            List of analogous patterns with similarity scores
        """
        analogous = []
        
        for analogy in self.analogies:
            # Compare with pattern_a
            similarity_a = self._compute_pattern_similarity(query_pattern, analogy['pattern_a'])
            if similarity_a >= self.similarity_threshold:
                analogous.append({
                    'pattern': analogy['pattern_b'],
                    'similarity': similarity_a,
                    'analogy_id': analogy['id'],
                    'context': analogy['context_b']
                })
            
            # Compare with pattern_b
            similarity_b = self._compute_pattern_similarity(query_pattern, analogy['pattern_b'])
            if similarity_b >= self.similarity_threshold:
                analogous.append({
                    'pattern': analogy['pattern_a'],
                    'similarity': similarity_b,
                    'analogy_id': analogy['id'],
                    'context': analogy['context_a']
                })
        
        # Sort by similarity and return top results
        analogous.sort(key=lambda x: x['similarity'], reverse=True)
        return analogous[:max_results]
    
    def _compute_pattern_similarity(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        """Compute similarity between two patterns"""
        # Ensure same shape for comparison
        if pattern_a.shape != pattern_b.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern_a.shape, pattern_b.shape))
            if len(min_shape) == 2:
                pattern_a = pattern_a[:min_shape[0], :min_shape[1]]
                pattern_b = pattern_b[:min_shape[0], :min_shape[1]]
        
        # Normalize patterns
        norm_a = pattern_a / (np.linalg.norm(pattern_a) + 1e-8)
        norm_b = pattern_b / (np.linalg.norm(pattern_b) + 1e-8)
        
        # Compute correlation
        correlation = np.dot(norm_a.flatten(), norm_b.flatten())
        return correlation
    
    def strengthen_analogy(self, analogy_id: str, strengthening_factor: float = 1.2):
        """Strengthen an existing analogy
        
        Args:
            analogy_id: ID of analogy to strengthen
            strengthening_factor: How much to strengthen
        """
        for analogy in self.analogies:
            if analogy['id'] == analogy_id:
                analogy['strength'] *= strengthening_factor
                analogy['strength'] = min(analogy['strength'], 1.0)
                break
    
    def get_metaphor_by_name(self, name: str) -> dict:
        """Get metaphor by name
        
        Args:
            name: Name of metaphor
            
        Returns:
            Metaphor data or None if not found
        """
        for metaphor in self.metaphor_mappings.values():
            if metaphor['name'] == name:
                return metaphor
        return None
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'analogies': [{
                'id': a['id'],
                'similarity': a['similarity'],
                'strength': a['strength'],
                'usage_count': a['usage_count']
            } for a in self.analogies],
            'metaphors': [{
                'id': m['id'],
                'name': m['name'],
                'transformation': m['transformation'],
                'usage_count': m['usage_count']
            } for m in self.metaphor_mappings.values()],
            'analogy_count': self.analogy_count,
            'total_processed': self.total_processed,
            'similarity_threshold': self.similarity_threshold,
            'bridge_strength': self.bridge.transfer_rate
        }
    
    def reset(self):
        """Reset analogizer state"""
        self.analogies.clear()
        self.analogy_network.clear()
        self.metaphor_mappings.clear()
        self.analogy_count = 0
        self.total_processed = 0
        self.bridge.clear_bridges()
        self.comparator.comparison_history.clear()
    
    def __repr__(self):
        return f"Analogizer(analogies={len(self.analogies)}, metaphors={len(self.metaphor_mappings)})"