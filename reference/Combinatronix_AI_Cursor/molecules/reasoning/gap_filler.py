# ============================================================================
# GapFiller - Fill Conceptual Gaps with Anticipated Patterns
# ============================================================================

"""
GapFiller - Fill conceptual gaps using filler and anticipation

Composition: Filler + Anticipator
Category: Reasoning
Complexity: Molecule (50-200 lines)

Fills conceptual gaps by detecting missing information, anticipating what
should be there based on context, and creating appropriate fill patterns.
This enables gap-filling inference, creative problem-solving, and
completion of incomplete information.

Example:
    >>> gap_filler = GapFiller(creativity=0.6, anticipation_strength=0.4)
    >>> filled_field = gap_filler.fill_gaps(field)
    >>> predictions = gap_filler.get_predictions()
    >>> gap_analysis = gap_filler.analyze_gaps(field)
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.tension_resolvers import FillerAtom
    from ...atoms.temporal import AnticipatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.tension_resolvers import FillerAtom
    from combinatronix.atoms.temporal import AnticipatorAtom
    from combinatronix.core import NDAnalogField


class GapFiller:
    """Fill conceptual gaps using filler and anticipation"""
    
    def __init__(self, creativity: float = 0.6, anticipation_strength: float = 0.4,
                 gap_threshold: float = 0.1, history_depth: int = 3):
        """
        Args:
            creativity: How creative the gap filling should be (0.0-1.0)
            anticipation_strength: How much to use anticipation for filling
            gap_threshold: Minimum gap size to trigger filling
            history_depth: How many past states to use for anticipation
        """
        self.filler = FillerAtom(
            creativity=creativity,
            gap_threshold=gap_threshold
        )
        self.anticipator = AnticipatorAtom(
            history_depth=history_depth,
            prediction_weight=anticipation_strength
        )
        
        # Gap filling state
        self.gap_analysis = {}  # gap_id -> gap_data
        self.fill_history = []  # List of fill operations
        self.predictions = []  # List of predictions made
        self.gap_count = 0
        self.total_processed = 0
        self.creativity = creativity
        self.anticipation_strength = anticipation_strength
    
    def fill_gaps(self, field: NDAnalogField, context: str = None) -> NDAnalogField:
        """Fill gaps in field using anticipation and creativity
        
        Args:
            field: Field to fill gaps in
            context: Context for gap filling
            
        Returns:
            Field with gaps filled
        """
        self.total_processed += 1
        
        # Analyze gaps first
        gap_analysis = self.analyze_gaps(field)
        
        # Create working copy
        filled_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        # Apply anticipation to get predictions
        self.anticipator.apply(filled_field)
        
        # Apply filler to fill gaps
        self.filler.apply(filled_field)
        
        # Record fill operation
        self._record_fill_operation(field, filled_field, gap_analysis, context)
        
        # Update field
        field.activation = filled_field.activation
        
        return field
    
    def analyze_gaps(self, field: NDAnalogField) -> dict:
        """Analyze gaps in field
        
        Args:
            field: Field to analyze
            
        Returns:
            Dictionary with gap analysis
        """
        # Detect gaps using filler
        gaps = self.filler._detect_gaps(field)
        
        # Analyze gap characteristics
        gap_analysis = {
            'gap_count': len(gaps),
            'gap_locations': gaps,
            'gap_density': len(gaps) / field.activation.size,
            'field_energy': np.sum(np.abs(field.activation)),
            'field_completeness': 1.0 - (len(gaps) / field.activation.size)
        }
        
        # Analyze individual gaps
        gap_details = []
        for i, gap_location in enumerate(gaps):
            gap_detail = self._analyze_individual_gap(field, gap_location)
            gap_detail['id'] = f"gap_{i}"
            gap_detail['location'] = gap_location
            gap_details.append(gap_detail)
        
        gap_analysis['gap_details'] = gap_details
        
        # Store analysis
        analysis_id = f"analysis_{self.total_processed}"
        self.gap_analysis[analysis_id] = gap_analysis
        
        return gap_analysis
    
    def _analyze_individual_gap(self, field: NDAnalogField, gap_location: tuple) -> dict:
        """Analyze individual gap characteristics
        
        Args:
            field: Field containing the gap
            gap_location: Location of the gap
            
        Returns:
            Dictionary with gap characteristics
        """
        i, j = gap_location
        
        # Get surrounding context
        context_values = []
        context_strength = 0.0
        
        if len(field.shape) == 2:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < field.shape[0] and 0 <= nj < field.shape[1] and 
                        (di != 0 or dj != 0)):  # Exclude center
                        context_values.append(field.activation[ni, nj])
                        context_strength += field.activation[ni, nj]
        
        # Calculate gap characteristics
        context_mean = np.mean(context_values) if context_values else 0.0
        context_variance = np.var(context_values) if len(context_values) > 1 else 0.0
        gap_strength = context_strength / len(context_values) if context_values else 0.0
        
        return {
            'context_mean': context_mean,
            'context_variance': context_variance,
            'gap_strength': gap_strength,
            'context_size': len(context_values),
            'is_isolated': context_strength < 0.1,
            'is_high_context': context_strength > 0.5
        }
    
    def _record_fill_operation(self, original_field: NDAnalogField, filled_field: NDAnalogField,
                              gap_analysis: dict, context: str):
        """Record gap filling operation"""
        fill_operation = {
            'operation_id': f"fill_{len(self.fill_history) + 1}",
            'original_energy': np.sum(np.abs(original_field.activation)),
            'filled_energy': np.sum(np.abs(filled_field.activation)),
            'gaps_filled': gap_analysis['gap_count'],
            'context': context,
            'creativity_used': self.creativity,
            'anticipation_used': self.anticipation_strength,
            'timestamp': self.total_processed
        }
        
        self.fill_history.append(fill_operation)
        
        # Store prediction if available
        if self.anticipator.last_prediction is not None:
            prediction_record = {
                'prediction_id': f"pred_{len(self.predictions) + 1}",
                'prediction': self.anticipator.last_prediction.copy(),
                'actual': filled_field.activation.copy(),
                'error': self.anticipator.get_prediction_error(filled_field),
                'timestamp': self.total_processed
            }
            self.predictions.append(prediction_record)
    
    def get_predictions(self) -> list:
        """Get recent predictions made by anticipator
        
        Returns:
            List of recent predictions
        """
        return self.predictions[-10:] if self.predictions else []
    
    def get_prediction_accuracy(self) -> dict:
        """Get prediction accuracy statistics
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.predictions:
            return {
                'total_predictions': 0,
                'average_error': 0.0,
                'accuracy': 0.0
            }
        
        errors = [pred['error'] for pred in self.predictions]
        avg_error = np.mean(errors)
        accuracy = 1.0 - avg_error  # Higher accuracy = lower error
        
        return {
            'total_predictions': len(self.predictions),
            'average_error': avg_error,
            'accuracy': accuracy,
            'recent_accuracy': np.mean(errors[-5:]) if len(errors) >= 5 else avg_error
        }
    
    def create_bridging_concept(self, field_a: NDAnalogField, field_b: NDAnalogField) -> np.ndarray:
        """Create bridging concept between two fields
        
        Args:
            field_a: First field
            field_b: Second field
            
        Returns:
            Bridging concept pattern
        """
        # Use filler to invent bridging concept
        bridge_pattern = self.filler.invent_bridging_concept(field_a, field_b)
        
        # Use anticipator to refine the bridge
        bridge_field = type('Field', (), {
            'activation': bridge_pattern,
            'shape': bridge_pattern.shape
        })()
        
        # Add to anticipator history
        self.anticipator.history.append(bridge_pattern)
        
        return bridge_pattern
    
    def fill_gaps_with_anticipation(self, field: NDAnalogField, 
                                   anticipation_weight: float = None) -> NDAnalogField:
        """Fill gaps using anticipation as primary method
        
        Args:
            field: Field to fill
            anticipation_weight: Weight for anticipation (uses default if None)
            
        Returns:
            Field with gaps filled using anticipation
        """
        if anticipation_weight is None:
            anticipation_weight = self.anticipation_strength
        
        # Create working copy
        filled_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        # Get prediction
        prediction = self.anticipator.predict(filled_field)
        
        # Find gaps
        gaps = self.filler._detect_gaps(filled_field)
        
        # Fill gaps with prediction
        for gap_location in gaps:
            if len(field.shape) == 2:
                i, j = gap_location
                if 0 <= i < prediction.shape[0] and 0 <= j < prediction.shape[1]:
                    # Blend prediction with current value
                    current_value = filled_field.activation[gap_location]
                    predicted_value = prediction[gap_location]
                    
                    filled_value = (current_value * (1 - anticipation_weight) + 
                                  predicted_value * anticipation_weight)
                    
                    filled_field.activation[gap_location] = filled_value
        
        return filled_field
    
    def get_gap_statistics(self) -> dict:
        """Get statistics about gap filling
        
        Returns:
            Dictionary with gap filling statistics
        """
        if not self.fill_history:
            return {
                'total_operations': 0,
                'total_gaps_filled': 0,
                'average_gaps_per_operation': 0.0,
                'energy_increase': 0.0
            }
        
        total_gaps = sum(op['gaps_filled'] for op in self.fill_history)
        avg_gaps = total_gaps / len(self.fill_history)
        
        energy_increases = [op['filled_energy'] - op['original_energy'] 
                          for op in self.fill_history]
        avg_energy_increase = np.mean(energy_increases)
        
        return {
            'total_operations': len(self.fill_history),
            'total_gaps_filled': total_gaps,
            'average_gaps_per_operation': avg_gaps,
            'energy_increase': avg_energy_increase,
            'creativity_level': self.creativity,
            'anticipation_level': self.anticipation_strength
        }
    
    def get_surprise_map(self, field: NDAnalogField) -> np.ndarray:
        """Get surprise map (difference from anticipation)
        
        Args:
            field: Field to analyze
            
        Returns:
            Surprise map showing unexpected regions
        """
        return self.anticipator.get_surprise(field)
    
    def adapt_creativity(self, adaptation_rate: float = 0.1):
        """Adapt creativity based on recent performance
        
        Args:
            adaptation_rate: How fast to adapt
        """
        if len(self.predictions) < 3:
            return
        
        # Calculate recent prediction accuracy
        recent_errors = [pred['error'] for pred in self.predictions[-5:]]
        avg_recent_error = np.mean(recent_errors)
        
        # Adapt creativity based on performance
        if avg_recent_error > 0.3:  # High error - reduce creativity
            self.creativity *= (1 - adaptation_rate)
            self.filler.creativity = self.creativity
        elif avg_recent_error < 0.1:  # Low error - increase creativity
            self.creativity *= (1 + adaptation_rate)
            self.filler.creativity = self.creativity
        
        # Keep within bounds
        self.creativity = np.clip(self.creativity, 0.1, 0.9)
    
    def get_filled_gaps(self) -> list:
        """Get list of all filled gap locations
        
        Returns:
            List of filled gap locations
        """
        return self.filler.get_filled_gaps()
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'gap_analysis_count': len(self.gap_analysis),
            'fill_operations': len(self.fill_history),
            'predictions': len(self.predictions),
            'creativity': self.creativity,
            'anticipation_strength': self.anticipation_strength,
            'total_processed': self.total_processed,
            'filled_gaps_count': len(self.filler.filled_gaps),
            'invented_patterns_count': len(self.filler.invented_patterns)
        }
    
    def reset(self):
        """Reset gap filler state"""
        self.gap_analysis.clear()
        self.fill_history.clear()
        self.predictions.clear()
        self.gap_count = 0
        self.total_processed = 0
        self.filler.filled_gaps.clear()
        self.filler.invented_patterns.clear()
        self.anticipator.reset()
    
    def __repr__(self):
        return f"GapFiller(creativity={self.creativity:.2f}, operations={len(self.fill_history)})"