"""
Emotion Molecules

Molecules for emotional processing and regulation, built from atomic operations.
These molecules handle mood regulation, emotional memory, empathy simulation,
and emotional amplification for affective computing and emotional intelligence.
"""

from .mood_regulator import MoodRegulator
from .emotional_memory import EmotionalMemory
from .empathy_simulator import EmpathySimulator
from .emotional_amplifier import EmotionalAmplifier

__all__ = [
    'MoodRegulator',
    'EmotionalMemory',
    'EmpathySimulator',
    'EmotionalAmplifier'
]