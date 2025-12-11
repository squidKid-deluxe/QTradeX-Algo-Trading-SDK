"""
Attention Molecules

Molecules for attention and focus mechanisms, built from atomic operations.
These molecules handle focus, saliency detection, novelty detection,
and attention shifting for selective processing.
"""

from .focus import Focus
from .saliency import Saliency
from .novelty_detector import NoveltyDetector
from .attention_shift import AttentionShift

__all__ = [
    'Focus',
    'Saliency',
    'NoveltyDetector',
    'AttentionShift'
]