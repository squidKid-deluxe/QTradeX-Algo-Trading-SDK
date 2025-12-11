"""
Reasoning Molecules

Molecules for reasoning and logical operations, built from atomic operations.
These molecules handle analogical reasoning, contradiction resolution,
gap filling, and pattern completion for complex cognitive tasks.
"""

from .analogizer import Analogizer
from .contradiction_resolver import ContradictionResolver
from .gap_filler import GapFiller
from .pattern_completer import PatternCompleter

__all__ = [
    'Analogizer',
    'ContradictionResolver',
    'GapFiller',
    'PatternCompleter'
]