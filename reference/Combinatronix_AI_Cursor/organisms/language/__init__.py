"""
Language Organisms

Organisms for language processing and generation, built from atoms and molecules.
These organisms handle language modeling, text generation, and linguistic
pattern recognition using the Combinatronix molecular architecture.
"""

from .mini_llm import MiniLLM, WordEcho

__all__ = [
    'MiniLLM',
    'WordEcho'
]

