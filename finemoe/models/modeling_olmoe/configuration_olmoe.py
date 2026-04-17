"""Configuration wrapper for the local OLMoE model package.

The modeling module already carries a compatibility definition for
``OlmoeConfig`` when the installed Transformers version lacks a native OLMoE
config. Exposing the same symbol here keeps the package import surface aligned
with the lazy import contract in ``modeling_olmoe.__init__``.
"""

from .modeling_olmoe import OlmoeConfig

__all__ = ["OlmoeConfig"]
