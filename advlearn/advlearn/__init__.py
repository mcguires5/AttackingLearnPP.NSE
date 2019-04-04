"""Adversarial learning framework for testing machine learning systems

Subpackages
-----------
base
    Module which provides the base classes for other modules
attacks
    Module which provides all attack implementations
defenses
    Module which provides all defense implementations
classifier
    Module which provides online classifier implementations
ensemble
    Module that combines multiple other attacks or defenses
utils
    Module which provides various utility functions
"""

__version__ = "0.1.0"
__all__ = [
    "base",
    "attacks",
    "defenses",
    "classifier",
    "ensemble",
    "utils",
    "__version__",
]
