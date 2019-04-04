"""Online classifiers"""
from .dummy import DummyClassifier
from .lasso import OnlineLasso

__all__ = ["DummyClassifier", "OnlineLasso"]
