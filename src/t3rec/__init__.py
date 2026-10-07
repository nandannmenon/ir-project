"""Inspectable, content-based news recommendation using sparse TF-IDF."""

from .models import Article, Recommendation, User
from .pipeline import Recommender
from .scoring import ScoreWeights

__all__ = ["Article", "Recommendation", "Recommender", "ScoreWeights", "User"]
