"""Rule-based explanations derived from the score components."""

from __future__ import annotations

from .models import Article
from .profiles import UserProfile
from .scoring import term_contributions
from .vectorize import SparseVector


def explain_recommendation(
    article: Article, profile: UserProfile, item_vector: SparseVector, *,
    relevance: float, jaccard: float, recency: float, popularity: float,
    category_match: float, net_score: float, mode: str = "personalized", top_n: int = 3,
    relevance_normalized: float | None = None, recency_available: bool = True,
    interest_label: str | None = None, interest_size: int = 0, history_size: int = 0,
    interest_vector: SparseVector | None = None,
) -> tuple[str, tuple[tuple[str, float], ...]]:
    # With interest profiles, the cosine was taken against the matching
    # interest, so its terms are the ones that explain the score.
    query = interest_vector if interest_vector is not None else profile.vector
    terms = term_contributions(query, item_vector)[:top_n]
    if mode == "cold_start":
        parts: list[str] = ["No usable history was available; ranking uses popularity and recency"]
    elif mode == "sparse_history":
        parts = ["Limited history was blended with popularity and recency to reduce overfitting"]
    elif profile.vector:
        parts: list[str] = ["Recommended because your history matches"]
    else:
        parts = ["Cold-start recommendation from available quality signals"]
    if interest_label:
        parts.append(f"matches your interest '{interest_label}' "
                     f"({interest_size} of {history_size} articles you read)")
    if terms:
        parts.append("top TF-IDF contributors: " + ", ".join(
            f"{term} ({value:.3f})" for term, value in terms))
    if category_match > 0:
        parts.append(f"category '{article.category}' (user-history weight {category_match:.3f})")
    cosine_text = f"cosine relevance {relevance:.3f}"
    if relevance_normalized is not None:
        cosine_text += f" ({relevance_normalized:.2f} of the best candidate)"
    parts.extend((cosine_text,
                  f"Jaccard/set similarity {jaccard:.3f}",
                  f"recency {recency:.3f}" if recency_available
                  else "recency n/a (no article timestamps)",
                  f"historical popularity {popularity:.3f}",
                  f"net score {net_score:.3f}"))
    return "; ".join(parts) + ".", terms
