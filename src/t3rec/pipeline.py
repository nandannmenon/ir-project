"""End-to-end sparse vector-space news recommender."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from math import log1p
from typing import Callable, Iterable, Mapping, Sequence

from .candidates import (build_champion_lists, build_cluster_index,
                         generate_candidates, generate_cluster_candidates)
from .diversity import (DiversityComparison, compare_diversity as build_diversity_comparison,
                       diversity_rerank)
from .explain import explain_recommendation
from .models import Article, Recommendation, User
from .profiles import (UserProfile, build_interest_profiles, build_user_profile,
                       proportional_quotas)
from .scoring import (ScoreWeights, article_feature_set, calculate_net_score,
                      category_match_score, cosine_relevance,
                      jaccard_similarity, popularity_score, recency_score)
from .text import TextConfig, preprocess_text
from .vectorize import ItemVectors, SparseVector, build_item_vectors


@dataclass(frozen=True, slots=True)
class ColdStartConfig:
    """Mode thresholds and fusion weights for short histories.

    With no clicks, only recency/popularity have nonzero weights. With one or
    two clicks, text remains useful but receives 50% total weight rather than
    the normal 85%; this explicitly limits the influence of a tiny sample.
    """

    sparse_history_max: int = 2
    sparse_weights: ScoreWeights = ScoreWeights(alpha=0.50, beta=0.10, gamma=0.15, delta=0.25)
    cold_start_weights: ScoreWeights = ScoreWeights(alpha=0.0, beta=0.0, gamma=0.30, delta=0.70)
    diversify_cold_start_by_category: bool = True

    def __post_init__(self) -> None:
        if self.sparse_history_max < 1:
            raise ValueError("sparse_history_max must be at least 1")


class Recommender:
    """Builds an inspectable TF-IDF index and scores champion-list candidates.

    Invalid or unavailable articles are removed before indexing. In the normal
    path, an interest champion list supplies a bounded candidate shortlist, so
    exact cosine scoring is performed on that shortlist rather than the catalog.
    """

    def __init__(self, articles: Iterable[Article], *, config: TextConfig | None = None,
                 click_counts: Mapping[str, int] | None = None,
                 available_article_ids: set[str] | None = None,
                 score_weights: ScoreWeights | None = None,
                 champion_limit: int = 50,
                 candidate_interest_limit: int = 20,
                 champion_per_interest_limit: int = 50,
                 cluster_leader_count: int | None = None,
                 cluster_followers_per_item: int = 2,
                 cluster_probe_count: int = 20,
                 history_duplicate_threshold: float | None = 0.9,
                 normalize_query_components: bool = True,
                 interest_threshold: float = 0.1,
                 max_interests: int = 3,
                 item_vectors: ItemVectors | None = None,
                 cold_start_config: ColdStartConfig | None = None):
        if history_duplicate_threshold is not None and not 0.0 < history_duplicate_threshold <= 1.0:
            raise ValueError("history_duplicate_threshold must be in (0, 1] or None")
        self.config = config or TextConfig()
        available = set(available_article_ids) if available_article_ids is not None else None
        self.rejected_articles: dict[str, str] = {}
        self.article_by_id: dict[str, Article] = {}
        for article in articles:
            article_id = (article.article_id or "").strip()
            if not article_id:
                self.rejected_articles[f"<row-{len(self.rejected_articles)}>"] = "missing article ID"
                continue
            if article_id in self.article_by_id:
                self.rejected_articles[article_id] = "duplicate article ID"
                continue
            if not article.document_text.strip():
                self.rejected_articles[article_id] = "missing title and abstract"
                continue
            if available is not None and article_id not in available:
                self.rejected_articles[article_id] = "unavailable article"
                continue
            self.article_by_id[article_id] = article
        source_vectors = item_vectors or build_item_vectors(self.article_by_id.values(), self.config)
        self.item_vectors = ItemVectors(
            {article_id: source_vectors.vector(article_id) for article_id in self.article_by_id},
            source_vectors.document_frequency, source_vectors.idf, source_vectors.vocabulary,
        )
        # Stopword-only documents produce no terms and cannot participate in retrieval.
        for article_id in tuple(self.article_by_id):
            if not self.item_vectors.vector(article_id):
                self.rejected_articles[article_id] = "no indexable terms"
                del self.article_by_id[article_id]
                self.item_vectors.vectors.pop(article_id, None)
        self.articles = list(self.article_by_id.values())
        self.available_article_ids = set(self.article_by_id)
        self.click_counts = {key: max(0, int(value)) for key, value in (click_counts or {}).items()}
        self.max_log_clicks = max((log1p(value) for value in self.click_counts.values()), default=0.0)
        self.score_weights = score_weights or ScoreWeights()
        self.champions = build_champion_lists(self.item_vectors, limit=champion_limit)
        self.cluster_index = build_cluster_index(self.item_vectors, cluster_leader_count,
                                                 followers_per_item=cluster_followers_per_item)
        self.cluster_probe_count = cluster_probe_count
        self.history_duplicate_threshold = history_duplicate_threshold
        self.normalize_query_components = normalize_query_components
        self.interest_threshold = interest_threshold
        self.max_interests = max_interests
        self.candidate_interest_limit = candidate_interest_limit
        self.champion_per_interest_limit = champion_per_interest_limit
        self.cold_start_config = cold_start_config or ColdStartConfig()
        self.last_diagnostics: dict[str, object] = {}

    @classmethod
    def from_processed(cls, dataset: object, **kwargs: object) -> "Recommender":
        """Create a recommender from ``ProcessedDataset`` without rebuilding TF-IDF."""
        articles = getattr(dataset, "articles")
        vectors = getattr(dataset, "item_vectors")
        clicks = getattr(dataset, "click_counts")
        return cls(articles.values(), item_vectors=vectors,
                   click_counts=clicks, **kwargs)

    def profile(self, user: User, *, as_of: datetime | None = None,
                half_life_days: float | None = None,
                interaction_weights: Sequence[float] | None = None) -> UserProfile:
        """Construct the user's normalized query vector from historical item rows."""
        return build_user_profile(user, self.item_vectors, self.article_by_id,
                                  as_of=as_of, half_life_days=half_life_days,
                                  interaction_weights=interaction_weights)

    def recommend(self, user: User, *, limit: int = 10, as_of: datetime | None = None,
                  candidate_mode: str = "champions", candidate_pool_size: int = 500,
                  diverse: bool = True, diversity_relevance_weight: float = 0.75,
                  diversity_weight: float | None = None,
                  score_weights: ScoreWeights | Mapping[str, float] | None = None,
                  half_life_days: float | None = None,
                  recency_half_life_days: float = 14.0,
                  interaction_weights: Sequence[float] | None = None,
                  candidate_ids: set[str] | None = None,
                  multi_interest: bool = False) -> list[Recommendation]:
        """Return top-K results with every normalized score component exposed.

        ``candidate_mode='champions'`` is the efficient default. ``'all'`` is
        useful for small collections and evaluation. A cold-start user has no
        profile vector; candidates are selected by recency/popularity quality,
        with optional category-only diversification.

        With ``multi_interest=True`` the history is split into interests
        (``build_interest_profiles``). Each interest retrieves its own
        candidates, an article's cosine is taken against its best-matching
        interest, and result slots are shared across interests in proportion
        to how many read articles each holds.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        if candidate_mode not in {"champions", "clusters", "all"}:
            raise ValueError("candidate_mode must be 'champions', 'clusters', or 'all'")
        if diversity_weight is not None:
            diversity_relevance_weight = 1.0 - diversity_weight
        if not 0.0 <= diversity_relevance_weight <= 1.0:
            raise ValueError("diversity_relevance_weight must be between 0 and 1")

        reference_time = as_of
        if reference_time is None:
            timestamps = [article.timestamp for article in self.articles if article.timestamp]
            reference_time = max(timestamps) if timestamps else None
        usable_interactions = sum(article_id in self.available_article_ids for article_id in user.history)
        if usable_interactions == 0:
            mode = "cold_start"
        elif usable_interactions <= self.cold_start_config.sparse_history_max:
            mode = "sparse_history"
        else:
            mode = "personalized"
        profile = self.profile(user, as_of=reference_time,
                               half_life_days=half_life_days,
                               interaction_weights=interaction_weights)
        interests = (build_interest_profiles(user, self.item_vectors, self.article_by_id,
                                             threshold=self.interest_threshold,
                                             max_interests=self.max_interests)
                     if multi_interest and profile.vector else [])
        queries = [interest.vector for interest in interests] or [profile.vector]
        per_query_pool = -(-candidate_pool_size // len(queries))
        allowed = candidate_ids
        chosen_weights = score_weights
        if chosen_weights is None:
            if mode == "cold_start":
                chosen_weights = self.cold_start_config.cold_start_weights
            elif mode == "sparse_history":
                chosen_weights = self.cold_start_config.sparse_weights
            else:
                chosen_weights = self.score_weights
        if candidate_mode in {"champions", "clusters"} and mode == "cold_start":
            # Cheap, content-independent quality shortlist. Recency participates
            # in candidate generation as well as final scoring.
            quality_weights = (chosen_weights.as_mapping() if isinstance(chosen_weights, ScoreWeights)
                               else dict(chosen_weights))
            quality_weight_sum = quality_weights.get("recency", 0.0) + quality_weights.get("popularity", 0.0)
            if quality_weight_sum <= 0:
                quality_weight_sum = 1.0
            def cold_quality(article: Article) -> float:
                fresh = recency_score(article, reference_time, recency_half_life_days) if reference_time else 0.0
                popular = popularity_score(article.article_id, self.click_counts,
                                           max_log_clicks=self.max_log_clicks)
                return (quality_weights.get("recency", 0.0) * fresh +
                        quality_weights.get("popularity", 0.0) * popular) / quality_weight_sum
            eligible = (article for article in self.articles
                        if article.article_id not in profile.consumed and
                        (allowed is None or article.article_id in allowed))
            candidates = heapq.nsmallest(candidate_pool_size, eligible,
                                         key=lambda article: (-cold_quality(article), article.article_id))
        elif candidate_mode == "champions" and profile.vector:
            # One champion-list retrieval per query vector (the whole profile,
            # or each interest), merged in query order.
            merged: dict[str, Article] = {}
            for query in queries:
                for article in generate_candidates(
                    self.article_by_id, profile.consumed,
                    profile=query, champions=self.champions,
                    item_vectors=self.item_vectors, limit=per_query_pool,
                    interest_limit=self.candidate_interest_limit,
                    per_interest_limit=self.champion_per_interest_limit,
                    available_ids=self.available_article_ids, candidate_ids=allowed,
                ):
                    merged.setdefault(article.article_id, article)
            candidates = list(merged.values())
        elif candidate_mode == "clusters" and profile.vector:
            merged = {}
            probes = max(1, -(-self.cluster_probe_count // len(queries)))
            for query in queries:
                for article in generate_cluster_candidates(
                    self.article_by_id, profile.consumed,
                    profile=query, cluster_index=self.cluster_index,
                    item_vectors=self.item_vectors, limit=per_query_pool,
                    probe_count=probes,
                    available_ids=self.available_article_ids, candidate_ids=allowed,
                ):
                    merged.setdefault(article.article_id, article)
            candidates = list(merged.values())
        elif candidate_mode in {"champions", "clusters"}:
            # Sparse histories can yield no matching champion; fall back to a
            # bounded quality shortlist while retaining sparse-history scoring.
            eligible = (article for article in self.articles
                        if article.article_id not in profile.consumed and
                        (allowed is None or article.article_id in allowed))
            candidates = heapq.nsmallest(
                candidate_pool_size, eligible,
                key=lambda article: (-(popularity_score(article.article_id, self.click_counts,
                                                        max_log_clicks=self.max_log_clicks) +
                                       (recency_score(article, reference_time, recency_half_life_days)
                                        if reference_time else 0.0)), article.article_id),
            )
        else:
            candidates = generate_candidates(
                self.article_by_id, profile.consumed,
                available_ids=self.available_article_ids, candidate_ids=allowed,
            )

        weights_map = (chosen_weights.as_mapping() if isinstance(chosen_weights, ScoreWeights)
                       else dict(chosen_weights))
        # Recency needs article timestamps, and MIND news.tsv has none. A
        # constant zero would only shrink every score, so its weight is dropped
        # and the explanation says recency is unavailable.
        recency_available = reference_time is not None and any(
            article.timestamp is not None for article in candidates)
        if not recency_available and weights_map.get("recency"):
            weights_map = {**weights_map, "recency": 0.0}

        components = []
        for article in candidates:
            item_vector = self.item_vectors.vector(article.article_id)
            # Against several interests, relevance is the cosine to the
            # best-matching one. -1 means no interest: the single-profile
            # path, or an article sharing no term with any interest.
            interest_index = -1
            if interests:
                similarities = [cosine_relevance(interest.vector, item_vector)
                                for interest in interests]
                best = max(range(len(interests)), key=lambda index: (similarities[index], -index))
                relevance = similarities[best]
                if relevance > 0:
                    interest_index = best
            else:
                relevance = cosine_relevance(profile.vector, item_vector)
            jaccard = jaccard_similarity(profile.feature_set, article_feature_set(article, item_vector))
            fresh = (recency_score(article, reference_time, recency_half_life_days)
                     if recency_available else 0.0)
            popular = popularity_score(article.article_id, self.click_counts,
                                       max_log_clicks=self.max_log_clicks)
            components.append((article, item_vector, relevance, jaccard, fresh, popular,
                               interest_index))
        # Cosine and Jaccard are query-dependent and small (top MIND cosines are
        # ~0.1-0.4), while popularity is a static [0,1] quality score g(d).
        # Dividing both by their maximum over this candidate set puts them on
        # the same [0,1] scale, so the configured weights compare like with like.
        if self.normalize_query_components:
            max_relevance = max((row[2] for row in components), default=0.0)
            max_jaccard = max((row[3] for row in components), default=0.0)
        else:
            max_relevance = max_jaccard = 1.0
        scored = []
        for article, item_vector, relevance, jaccard, fresh, popular, interest_index in components:
            relevance_n = relevance / max_relevance if max_relevance else 0.0
            jaccard_n = jaccard / max_jaccard if max_jaccard else 0.0
            net = calculate_net_score(relevance_n, jaccard=jaccard_n, recency=fresh,
                                      popularity=popular, weights=weights_map)
            scored.append((net, article, item_vector, relevance, jaccard, fresh, popular,
                           relevance_n, jaccard_n, interest_index))

        # Heap-based top-K: heapify is linear, and only popped items are
        # checked against history and explained. Copies of already-read
        # articles are skipped so they never take a result slot.
        selection_size = min(len(scored), max(limit * 5, limit) if diverse else limit)
        find_history_copy = self._history_duplicate_finder(user.history)
        heap = [(-row[0], row[1].article_id, index) for index, row in enumerate(scored)]
        heapq.heapify(heap)
        # With several interests, each gets slots in proportion to its share
        # of the history. An item whose interest is already full waits in
        # ``overflow``; unfilled slots are later given out by net score. The
        # scan is bounded so a sparse interest cannot force a full heap drain.
        quotas = (proportional_quotas([len(interest.members) for interest in interests],
                                      selection_size) if len(interests) > 1 else None)
        initial_quotas = list(quotas) if quotas is not None else None
        scan_budget = selection_size * 20 if quotas is not None else None
        chosen = []
        overflow = []
        removed_copies: list[tuple[str, str]] = []
        while heap and len(chosen) < selection_size:
            if scan_budget is not None:
                if scan_budget == 0:
                    break
                scan_budget -= 1
            _, article_id, index = heapq.heappop(heap)
            copy_of = find_history_copy(scored[index][1], scored[index][2])
            if copy_of is not None:
                removed_copies.append((article_id, copy_of))
                continue
            if quotas is not None:
                interest_index = scored[index][9]
                if interest_index < 0 or quotas[interest_index] == 0:
                    overflow.append(scored[index])
                    continue
                quotas[interest_index] -= 1
            chosen.append(scored[index])
        for row in overflow:
            if len(chosen) >= selection_size:
                break
            chosen.append(row)
        while heap and len(chosen) < selection_size:
            _, article_id, index = heapq.heappop(heap)
            copy_of = find_history_copy(scored[index][1], scored[index][2])
            if copy_of is not None:
                removed_copies.append((article_id, copy_of))
                continue
            chosen.append(scored[index])
        chosen.sort(key=lambda row: (-row[0], row[1].article_id))

        top_candidates: list[Recommendation] = []
        owners = [row[9] for row in chosen]
        for index, (net, article, item_vector, relevance, jaccard, fresh, popular,
                    relevance_n, jaccard_n, interest_index) in enumerate(chosen, start=1):
            category = category_match_score(profile, article)
            interest = interests[interest_index] if interest_index >= 0 else None
            explanation, terms = explain_recommendation(
                article, profile, item_vector, relevance=relevance, jaccard=jaccard,
                recency=fresh, popularity=popular, category_match=category,
                net_score=net, mode=mode, relevance_normalized=relevance_n,
                recency_available=recency_available,
                interest_label=interest.label if interest else None,
                interest_size=len(interest.members) if interest else 0,
                history_size=sum(len(item.members) for item in interests),
                interest_vector=interest.vector if interest else None,
            )
            top_candidates.append(Recommendation(
                article=article, relevance=relevance, jaccard=jaccard,
                category_match=category, recency=fresh, popularity=popular,
                net_score=net, final_score=net, explanation=explanation,
                term_contributions=terms, candidate_count=len(candidates),
                score_weights=weights_map, mode=mode,
                original_rank=index, rank=index, reranked_rank=index,
                relevance_normalized=relevance_n, jaccard_normalized=jaccard_n,
                interest=interest.label if interest else "",
            ))
        if diverse and initial_quotas is not None:
            ranked = self._diversify_within_interests(
                top_candidates, owners, [len(interest.members) for interest in interests],
                limit=limit, relevance_weight=diversity_relevance_weight)
        elif diverse:
            ranked = diversity_rerank(top_candidates, self.item_vectors, limit=limit,
                                      relevance_weight=diversity_relevance_weight,
                                      category_only=(mode == "cold_start" and
                                          self.cold_start_config.diversify_cold_start_by_category))
        else:
            ranked = top_candidates[:limit]
        ranked = [replace(result, rank=index, reranked_rank=index,
                          final_score=(result.final_score if diverse else result.net_score))
                  for index, result in enumerate(ranked, start=1)]
        self.last_diagnostics = {
            "user_id": user.user_id,
            "profile_vector": profile.vector,
            "profile_features": sorted(profile.feature_set),
            "candidate_count": len(candidates),
            "candidate_ids": [article.article_id for article in candidates],
            "candidate_mode": candidate_mode,
            "recommendation_mode": mode,
            "usable_interaction_count": usable_interactions,
            "score_weights": weights_map,
            "recency_available": recency_available,
            "normalization_max": {"relevance": max_relevance, "jaccard": max_jaccard},
            "history_copies_removed": removed_copies,
            "interests": [{"label": interest.label, "category": interest.category,
                           "size": len(interest.members), "members": list(interest.members)}
                          for interest in interests],
            "interest_quotas": initial_quotas,
            "reference_time": reference_time.isoformat() if reference_time else None,
            "rejected_article_count": len(self.rejected_articles),
        }
        return ranked

    def _diversify_within_interests(
        self, candidates: list[Recommendation], owners: list[int], sizes: list[int], *,
        limit: int, relevance_weight: float,
    ) -> list[Recommendation]:
        """Share the final ``limit`` slots across interests, then run MMR per interest.

        Running MMR over the whole pool would let the strongest interest win
        every slot again; reranking inside each interest keeps the
        proportional quotas while still removing near-duplicates. Slots an
        interest cannot fill go to the remaining candidates, also by MMR.
        """
        ranked: list[Recommendation] = []
        leftovers: list[Recommendation] = []
        for index, quota in enumerate(proportional_quotas(sizes, limit)):
            group = [row for row, owner in zip(candidates, owners) if owner == index]
            picked = diversity_rerank(group, self.item_vectors, limit=quota,
                                      relevance_weight=relevance_weight)
            ranked.extend(picked)
            picked_ids = {row.article_id for row in picked}
            leftovers.extend(row for row in group if row.article_id not in picked_ids)
        leftovers.extend(row for row, owner in zip(candidates, owners) if owner < 0)
        if len(ranked) < limit and leftovers:
            leftovers.sort(key=lambda row: (-row.net_score, row.article_id))
            ranked.extend(diversity_rerank(leftovers, self.item_vectors,
                                           limit=limit - len(ranked),
                                           relevance_weight=relevance_weight))
        ranked.sort(key=lambda row: (-row.final_score, row.article_id))
        return ranked

    def _history_duplicate_finder(
        self, history: Sequence[str],
    ) -> Callable[[Article, SparseVector], str | None]:
        """Return a check for re-published copies of articles the user already read.

        MIND republishes the same story under new IDs (e.g. once under 'news'
        and once under 'finance'). A candidate is a copy when its normalized
        title equals a read article's, or its cosine to a read article is at
        least ``history_duplicate_threshold``. Follow-up stories (cosine about
        0.3-0.7) are kept on purpose: MIND users click them above the base rate.
        """
        threshold = self.history_duplicate_threshold
        read = [article_id for article_id in dict.fromkeys(history)
                if article_id in self.article_by_id]
        if threshold is None or not read:
            return lambda article, vector: None
        titles: dict[tuple[str, ...], str] = {}
        postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for article_id in read:
            key = tuple(preprocess_text(self.article_by_id[article_id].title, self.config))
            if key:
                titles.setdefault(key, article_id)
            for term, weight in self.item_vectors.vector(article_id).items():
                postings[term].append((article_id, weight))

        def find(article: Article, vector: SparseVector) -> str | None:
            key = tuple(preprocess_text(article.title, self.config))
            if key in titles:
                return titles[key]
            dots: dict[str, float] = defaultdict(float)
            for term, weight in vector.items():
                for read_id, read_weight in postings.get(term, ()):
                    dots[read_id] += weight * read_weight
            # Item rows are L2-normalized, so each dot product is a cosine.
            best = max(dots.items(), key=lambda pair: (pair[1], pair[0]), default=None)
            return best[0] if best is not None and best[1] >= threshold else None

        return find

    def compare_diversity(self, user: User, *, top_n: int = 50, top_k: int = 10,
                          as_of: datetime | None = None,
                          candidate_mode: str = "champions",
                          candidate_pool_size: int | None = None,
                          relevance_weight: float = 0.75,
                          score_weights: ScoreWeights | Mapping[str, float] | None = None,
                          half_life_days: float | None = None,
                          multi_interest: bool = False) -> DiversityComparison:
        """Compare net-score top K with diversity reranking from the same top N."""
        if top_n < 1 or top_k < 0 or top_k > top_n:
            raise ValueError("require top_n >= 1 and 0 <= top_k <= top_n")
        candidates = self.recommend(
            user, limit=top_n, as_of=as_of, candidate_mode=candidate_mode,
            candidate_pool_size=candidate_pool_size or top_n, diverse=False,
            score_weights=score_weights, half_life_days=half_life_days,
            multi_interest=multi_interest,
        )
        cold = bool(candidates and candidates[0].mode == "cold_start")
        return build_diversity_comparison(
            candidates, self.item_vectors, top_k=top_k,
            relevance_weight=relevance_weight,
            category_only=(cold and self.cold_start_config.diversify_cold_start_by_category),
        )

    def popular(self, *, limit: int = 10, exclude: set[str] | None = None) -> list[Article]:
        """Reproducible popularity-only baseline, with stable ID tie breaking."""
        excluded = exclude or set()
        if limit <= 0:
            return []
        return heapq.nsmallest(
            limit,
            (article for article in self.articles if article.article_id not in excluded),
            key=lambda article: (-self.click_counts.get(article.article_id, 0), article.article_id),
        )
