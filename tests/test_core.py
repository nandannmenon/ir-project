from datetime import datetime, timedelta
import unittest
import shutil
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from t3rec.models import Article, Recommendation, User
from t3rec.candidates import generate_cluster_candidates
from t3rec.pipeline import Recommender
from t3rec.diversity import compare_diversity, diversity_rerank
from t3rec.profiles import build_interest_profiles, build_user_profile, proportional_quotas
from t3rec.scoring import (ScoreWeights, calculate_net_score, jaccard_similarity,
                           popularity_score, recency_score)
from t3rec.text import TextConfig, preprocess_text
from t3rec.vectorize import build_item_vectors, cosine_similarity
from t3rec.evaluate import (chronological_split, evaluate_impressions,
                            run_evaluation, write_evaluation)
from t3rec.demo_ir import run_demo
from t3rec.ingest import ingest_mind


class CoreScoringTests(unittest.TestCase):
    def test_preprocessing_is_configurable_and_transparent(self):
        self.assertEqual(preprocess_text("The SPACE-rock's 2024 mission!"), ["space-rock's", "mission"])
        self.assertEqual(preprocess_text("a cat", TextConfig(stopwords=frozenset())), ["cat"])

    def test_tfidf_df_idf_and_cosine(self):
        docs = [Article("1", "red apple"), Article("2", "red berry")]
        index = build_item_vectors(docs)
        self.assertEqual(index.document_frequency["red"], 2)
        self.assertGreater(index.idf["apple"], index.idf["red"])
        self.assertGreater(cosine_similarity(index.vector("1"), index.vector("2")), 0)
        self.assertEqual(cosine_similarity({}, index.vector("1")), 0)

    def test_user_profile_weighting_and_consumed(self):
        docs = [Article("1", "red apple"), Article("2", "blue berry")]
        index = build_item_vectors(docs)
        profile = build_user_profile(User("u", ("1", "2")), index, {a.article_id: a for a in docs})
        self.assertEqual(profile.consumed, {"1", "2"})
        self.assertTrue(profile.vector)

    def test_profile_explicit_interaction_weights_and_time_decay(self):
        docs = [Article("1", "space exploration"), Article("2", "football championship")]
        index = build_item_vectors(docs)
        by_id = {article.article_id: article for article in docs}
        old = datetime(2024, 1, 1)
        recent = datetime(2024, 1, 11)
        user = User("u", ("1", "2"), (old, recent))
        weighted = build_user_profile(user, index, by_id, interaction_weights=(0.0, 1.0))
        self.assertGreater(cosine_similarity(weighted.vector, index.vector("2")),
                           cosine_similarity(weighted.vector, index.vector("1")))
        decayed = build_user_profile(user, index, by_id, as_of=recent, half_life_days=10)
        self.assertGreater(cosine_similarity(decayed.vector, index.vector("2")),
                           cosine_similarity(decayed.vector, index.vector("1")))

    def test_set_quality_popularity_and_fusion_scores(self):
        self.assertAlmostEqual(jaccard_similarity({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard_similarity(set(), set()), 0)
        self.assertGreater(popularity_score("x", {"x": 10, "y": 1}), popularity_score("y", {"x": 10, "y": 1}))
        self.assertEqual(calculate_net_score(1, weights={"relevance": 1}), 1)
        article = Article("x", "hello", timestamp=datetime(2024, 1, 1))
        self.assertGreater(recency_score(article, datetime(2024, 1, 2)), recency_score(article, datetime(2024, 2, 1)))

    def test_recommendation_excludes_history_and_explains_terms(self):
        docs = [Article("1", "space planet news", category="science"),
                Article("2", "planet telescope news", category="science"),
                Article("3", "football game result", category="sports")]
        rec = Recommender(docs, click_counts={"2": 2, "3": 20})
        rows = rec.recommend(User("u", ("1",)), limit=2, candidate_mode="all", diverse=False)
        self.assertTrue(rows)
        self.assertNotIn("1", [row.article.article_id for row in rows])
        self.assertIn("top TF-IDF contributors", rows[0].explanation)

    def test_cluster_pruning_generates_bounded_deterministic_candidates(self):
        docs = [
            Article("A", "mars rover science"), Article("B", "mars planet rover"),
            Article("C", "football match sports"), Article("D", "soccer team win"),
            Article("E", "space telescope planet"), Article("F", "music album artist"),
            Article("G", "rock song music"), Article("H", "basketball sports team"),
        ]
        recommender = Recommender(docs, cluster_leader_count=2)
        user = User("u", ("A",))
        self.assertEqual(recommender.cluster_probe_count, 20)

        results = recommender.recommend(
            user, limit=3, candidate_mode="clusters", candidate_pool_size=4, diverse=False
        )
        first_candidates = recommender.last_diagnostics["candidate_ids"]
        recommender.recommend(
            user, limit=3, candidate_mode="clusters", candidate_pool_size=4, diverse=False
        )

        self.assertEqual(recommender.cluster_index.leaders, ("A", "C"))
        self.assertEqual(first_candidates, recommender.last_diagnostics["candidate_ids"])
        self.assertLessEqual(len(first_candidates), 4)
        self.assertTrue(all(row.article_id != "A" for row in results))
        self.assertEqual(recommender.last_diagnostics["candidate_mode"], "clusters")
        query = recommender.item_vectors.vector("A")
        nearest_leader = max(
            recommender.cluster_index.leaders,
            key=lambda leader: cosine_similarity(query, recommender.item_vectors.vector(leader)),
        )
        blocked_cluster = set(recommender.cluster_index.followers[nearest_leader])
        expanded = generate_cluster_candidates(
            recommender.article_by_id, blocked_cluster, profile=query,
            cluster_index=recommender.cluster_index, item_vectors=recommender.item_vectors,
            limit=1,
        )
        self.assertEqual(len(expanded), 1)
        self.assertNotIn(expanded[0].article_id, blocked_cluster)

    def test_impression_evaluation_and_popularity_baseline(self):
        docs = [Article("1", "space planet"), Article("2", "planet telescope"),
                Article("3", "football match")]
        recommender = Recommender(docs, click_counts={"2": 8, "3": 2})
        impression = (User("u", ("1",)), {"2", "3"}, {"2"})
        personalized = evaluate_impressions(recommender, [impression], k=1)
        baseline = evaluate_impressions(recommender, [impression], k=1, baseline=True)
        self.assertEqual(personalized.impressions, 1)
        self.assertAlmostEqual(personalized.p_at_k, 1.0)
        self.assertAlmostEqual(baseline.recall, 1.0)

    def test_candidate_recommendation_cold_start_weight_changes_and_availability(self):
        docs = [
            Article("H", "space rocket science astronomy", category="science"),
            Article("R1", "space rocket science research", category="science"),
            Article("R2", "space science", category="science"),
            Article("P", "football game champion", category="sports"),
            Article("BAD", "", "", category="unknown"),
        ]
        rec = Recommender(docs, click_counts={"P": 100, "R2": 10, "R1": 0},
                          available_article_ids={"H", "R1", "R2", "P", "BAD"})
        user = User("u", ("H",))
        primary = rec.recommend(user, limit=3, diverse=False)
        self.assertTrue(primary)
        self.assertEqual(primary[0].mode, "sparse_history")
        self.assertEqual(primary[0].score_weights["relevance"], 0.50)
        self.assertLess(primary[0].candidate_count, len(docs))
        self.assertEqual(primary[0].rank, 1)
        self.assertTrue(all(row.article_id != "H" for row in primary))
        self.assertNotIn("BAD", rec.article_by_id)
        self.assertIn("cosine relevance", primary[0].explanation)
        popularity_first = rec.recommend(
            user, limit=3, candidate_mode="all", diverse=False,
            score_weights=ScoreWeights(alpha=0.25, beta=0.05, gamma=0.05, delta=0.65),
        )
        self.assertNotEqual([x.article_id for x in primary],
                            [x.article_id for x in popularity_first])
        cold = rec.recommend(User("cold", ()), limit=2, diverse=False)
        self.assertEqual(cold[0].mode, "cold_start")
        self.assertEqual(cold[0].score_weights["relevance"], 0.0)
        self.assertEqual(cold[0].article_id, "P")
        self.assertEqual(cold[0].cosine_score, 0.0)
        cold_cluster = rec.recommend(
            User("cold", ()), limit=2, candidate_mode="clusters",
            candidate_pool_size=2, diverse=False,
        )
        self.assertLessEqual(cold_cluster[0].candidate_count, 2)
        self.assertEqual(rec.last_diagnostics["candidate_mode"], "clusters")

    def test_normal_profile_is_personalized_mode(self):
        docs = [
            Article("H1", "space rocket"), Article("H2", "planet telescope"),
            Article("H3", "astronomy moon"), Article("R", "space astronomy moon"),
            Article("X", "football field"),
        ]
        rec = Recommender(docs, click_counts={"R": 2})
        results = rec.recommend(User("normal", ("H1", "H2", "H3")), diverse=False)
        self.assertTrue(results)
        self.assertEqual(results[0].mode, "personalized")
        self.assertEqual(results[0].score_weights["relevance"], 0.75)
        self.assertGreater(results[0].cosine_score, 0)

    def test_diversity_comparison_tracks_penalties_and_coverage(self):
        docs = [
            Article("A", "space rocket science", category="science"),
            Article("B", "space rocket science", category="science"),
            Article("C", "football match sports", category="sports"),
        ]
        index = build_item_vectors(docs)
        rows = [
            Recommendation(doc, 0.0, 0.0, 0.0, 0.0, 0.0, score, score, "test", original_rank=rank)
            for doc, score, rank in zip(docs, (0.90, 0.89, 0.80), (1, 2, 3))
        ]
        comparison = compare_diversity(rows, index, top_k=2, relevance_weight=0.75)
        self.assertEqual([row.article_id for row in comparison.without_diversity], ["A", "B"])
        self.assertEqual([row.article_id for row in comparison.with_diversity], ["A", "C"])
        self.assertEqual(comparison.baseline_metrics.unique_category_count, 1)
        self.assertEqual(comparison.diversified_metrics.unique_category_count, 2)
        self.assertGreater(comparison.baseline_metrics.average_pairwise_similarity,
                           comparison.diversified_metrics.average_pairwise_similarity)
        full_rerank = diversity_rerank(rows, index, limit=3, relevance_weight=0.75)
        self.assertEqual(full_rerank[1].original_rank, 3)
        self.assertEqual(full_rerank[1].reranked_rank, 2)
        self.assertGreater(full_rerank[2].redundancy_penalty, 0)

    def test_mind_ingestion_builds_and_reuses_deterministic_cache(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "work") as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            shutil.copyfile(repo / "data/sample_mind/news.tsv", raw / "news.tsv")
            shutil.copyfile(repo / "data/sample_mind/behaviors.tsv", raw / "behaviors.tsv")
            output = root / "processed"
            first = ingest_mind(raw, output)
            cache_before = (output / "dataset.json.gz").read_bytes()
            second = ingest_mind(raw, output)
            cache_after = (output / "dataset.json.gz").read_bytes()
            self.assertEqual(first.validation["article_count"], 5)
            self.assertEqual(first.validation["user_count"], 2)
            self.assertEqual(first.validation["impression_count"], 3)
            self.assertEqual(first.validation["interaction_count"], 4)
            self.assertEqual(first.validation["tfidf_nonzero_count"],
                             sum(len(v) for v in first.item_vectors.vectors.values()))
            self.assertGreater(first.validation["vocabulary_size"], 0)
            self.assertGreater(first.validation["tfidf_sparsity"], 0)
            self.assertEqual(first.validation["missing_fields"]["missing_abstract"], 1)
            self.assertEqual(first.validation["missing_fields"]["missing_category"], 1)
            self.assertEqual(first.item_vectors.vectors, second.item_vectors.vectors)
            self.assertEqual(first.impressions, second.impressions)
            recommender = Recommender.from_processed(first)
            self.assertEqual(recommender.item_vectors.vectors, first.item_vectors.vectors)
            self.assertEqual(recommender.click_counts, first.click_counts)
            self.assertEqual(cache_before, cache_after)
            ingest_mind(raw, output, force=True)
            self.assertEqual(cache_before, (output / "dataset.json.gz").read_bytes())

    def test_chronological_evaluation_writes_metrics_and_plots(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / "work") as temp:
            dataset = ingest_mind(repo / "data/sample_mind", Path(temp) / "cache")
            split = chronological_split(dataset.impressions, test_fraction=0.2)
            self.assertLessEqual(max(row.timestamp for row in split.train), split.cutoff)
            self.assertGreater(min(row.timestamp for row in split.test), split.cutoff)
            result = run_evaluation(dataset, test_fraction=0.2, ks=(1, 2))
            self.assertEqual(result.summary["train_impressions"], 2)
            self.assertEqual(result.summary["test_impressions"], 1)
            self.assertEqual(result.summary["evaluated_impressions"], 1)
            self.assertEqual(len(result.aggregate), 6)
            self.assertEqual({row["Model"] for row in result.aggregate},
                             {"A. Popularity baseline", "B. TF-IDF cosine", "C. TF-IDF + Jaccard",
                              "D. TF-IDF + Jaccard + recency/popularity",
                              "E. Full model + diversity reranking",
                              "F. Multi-interest profiles"})
            popularity = next(row for row in result.aggregate
                              if row["Model"] == "A. Popularity baseline")
            # N1003 is clicked only in the held-out row. Train-only popularity
            # must not use that future label to put it first.
            self.assertEqual(popularity["P@1"], 0.0)
            self.assertEqual(popularity["R@2"], 1.0)
            output_dir = Path(temp) / "evaluation"
            write_evaluation(result, output_dir, ks=(1, 2))
            for filename in ("aggregate_metrics.csv", "per_user_metrics.csv", "run_summary.json",
                             "ranking_metrics.svg", "coverage_diversity.svg", "ablation_results.csv",
                             "ablation_results.json", "ablation_plot.svg"):
                self.assertTrue((output_dir / filename).is_file(), filename)
            self.assertEqual(result.summary["tuning_policy"],
                             "Fixed component weights specified before evaluation; no evaluation-set tuning.")
            self.assertEqual(len(result.summary["evaluation_set_sha256"]), 64)
            # The held-out clicked N1003 is excluded from train-only popularity.
            self.assertEqual(dataset.impressions[-1].clicked, ("N1003",))
            trace = StringIO()
            with redirect_stdout(trace):
                run_demo(dataset, "U_SAMPLE_1", k=2, show_candidates=3)
            rendered = trace.getvalue()
            for heading in ("USER PROFILE", "CANDIDATE GENERATION", "ARTICLE SCORING",
                            "FINAL RANKING AND EXPLANATIONS", "DIVERSITY COMPARISON",
                            "SPARSE TF-IDF INSPECTION"):
                self.assertIn(heading, rendered)
            self.assertIn("TF-IDF=", rendered)
            self.assertIn("DF=", rendered)


class ResultQualityFixTests(unittest.TestCase):
    def test_republished_copies_of_read_articles_are_skipped_but_follow_ups_kept(self):
        docs = [
            Article("READ", "Carrier fined for throttling unlimited data plans",
                    "regulator penalty mobile customers", category="finance"),
            Article("COPY", "Carrier fined for throttling unlimited data plans",
                    "a different abstract about the same penalty", category="news"),
            Article("FOLLOW", "Customers seek refunds after carrier throttling fine",
                    "mobile plan refunds", category="news"),
            Article("OTHER", "Local team wins championship game", category="sports"),
        ]
        rec = Recommender(docs)
        ids = [row.article_id for row in rec.recommend(
            User("u", ("READ",)), limit=3, candidate_mode="all", diverse=False)]
        self.assertNotIn("COPY", ids)
        self.assertIn("FOLLOW", ids)
        self.assertEqual(rec.last_diagnostics["history_copies_removed"], [("COPY", "READ")])
        unfiltered = Recommender(docs, history_duplicate_threshold=None)
        self.assertIn("COPY", [row.article_id for row in unfiltered.recommend(
            User("u", ("READ",)), limit=3, candidate_mode="all", diverse=False)])

    def test_cosine_and_jaccard_are_scaled_to_the_best_candidate(self):
        docs = [Article("H", "mars rover science mission"), Article("A", "mars rover landing"),
                Article("B", "mars mission budget"), Article("C", "football final score")]
        rec = Recommender(docs, click_counts={"C": 50})
        rows = rec.recommend(User("u", ("H",)), limit=3, candidate_mode="all", diverse=False)
        self.assertAlmostEqual(max(row.relevance_normalized for row in rows), 1.0)
        maxima = rec.last_diagnostics["normalization_max"]
        for row in rows:
            self.assertAlmostEqual(row.relevance_normalized, row.relevance / maxima["relevance"])
            weights = row.score_weights
            expected = (weights["relevance"] * row.relevance_normalized +
                        weights["jaccard"] * row.jaccard_normalized +
                        weights["recency"] * row.recency +
                        weights["popularity"] * row.popularity) / sum(weights.values())
            self.assertAlmostEqual(row.net_score, expected)
        cosine_only = rec.recommend(User("u", ("H",)), limit=3, candidate_mode="all", diverse=False,
                                    score_weights=ScoreWeights(1, 0, 0, 0))
        self.assertEqual([row.article_id for row in cosine_only],
                         [row.article_id for row in sorted(cosine_only, key=lambda r: -r.relevance)])

    def test_recency_weight_is_dropped_without_timestamps(self):
        undated = Recommender([Article("H", "space rocket"), Article("A", "space telescope"),
                               Article("B", "rocket launch")])
        row = undated.recommend(User("u", ("H",)), limit=1, candidate_mode="all", diverse=False)[0]
        self.assertEqual(row.score_weights["recency"], 0.0)
        self.assertFalse(undated.last_diagnostics["recency_available"])
        self.assertIn("recency n/a", row.explanation)
        day = datetime(2024, 1, 10)
        dated = Recommender([Article("H", "space rocket", timestamp=day),
                             Article("A", "space telescope", timestamp=day),
                             Article("B", "rocket launch", timestamp=day - timedelta(days=30))])
        row = dated.recommend(User("u", ("H",)), limit=1, candidate_mode="all", diverse=False,
                              score_weights=ScoreWeights())[0]
        self.assertEqual(row.score_weights["recency"], ScoreWeights().gamma)
        self.assertTrue(dated.last_diagnostics["recency_available"])

    def test_cluster_followers_attach_to_several_leaders_and_probe_count_bounds_search(self):
        docs = [Article("A", "mars rover science"), Article("B", "mars rover football"),
                Article("C", "football match sports"), Article("E", "space telescope planet")]
        multi = Recommender(docs, cluster_leader_count=2, cluster_followers_per_item=2)
        self.assertEqual(multi.cluster_index.leaders, ("A", "C"))
        # B shares terms with both leaders, so with b1=2 it is listed under both.
        self.assertIn("B", multi.cluster_index.followers["A"])
        self.assertIn("B", multi.cluster_index.followers["C"])
        single = Recommender(docs, cluster_leader_count=2, cluster_followers_per_item=1)
        self.assertEqual(sum("B" in members for members in single.cluster_index.followers.values()), 1)

        def probe(consumed, probe_count):
            return [article.article_id for article in generate_cluster_candidates(
                multi.article_by_id, consumed, profile=multi.item_vectors.vector("A"),
                cluster_index=multi.cluster_index, item_vectors=multi.item_vectors,
                limit=1, probe_count=probe_count)]

        self.assertEqual(probe({"A"}, 1), ["B"])
        # The nearest cluster is exhausted, so the next leader is probed to fill the list.
        self.assertEqual(probe({"A", "B"}, 1), ["C"])
        with self.assertRaises(ValueError):
            probe({"A"}, 0)


class MultiInterestTests(unittest.TestCase):
    DOCS = [
        Article("S1", "rocket launch orbit satellite", category="science"),
        Article("S2", "rocket orbit mission astronauts", category="science"),
        Article("S3", "satellite orbit rocket telescope", category="science"),
        Article("K1", "pasta sauce recipe garlic", category="food"),
        Article("C1", "rocket orbit launch window", category="science"),
        Article("C2", "satellite rocket orbit mission", category="science"),
        Article("C3", "orbit telescope rocket satellite", category="science"),
        Article("C4", "astronauts rocket launch orbit", category="science"),
        Article("C5", "garlic pasta recipe dinner", category="food"),
    ]

    def test_history_splits_into_leader_clustered_interests(self):
        index = build_item_vectors(self.DOCS)
        by_id = {article.article_id: article for article in self.DOCS}
        interests = build_interest_profiles(User("u", ("S1", "K1", "S2", "S3")), index, by_id,
                                            threshold=0.1, max_interests=5)
        self.assertEqual([interest.members for interest in interests],
                         [("S1", "S2", "S3"), ("K1",)])
        self.assertEqual([interest.category for interest in interests], ["science", "food"])
        capped = build_interest_profiles(User("u", ("S1", "K1", "S2", "S3")), index, by_id,
                                         threshold=0.1, max_interests=1)
        self.assertEqual(len(capped), 1)
        self.assertEqual(len(capped[0].members), 4)

    def test_proportional_quotas_use_largest_remainder(self):
        self.assertEqual(proportional_quotas([12, 2], 10), [9, 1])
        self.assertEqual(proportional_quotas([1, 1, 1], 2), [1, 1, 0])
        self.assertEqual(sum(proportional_quotas([5, 3, 2], 7)), 7)

    def test_minority_interest_gets_a_proportional_slot(self):
        rec = Recommender(self.DOCS, history_duplicate_threshold=None)
        user = User("u", ("S1", "S2", "S3", "K1"))
        single = rec.recommend(user, limit=4, candidate_mode="all", diverse=False)
        self.assertNotIn("C5", [row.article_id for row in single])
        multi = rec.recommend(user, limit=4, candidate_mode="all", diverse=False,
                              multi_interest=True)
        self.assertIn("C5", [row.article_id for row in multi])
        self.assertEqual(rec.last_diagnostics["interest_quotas"], [3, 1])
        food = next(row for row in multi if row.article_id == "C5")
        self.assertIn("matches your interest", food.explanation)
        self.assertEqual(food.interest, rec.last_diagnostics["interests"][1]["label"])
        scores = [row.net_score for row in multi]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_diversity_reranking_keeps_interest_quotas(self):
        rec = Recommender(self.DOCS, history_duplicate_threshold=None)
        rows = rec.recommend(User("u", ("S1", "S2", "S3", "K1")), limit=4, candidate_mode="all",
                             diverse=True, multi_interest=True)
        self.assertEqual(len(rows), 4)
        self.assertIn("C5", [row.article_id for row in rows])
        finals = [row.final_score for row in rows]
        self.assertEqual(finals, sorted(finals, reverse=True))

    def test_article_matching_no_interest_is_unlabeled(self):
        docs = [*self.DOCS, Article("X", "football final score", category="sports")]
        rec = Recommender(docs, history_duplicate_threshold=None)
        rows = rec.recommend(User("u", ("S1", "S2", "S3", "K1")), limit=8, candidate_mode="all",
                             diverse=False, multi_interest=True)
        unrelated = next(row for row in rows if row.article_id == "X")
        self.assertEqual(unrelated.interest, "")
        self.assertNotIn("matches your interest", unrelated.explanation)


if __name__ == "__main__":
    unittest.main()
