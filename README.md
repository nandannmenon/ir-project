# T3 News Recommender — CSD358 IR hackathon, track T3

An inspectable content-based news recommender built on Microsoft's MIND dataset. Articles are documents, a reader's history is the query, and sparse TF-IDF cosine similarity is the main relevance signal. Around it sit champion lists and cluster pruning for candidate generation, Jaccard, popularity and recency as secondary signals in a net score, heap-based top-K selection, diversity reranking (MMR), a "content seen?" filter for re-published copies of articles already read, and explanations built from the terms that contribute most to each cosine.

**What is new: multi-interest profiles.** Instead of averaging a reader's whole history into one query vector, leader–follower clustering (the idea behind cluster pruning, applied to one reader's history) splits it into up to three interests. Each interest retrieves its own champion-list candidates, every article is scored against its best-matching interest, and result slots are shared across interests in proportion to how much the reader read about each. A reader whose history is 40% one story no longer gets a list that is 100% that story, and every recommendation names the interest it serves.

## Requirements

- Python 3.10 or newer (tested on Windows 11 with Python 3.12 and 3.14).
- Standard library only — nothing to `pip install`.
- For the MIND experiments: about 2 GB of free memory and the MINDsmall training split (see [Get the MIND data](#get-the-mind-data)).

Run every command from the repository root with `src` on the Python path:

| Shell | Once per terminal | Python command |
| --- | --- | --- |
| Windows PowerShell | `$env:PYTHONPATH = "src"` | `python` |
| macOS / Linux | `export PYTHONPATH=src` | `python3` |

The commands below are written with `python`; use `python3` on macOS/Linux if `python` is not Python 3.

## Quick start (no download needed)

```text
python -m unittest discover -s tests
python -m t3rec.cli --fixture data/synthetic.json --user U1 --top-k 3
python -m t3rec.demo_ir --user U_SAMPLE_1 --k 3
python -m t3rec.web --port 8765
```

Open http://127.0.0.1:8765, pick reader **U1**, and click **Run recommendations**. Turn on **Multi-interest profile** to see the reader's interests and which one each result serves. `data/synthetic.json` is a hand-written 10-article toy catalog and `data/sample_mind/` is a 5-article sample in MIND's TSV layout; both are smoke tests, too small for quality conclusions.

More CLI options on the toy catalog: `--all-users --cold-start --compare-weights` runs every reader plus a cold-start reader under two weight settings, and `--compare-diversity` compares plain top-K with diversity-reranked top-K from the same candidate pool.

## Get the MIND data

1. Download **MINDsmall_train** from the [official Microsoft page](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news) and review the Microsoft Research License Terms.
2. Extract it so these files exist:
   ```text
   data/raw/MINDsmall_train/news.tsv
   data/raw/MINDsmall_train/behaviors.tsv
   ```
3. Build the processed cache (cleaned records and TF-IDF rows; built once, then reused):
   ```text
   python -m t3rec.ingest_cli --raw-dir data/raw/MINDsmall_train --output-dir data/processed/MINDsmall_train
   ```

The dataset is not in this repository: `data/raw/`, `data/processed/` and `outputs/**/cache/` are Git-ignored. Do not commit MIND files. Ingestion fingerprints both TSV files and the preprocessing settings, so a matching cache is reused and changed inputs trigger a rebuild (`--force` rebuilds explicitly).

## Run on MIND

**Step-by-step IR trace** (profile terms, interests, champion candidates, every score component, explanations, diversity comparison, a postings list with DF/IDF):

```text
python -m t3rec.demo_ir --raw-dir data/raw/MINDsmall_train --cache-dir data/processed/MINDsmall_train --user U13740 --k 5
python -m t3rec.demo_ir --raw-dir data/raw/MINDsmall_train --cache-dir data/processed/MINDsmall_train --user U8402 --k 5 --multi-interest
```

The trace uses the user's latest impression and that row's pre-impression history. `--term TERM` inspects one vocabulary term's postings; `--show-candidates N` prints more scored candidates. User IDs are in the second column of `behaviors.tsv`.

**Browser workbench on MIND** (ready after about 20 seconds; the reader list shows the first 300 users by ID):

```text
python -m t3rec.web --raw-dir data/raw/MINDsmall_train --cache-dir data/processed/MINDsmall_train --port 8765
```

**Evaluation** (about 12 minutes; writes CSV tables, `run_summary.json` and three SVG charts to the output folder):

```text
python -m t3rec.eval_cli --raw-dir data/raw/MINDsmall_train --output-dir outputs/evaluation/MINDsmall_train --test-fraction 0.2 --ks 5,10
```

**Validation tuning** of the multi-interest settings (18 settings on 4,000 validation impressions took about 100 minutes on a laptop; `--impressions 1000` gives a quicker, noisier check). It uses only the training period, never the test impressions:

```text
python -m t3rec.tune_cli --cache-dir data/processed/MINDsmall_train
```

## Results

MINDsmall_train, chronological split at 2019-11-13 20:36:19: 125,572 training impressions; 31,290 held-out impressions from 20,654 users. Each held-out impression's shown articles (40 on average, 1.54 clicks) are ranked; metrics are macro-averaged per user. Full tables are in `outputs/evaluation/MINDsmall_train/`.

| Model | P@5 | P@10 | R@10 | NDCG@5 | NDCG@10 | Category diversity@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random order (reference) | 0.075 | 0.060 | 0.485 | 0.205 | 0.265 | – |
| A. Popularity baseline | 0.079 | 0.063 | 0.506 | 0.226 | 0.286 | 0.663 |
| B. TF-IDF cosine | 0.099 | 0.073 | 0.560 | 0.286 | 0.343 | 0.629 |
| C. + Jaccard | 0.101 | 0.073 | 0.565 | 0.291 | 0.347 | 0.615 |
| D. + recency/popularity | 0.101 | 0.074 | 0.567 | 0.292 | **0.348** | 0.615 |
| E. D + diversity reranking | 0.101 | 0.074 | **0.568** | 0.292 | **0.348** | 0.620 |
| F. Multi-interest profiles | **0.102** | 0.073 | 0.565 | **0.293** | 0.347 | **0.623** |
| Best possible (reference) | 0.296 | 0.151 | 1.000 | 1.000 | 1.000 | – |

Paired differences across the 20,654 users (95% confidence intervals):

- **B vs A:** NDCG@10 +0.057 [+0.053, +0.062]. Content matching clearly beats popularity, which is barely above random because 69% of held-out clicks go to articles with no training clicks.
- **C vs B:** +0.004 [+0.003, +0.004]. **D vs C:** +0.001 [+0.001, +0.002], after the freshness proxy was changed to first exposure (with the earlier last-exposure proxy, D was no better than C).
- **E vs D:** no accuracy difference [−0.0002, +0.0008]; category diversity +0.005 [+0.005, +0.006].
- **F vs D:** no accuracy difference on NDCG@10 [−0.002, +0.001], NDCG@5 or P@5; P@10 −0.0005 [−0.0009, −0.0000]; category diversity +0.008 [+0.007, +0.010], and the lowest pairwise similarity of all models (0.011).

R@k is high even for random order because the median impression shows only 26 articles, so P@k and NDCG are the more informative metrics. MIND clicks are implicit feedback, and this split is not the official MIND test split, so these numbers are not comparable with the MIND leaderboard.

Supplementary checks on full-catalog recommendations (top 10 from all 51k articles for 200 held-out users, champion lists): multi-interest profiles cover 99% of a reader's interests with at least two read articles versus 74% for a single profile, the largest interest's share of the list falls from 80% to 61%, and near-duplicate pairs fall from 4.1 to 2.4. Champion lists reproduce 79% of exact full-catalog top-10 results in 31–42 ms per query versus 1.5–2 s for exact scoring; cluster pruning also reaches 79% but takes 290–380 ms (timings varied between runs on a shared laptop; the ratios held). Held-out clicks found in the top 10 are too few at this sample size (about five) to separate the methods.

## What works and what is planned

**Works now**

- TF-IDF indexing of 51,278 MIND articles (log TF, smoothed IDF, L2-normalized sparse rows) with a deterministic cache.
- User profiles as query vectors, single or multi-interest; cold-start and short-history modes.
- Three candidate generators: exact full-catalog scoring, champion lists, cluster pruning (√N farthest-first leaders, b1=2, b2=20).
- Net score from max-normalized cosine and Jaccard plus popularity g(d) and recency; heap-based top-K; MMR diversity reranking; "content seen?" filtering of re-published copies; term-level explanations.
- Browser workbench, terminal IR trace, chronological evaluation with six models, validation tuning script, 22 unit tests.

**Planned / known limitations**

- No stemming or lemmatization ("recipe" does not match "recipes") and a short stopword list, so common words such as "new" can create weak matches.
- Interest labels come from the top centroid terms and are sometimes uninformative (for example "before · older · hold" for a mixed group of news stories). Text-only clustering of news histories is weak because most article pairs share few terms; using subcategories or entity embeddings to define interests is the next step.
- MIND has no publication dates. The evaluation uses a first-exposure proxy; the workbench and trace have no recency signal at all.
- Popularity uses training-period counts, which say nothing about the 69% of test-period clicks that go to articles with no training clicks. Short-window (last-hours) popularity is a natural extension.
- Content-only lexical model: no BM25, dense retrieval or learning-to-rank yet. These are the planned course-project extensions, along with evaluating on the official MIND dev split.

## Chronological evaluation protocol

The evaluation uses one global chronological timestamp cutoff. Training impressions are at or before the cutoff; held-out impressions are strictly later (timestamp ties stay on the training side). Interactions without parseable timestamps are omitted and counted in `run_summary.json`. Each held-out impression contributes only its recorded pre-impression history and its own displayed articles. Clicked labels are relevant items only; already consumed items, unknown article IDs, and unshown clicks cannot become recommendations or positives. Shown articles that are re-published copies of something the user already read are removed for every model, including the baseline. Popularity is calculated from training clicks only. MIND `news.tsv` has no publication timestamps, so the recency feature uses a proxy the system could know at the cutoff: the first training impression that showed the article, the start of the training log for articles only seen in reading histories, and the cutoff itself for articles never seen before it (new in the test period). It should not be described as true publication freshness.

The evaluation is a six-model ablation. A is the training-click popularity baseline; B uses cosine alone; C uses cosine+Jaccard; D adds recency and popularity; E applies diversity reranking to D; F is D with multi-interest profiles. All variants run on the same held-out impression loop, candidate lists, and relevance labels; `evaluation_set_sha256` in the JSON fingerprints that shared set. Metrics are calculated per impression, averaged within each user, then macro-averaged across users so highly active users do not dominate. Precision divides by K even when fewer than K candidates are available. Coverage is distinct recommended articles over the valid indexed catalog; category diversity is the fraction of nonempty categories represented in the returned list. Mean pairwise cosine is also emitted (lower means less lexical redundancy). Binary NDCG is included as an additional metric.

The component weights are fixed before evaluation and are not selected by held-out results. The multi-interest settings (at most 3 interests, joining threshold 0.10) were chosen with `t3rec.tune_cli` on the latest 12.5% of the training period, before the test impressions were scored. Cosine and Jaccard are divided by their maximum over each impression's candidates, then fused as a weighted mean, dividing by the sum of active weights:

| Variant | Cosine | Jaccard | Recency | Popularity | Diversity | Profile |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A. Popularity baseline | — | — | — | train-click count sort | None | — |
| B. TF-IDF cosine | 1.00 | 0.00 | 0.00 | 0.00 | None | single |
| C. TF-IDF + Jaccard | 0.90 | 0.10 | 0.00 | 0.00 | None | single |
| D. TF-IDF + Jaccard + recency/popularity | 0.75 | 0.10 | 0.05 | 0.10 | None | single |
| E. Full model + diversity reranking | 0.75 | 0.10 | 0.05 | 0.10 | λ=0.75 | single |
| F. Multi-interest profiles | 0.75 | 0.10 | 0.05 | 0.10 | None | ≤3 interests |

The command writes `aggregate_metrics.csv`, `per_user_metrics.csv`, `run_summary.json`, `ranking_metrics.svg`, `coverage_diversity.svg`, `ablation_results.csv`, `ablation_results.json`, and `ablation_plot.svg`. The aggregate table includes P/R/NDCG at every requested K, coverage, category diversity, and average pairwise cosine at the largest K. MIND's future clicks are implicit feedback rather than manually judged relevance, so no extra "human relevance" labels are claimed.

The pipeline parses article IDs, title, abstract, category, subcategory, URL, and title/abstract entity fields; parses impression timestamps, pre-impression histories, shown candidates, and click labels; counts missing fields and unknown article references; builds the sparse TF-IDF rows; and logs a validation summary. It collects no data beyond the anonymized identifiers and interactions already present in MIND. The timestamp on a MIND behavior row describes the impression, not each historical click, so it is not used as a history interaction time.

## IR choices

- Tokenization and stopword removal are explicit in `text.py`; stemming is optional via a caller-provided function and disabled by default.
- Article rows are sparse term-to-weight mappings. TF uses `1 + log(count)`, smoothed IDF is `log((1 + N)/(1 + df)) + 1`, and each row is L2 normalized.
- User query: `normalize(sum_i interaction_weight_i * exp(-ln(2)*age_i/half_life) * article_tfidf_i)`. Clicks default to weight 1; explicit per-interaction weights and optional time decay are supported.
- Multi-interest profiles (`profiles.build_interest_profiles`): the history is visited in reading order; each article joins the interest whose centroid it is most similar to if the cosine is at least 0.10, otherwise it leads a new interest, up to 3. Each interest's champion lists supply candidates; an article's relevance is its cosine to the best-matching interest; top-K slots are split across interests by the largest-remainder method in proportion to interest size, and when diversity is on, MMR runs within each interest so it cannot undo the split. Articles that share no term with any interest only fill slots no interest could fill. Validation on the training period showed more interests lowering NDCG (3 → 0.379, 5 → 0.374 against 0.382 for a single profile) while the joining threshold made no difference, because most news articles share almost no terms.
- Champion lists retrieve articles with high TF-IDF weights for the user's strongest terms. Cluster pruning starts with the lexicographically first article, then uses deterministic farthest-first cosine selection for the remaining √N leaders: each new leader is the article least similar to its nearest selected leader. It attaches every other article to its `b1=2` nearest leaders (`cluster_followers_per_item`); a query probes the `b2=20` leaders nearest the profile (`cluster_probe_count`) and exactly scores a bounded shortlist of their followers. Sparse similarities use inverted indexes rather than dense pairwise comparisons. On 100 evenly spaced MIND held-out impression profiles, for full-catalog cosine retrieval with a 100-item shortlist, b1=2/b2=20 overlapped 78.1% of exact cosine top 10, versus 64.5% for b1=2/b2=10; shortlist generation took 31.8 versus 18.5 seconds across the 100 queries. This is a preliminary retrieval-only sample, not an offline recommendation-quality result; the full-catalog evaluation uses displayed-click labels and cannot measure quality for unshown articles. Both retrieval modes are selectable; `candidate_mode="all"` remains available for evaluation. Cold-start users use a deterministic quality-ranked candidate fallback.
- Jaccard compares typed sets of user-interest terms and category/subcategory features with each item's terms and zones. It remains secondary to cosine.
- The default weighted mean is `0.75*cosine + 0.10*jaccard + 0.05*recency + 0.10*popularity`. Raw MIND cosines are small (top candidates typically 0.1–0.4) while popularity spans `[0,1]`, so cosine and Jaccard are first divided by their maximum over the scored candidate set (`normalize_query_components`); the weights then compare like with like. Results expose both the raw and the scaled values (`relevance`/`relevance_normalized`, `jaccard`/`jaccard_normalized`). Recency is exponential with a 14-day half-life; popularity is `log1p(clicks)` divided by the maximum, a static quality score g(d). `ScoreWeights` exposes each weight and validates them.
- Recency needs article timestamps. When no candidate has one (always the case for MIND `news.tsv` outside the evaluation), the recency weight is set to 0 for that request, the result's `score_weights` show it, and explanations say "recency n/a". On the validation slice, a 1-day half-life changed NDCG@10 by only +0.001 over 14 days, so the default was kept.
- MIND republishes the same story under new IDs (for example once under `news` and once under `finance`). A candidate whose normalized title equals an already-read article's title, or whose cosine to one is at least `history_duplicate_threshold=0.9`, is skipped during top-K selection and listed in `last_diagnostics["history_copies_removed"]` (a "content seen?" check against the reading history). The threshold is deliberately high: on 5,000 MIND impressions, shown articles with cosine 0.3–0.8 to something already read were clicked at 7–16% versus a 4% base rate, so related follow-up stories are kept.
- A heap selects top-K candidates (heapify, then pop until K non-copies are found; explanations are built only for popped items) before optional greedy diversity reranking. Results expose both `net_score` and the diversity-adjusted `final_score`.
- Diversity uses a greedy maximal-marginal-relevance rule: `adjusted = λ * net_score - (1-λ) * max_cosine_to_selected`, with configurable `λ` (`diversity_relevance_weight`, default 0.75). The result records raw redundancy and its weighted penalty. `Recommender.compare_diversity` evaluates ordinary net-score top K and reranked top K from the same top-N candidates and reports category coverage plus mean pairwise cosine.
- Each recommendation reports its mode: `cold_start` for zero usable clicks (quality-only ranking), `sparse_history` for one or two clicks (profile plus stronger quality weights), and `personalized` for longer histories. Sparse mode uses 50% cosine, 10% Jaccard, 15% recency, and 25% popularity by default; personalized mode uses the standard weights above. Cold start uses 30% recency and 70% popularity and can diversify by category without using user-content similarity.
- Explanations use shared terms and displayed score signals, without an LLM.

## Code map

| IR concept | Where |
| --- | --- |
| Tokenization, stopwords, normalization | `src/t3rec/text.py` |
| TF-IDF rows, DF/IDF, cosine | `src/t3rec/vectorize.py` |
| Document parsing, cache, validation | `src/t3rec/ingest.py`, `src/t3rec/data.py` |
| Query vector, multi-interest clustering, slot quotas | `src/t3rec/profiles.py` |
| Champion lists, cluster pruning | `src/t3rec/candidates.py` |
| Jaccard, popularity g(d), recency, net score | `src/t3rec/scoring.py` |
| Heap top-K, copy filter, interest quotas, pipeline | `src/t3rec/pipeline.py` |
| MMR diversity | `src/t3rec/diversity.py` |
| Explanations | `src/t3rec/explain.py` |
| Evaluation and tuning | `src/t3rec/evaluate.py`, `src/t3rec/eval_cli.py`, `src/t3rec/tune_cli.py` |
| Interfaces | `src/t3rec/web.py` (+ `web/`), `src/t3rec/demo_ir.py`, `src/t3rec/cli.py` |

## Data credit

MIND: Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. "MIND: A Large-scale Dataset for News Recommendation." ACL 2020. Used under the Microsoft Research License Terms; not redistributed here. `data/synthetic.json` and `data/sample_mind/` are hand-written toy data.
