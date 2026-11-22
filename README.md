# T3 News Recommender 

An inspectable content-based news recommender built on Microsoft's MIND dataset. Articles are documents, a reader's history is the query, and sparse TF-IDF cosine similarity is the main relevance signal. Around it sit champion lists and cluster pruning for candidate generation, Jaccard, popularity and recency as secondary signals in a net score, heap-based top-K selection, diversity reranking (MMR), a "content seen?" filter for re-published copies of articles already read, and explanations built from the terms that contribute most to each cosine.

**What is new: multi-interest profiles.** Instead of averaging a reader's whole history into one query vector, leader–follower clustering (the idea behind cluster pruning, applied to one reader's history) splits it into up to three interests. Each interest retrieves its own champion-list candidates, every article is scored against its best-matching interest, and result slots are shared across interests in proportion to how much the reader read about each. A reader whose history is 40% one story no longer gets a list that is 100% that story, and every recommendation names the interest it serves.

**Also added: entities and learning to rank.** MIND tags articles with Wikidata entities and ships an embedding per entity. Each article gets a centred entity vector, interests are labelled by the entity names their articles share ("Gordon Hayward", "Ukraine · Donald Trump"), and entity similarity is one of eight features of a pairwise learning-to-rank model that replaces the hand-set fusion weights with weights learned from training-period clicks.

## Setup (Windows and macOS)

Everything runs on Python's standard library: there is nothing to `pip install`. Each step shows the Windows (PowerShell) command first, then the macOS (Terminal) one.

### 1. Install Python 3.10+ and Git

- **Windows:** install Python from [python.org](https://www.python.org/downloads/) (tick **"Add python.exe to PATH"** in the installer) and Git from [git-scm.com](https://git-scm.com/download/win).
- **macOS:** install Python from [python.org](https://www.python.org/downloads/) or with Homebrew (`brew install python`). Git comes with the Xcode command-line tools: run `xcode-select --install` if `git --version` fails.

Check both work:

```powershell
# Windows
python --version
git --version
```

```bash
# macOS
python3 --version
git --version
```

### 2. Get the code

```powershell
# Windows
git clone https://github.com/nandannmenon/ir-project.git
cd ir-project
```

```bash
# macOS
git clone https://github.com/nandannmenon/ir-project.git
cd ir-project
```

Run every later command from this `ir-project` folder.

### 3. Point Python at the code

Do this **in every new terminal window** before running anything; without it Python reports `No module named 't3rec'`.

```powershell
# Windows
$env:PYTHONPATH = "src"
```

```bash
# macOS
export PYTHONPATH=src
```

### 4. Check it works (no download needed)

```powershell
# Windows
python -m unittest discover -s tests
python -m t3rec.web --port 8765
```

```bash
# macOS
python3 -m unittest discover -s tests
python3 -m t3rec.web --port 8765
```

All tests should pass. Then open http://127.0.0.1:8765, pick reader **U1**, and click **Run recommendations**; press Ctrl+C in the terminal to stop the server. This uses `data/synthetic.json`, a hand-written 10-article toy catalog. `data/sample_mind/` is a 5-article sample in MIND's format; both are smoke tests only.

More options on the toy catalog (`python -m t3rec.cli --fixture data/synthetic.json --user U1 --top-k 3`): `--all-users --cold-start --compare-weights` runs every reader plus a cold-start reader under two weight settings, and `--compare-diversity` compares plain top-K with diversity-reranked top-K from the same candidate pool.

### 5. Get the MIND data

The dataset is not in this repository: its license does not allow redistribution, and it is too large for GitHub.

1. Download **MINDsmall_train** from the [official Microsoft page](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news) and review the Microsoft Research License Terms.
2. Extract it into `data/raw/MINDsmall_train` (these commands assume the zip is in your Downloads folder):

   ```powershell
   # Windows
   Expand-Archive -Path "$HOME\Downloads\MINDsmall_train.zip" -DestinationPath data\raw\MINDsmall_train
   Get-ChildItem data\raw\MINDsmall_train
   ```

   ```bash
   # macOS
   mkdir -p data/raw/MINDsmall_train
   unzip ~/Downloads/MINDsmall_train.zip -d data/raw/MINDsmall_train
   ls data/raw/MINDsmall_train
   ```

3. Check that these three files sit **directly** in `data/raw/MINDsmall_train`:

   ```text
   data/raw/MINDsmall_train/news.tsv
   data/raw/MINDsmall_train/behaviors.tsv
   data/raw/MINDsmall_train/entity_embedding.vec
   ```

   Some copies of the zip contain their own `MINDsmall_train` folder. If you see `data/raw/MINDsmall_train/MINDsmall_train/news.tsv`, move the files up one level (or extract into `data/raw` instead). `entity_embedding.vec` is needed for the entity features: without it the system still runs, but interest labels fall back to top words and the learning-to-rank results will not match this README.

### 6. Build the processed cache (once)

This cleans the articles and builds the TF-IDF index; later runs reuse it.

```powershell
# Windows
python -m t3rec.ingest_cli --raw-dir data/raw/MINDsmall_train --output-dir data/processed/MINDsmall_train
```

```bash
# macOS
python3 -m t3rec.ingest_cli --raw-dir data/raw/MINDsmall_train --output-dir data/processed/MINDsmall_train
```

`data/raw/`, `data/processed/` and `outputs/**/cache/` are Git-ignored: never commit MIND files. The cache records a fingerprint of both TSV files and the preprocessing settings, so changed inputs trigger a rebuild (`--force` rebuilds explicitly).

### 7. Run the system on MIND

```powershell
# Windows
python -m t3rec.web --raw-dir data/raw/MINDsmall_train --cache-dir data/processed/MINDsmall_train --port 8765
```

```bash
# macOS
python3 -m t3rec.web --raw-dir data/raw/MINDsmall_train --cache-dir data/processed/MINDsmall_train --port 8765
```

Wait about 25 seconds for `T3 recommender workbench: http://127.0.0.1:8765`, then open that address. The **Multi-interest profile**, **Entity interest names** and **Learned ranking (LTR)** toggles switch the add-ons on and off; the learned model is read from `outputs/evaluation/MINDsmall_train/run_summary.json` (or `--ltr-model PATH`). The next section lists the other commands (terminal trace, evaluation, tuning); on macOS, write `python3` wherever they say `python`.

**Common problems**

- `No module named 't3rec'`: you are not in the `ir-project` folder, or step 3 was not run in this terminal.
- `python` not found on macOS: use `python3`.
- `Address already in use`: another copy of the server is running; stop it with Ctrl+C or pick another port, e.g. `--port 8766`.

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

**Validation tuning** uses only the training period, never the test impressions. `--grid text` (the default) sweeps the recency half-life and word-only interest settings (18 settings on 4,000 validation impressions took about 100 minutes on a laptop); `--grid entity` sweeps the entity weight and threshold for entity-aware interests (11 settings on 2,000 impressions, about 25 minutes). `--impressions 1000` gives a quicker, noisier check:

```text
python -m t3rec.tune_cli --cache-dir data/processed/MINDsmall_train
python -m t3rec.tune_cli --cache-dir data/processed/MINDsmall_train --grid entity --impressions 2000
```

The evaluation trains the learning-to-rank model itself, on 6,000 impressions sampled from the latest 12.5% of the training period (`--ltr-impressions` changes the sample size); the learned weights are written to `run_summary.json`.

## Results

MINDsmall_train, chronological split at 2019-11-13 20:36:19: 125,572 training impressions; 31,290 held-out impressions from 20,654 users. Each held-out impression's shown articles (40 on average, 1.54 clicks) are ranked; metrics are macro-averaged per user. Full tables are in `outputs/evaluation/MINDsmall_train/`.

| Model | P@5 | P@10 | R@10 | NDCG@5 | NDCG@10 | Category diversity@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random order (reference) | 0.075 | 0.060 | 0.485 | 0.205 | 0.265 | – |
| A. Popularity baseline | 0.079 | 0.063 | 0.506 | 0.226 | 0.286 | 0.663 |
| B. TF-IDF cosine | 0.099 | 0.073 | 0.560 | 0.286 | 0.343 | 0.629 |
| C. + Jaccard | 0.101 | 0.073 | 0.565 | 0.291 | 0.347 | 0.615 |
| D. + recency/popularity | 0.101 | 0.074 | 0.567 | 0.292 | 0.348 | 0.615 |
| E. D + diversity reranking | 0.101 | 0.074 | 0.568 | 0.292 | 0.348 | 0.620 |
| F. Multi-interest profiles | 0.102 | 0.073 | 0.565 | 0.293 | 0.347 | **0.623** |
| G. Entity-aware multi-interest | 0.102 | 0.073 | 0.565 | 0.294 | 0.348 | **0.623** |
| H. Learning-to-rank | **0.109** | **0.078** | **0.599** | **0.317** | **0.373** | 0.562 |
| Best possible (reference) | 0.296 | 0.151 | 1.000 | 1.000 | 1.000 | – |

Paired differences across the 20,654 users (95% confidence intervals):

- **B vs A:** NDCG@10 +0.057 [+0.053, +0.062]. Content matching clearly beats popularity, which is barely above random because 69% of held-out clicks go to articles with no training clicks.
- **C vs B:** +0.004 [+0.003, +0.004]. **D vs C:** +0.001 [+0.001, +0.002], after the freshness proxy was changed to first exposure (with the earlier last-exposure proxy, D was no better than C).
- **E vs D:** no accuracy difference [−0.0002, +0.0008]; category diversity +0.005 [+0.005, +0.006].
- **F vs D:** no accuracy difference on NDCG@10 [−0.002, +0.001], NDCG@5 or P@5; P@10 −0.0005 [−0.0009, −0.0000]; category diversity +0.008 [+0.007, +0.010], and the lowest pairwise similarity of all models (0.011).
- **G vs F:** no accuracy difference on NDCG@10 [−0.001, +0.002]; grouping interests with entity similarity changes neither accuracy nor category diversity, so it is not the default.
- **Entity signal inside H:** retraining H without entity similarity (same split, training slice and seed) lowers NDCG@10 from 0.373 to 0.370 (−0.004 [−0.006, −0.002]), NDCG@5 by 0.005 and category diversity by 0.022 (all significant): entities contribute about 15% of H's gain over D, although grouping interests with them (G) adds nothing.
- **H vs D:** NDCG@10 +0.025 [+0.023, +0.028], P@5 +0.008 [+0.007, +0.009]: learned weights beat the hand-set ones, the largest gain after B vs A. The learned model leans most on entity similarity, recency and popularity (standardized weights 0.21, 0.20, 0.17; word cosine 0.05) and orders 64.5% of clicked-vs-skipped pairs correctly on held-out training-period impressions. It costs category diversity (−0.052 [−0.054, −0.051]) because it favours fresh and popular stories.

R@k is high even for random order because the median impression shows only 26 articles, so P@k and NDCG are the more informative metrics. MIND clicks are implicit feedback, and this split is not the official MIND test split, so these numbers are not comparable with the MIND leaderboard.

Supplementary checks on full-catalog recommendations (top 10 from all 51k articles for 200 held-out users, champion lists): multi-interest profiles cover 99% of a reader's interests with at least two read articles versus 74% for a single profile, the largest interest's share of the list falls from 80% to 61%, and near-duplicate pairs fall from 4.1 to 2.4. Champion lists reproduce 79% of exact full-catalog top-10 results in 31–42 ms per query versus 1.5–2 s for exact scoring; cluster pruning also reaches 79% but takes 290–380 ms (timings varied between runs on a shared laptop; the ratios held). Held-out clicks found in the top 10 are too few at this sample size (about five) to separate the methods.

## What works and what is planned

**Works now**

- TF-IDF indexing of 51,278 MIND articles (log TF, smoothed IDF, L2-normalized sparse rows) with a deterministic cache.
- User profiles as query vectors, single or multi-interest; cold-start and short-history modes.
- Three candidate generators: exact full-catalog scoring, champion lists, cluster pruning (√N farthest-first leaders, b1=2, b2=20).
- Net score from max-normalized cosine and Jaccard plus popularity g(d) and recency; heap-based top-K; MMR diversity reranking; "content seen?" filtering of re-published copies; term-level explanations.
- Centred Wikidata entity vectors for 44,262 articles (86%) from MIND's entity embeddings; interests labelled by shared entity names.
- A pairwise logistic learning-to-rank model over eight IR features, trained on training-period clicks.
- Browser workbench, terminal IR trace, chronological evaluation with eight models, validation tuning script, 28 unit tests.

**Planned / known limitations**

- No stemming or lemmatization ("recipe" does not match "recipes") and a short stopword list, so common words such as "new" can create weak matches.
- Interests are still grouped by words. Grouping with entity similarity was tested (model G) and did not change accuracy (NDCG@10 0.3750 vs 0.3749 on validation); on inspection it could merge unrelated stories, so entities are used for labels and as a ranking feature instead (where they add +0.004 NDCG@10). Articles without entities, and interests whose articles share no entity, fall back to top-word labels.
- MIND has no publication dates. The evaluation uses a first-exposure proxy; the workbench and trace have no recency signal at all.
- Popularity uses training-period counts, which say nothing about the 69% of test-period clicks that go to articles with no training clicks. Short-window (last-hours) popularity is a natural extension.
- The learning-to-rank model buys accuracy with less category variety (see Results); combining it with MMR or interest quotas is the obvious next step.
- No BM25 or dense text retrieval yet; these are the remaining planned extensions, along with evaluating on the official MIND dev split.

## Chronological evaluation protocol

The evaluation uses one global chronological timestamp cutoff. Training impressions are at or before the cutoff; held-out impressions are strictly later (timestamp ties stay on the training side). Interactions without parseable timestamps are omitted and counted in `run_summary.json`. Each held-out impression contributes only its recorded pre-impression history and its own displayed articles. Clicked labels are relevant items only; already consumed items, unknown article IDs, and unshown clicks cannot become recommendations or positives. Shown articles that are re-published copies of something the user already read are removed for every model, including the baseline. Popularity is calculated from training clicks only. MIND `news.tsv` has no publication timestamps, so the recency feature uses a proxy the system could know at the cutoff: the first training impression that showed the article, the start of the training log for articles only seen in reading histories, and the cutoff itself for articles never seen before it (new in the test period). It should not be described as true publication freshness.

The evaluation is an eight-model ablation. A is the training-click popularity baseline; B uses cosine alone; C uses cosine+Jaccard; D adds recency and popularity; E applies diversity reranking to D; F is D with multi-interest profiles; G is F with interests grouped by word and entity similarity; H ranks with weights learned by a pairwise logistic model. All variants run on the same held-out impression loop, candidate lists, and relevance labels; `evaluation_set_sha256` in the JSON fingerprints that shared set. Metrics are calculated per impression, averaged within each user, then macro-averaged across users so highly active users do not dominate. Precision divides by K even when fewer than K candidates are available. Coverage is distinct recommended articles over the valid indexed catalog; category diversity is the fraction of nonempty categories represented in the returned list. Mean pairwise cosine is also emitted (lower means less lexical redundancy). Binary NDCG is included as an additional metric.

The component weights are fixed before evaluation and are not selected by held-out results. The multi-interest settings (at most 3 interests, joining threshold 0.10) were chosen with `t3rec.tune_cli` on the latest 12.5% of the training period, before the test impressions were scored. Cosine and Jaccard are divided by their maximum over each impression's candidates, then fused as a weighted mean, dividing by the sum of active weights:

| Variant | Cosine | Jaccard | Recency | Popularity | Diversity | Profile |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A. Popularity baseline | — | — | — | train-click count sort | None | — |
| B. TF-IDF cosine | 1.00 | 0.00 | 0.00 | 0.00 | None | single |
| C. TF-IDF + Jaccard | 0.90 | 0.10 | 0.00 | 0.00 | None | single |
| D. TF-IDF + Jaccard + recency/popularity | 0.75 | 0.10 | 0.05 | 0.10 | None | single |
| E. Full model + diversity reranking | 0.75 | 0.10 | 0.05 | 0.10 | λ=0.75 | single |
| F. Multi-interest profiles | 0.75 | 0.10 | 0.05 | 0.10 | None | ≤3 interests |
| G. Entity-aware multi-interest | 0.75 | 0.10 | 0.05 | 0.10 | None | ≤3 interests, entity weight 0.3 when grouping |
| H. Learning-to-rank | learned | learned | learned | learned | None | eight features incl. entity similarity |

The command writes `aggregate_metrics.csv`, `per_user_metrics.csv`, `run_summary.json`, `ranking_metrics.svg`, `coverage_diversity.svg`, `ablation_results.csv`, `ablation_results.json`, and `ablation_plot.svg`. The aggregate table includes P/R/NDCG at every requested K, coverage, category diversity, and average pairwise cosine at the largest K. MIND's future clicks are implicit feedback rather than manually judged relevance, so no extra "human relevance" labels are claimed.

The pipeline parses article IDs, title, abstract, category, subcategory, URL, and title/abstract entity fields; parses impression timestamps, pre-impression histories, shown candidates, and click labels; counts missing fields and unknown article references; builds the sparse TF-IDF rows; and logs a validation summary. It collects no data beyond the anonymized identifiers and interactions already present in MIND. The timestamp on a MIND behavior row describes the impression, not each historical click, so it is not used as a history interaction time.

## IR choices

- Tokenization and stopword removal are explicit in `text.py`; stemming is optional via a caller-provided function and disabled by default.
- Article rows are sparse term-to-weight mappings. TF uses `1 + log(count)`, smoothed IDF is `log((1 + N)/(1 + df)) + 1`, and each row is L2 normalized.
- User query: `normalize(sum_i interaction_weight_i * exp(-ln(2)*age_i/half_life) * article_tfidf_i)`. Clicks default to weight 1; explicit per-interaction weights and optional time decay are supported.
- Multi-interest profiles (`profiles.build_interest_profiles`): the history is visited in reading order; each article joins the interest whose centroid it is most similar to if the cosine is at least 0.10, otherwise it leads a new interest, up to 3. Each interest's champion lists supply candidates; an article's relevance is its cosine to the best-matching interest; top-K slots are split across interests by the largest-remainder method in proportion to interest size, and when diversity is on, MMR runs within each interest so it cannot undo the split. Articles that share no term with any interest only fill slots no interest could fill. Validation on the training period showed more interests lowering NDCG (3 → 0.379, 5 → 0.374 against 0.382 for a single profile) while the joining threshold made no difference, because most news articles share almost no terms.
- Entities (`entities.py`): an article's entity vector is the confidence-weighted mean of the MIND embeddings of its title (weight 2) and abstract (weight 1) entities. Raw embeddings share a common direction (random article pairs score 0.29), so vectors are centred on the catalog mean (random pairs about 0.0, same-subcategory pairs about 0.2, same-subcategory pairs with no shared word 0.16) and L2-normalized. Interest labels are the entity name mentioned by the most member articles, plus a second only when at least two members mention it. Blending entity cosine into candidate matching lowered validation NDCG@10 at every weight (0.375 to 0.354–0.365), so candidates are matched by words; entity similarity to the reader's history enters ranking only through the learned model; removing it from that model lowers NDCG@10 by 0.004.
- Learning to rank (`ltr.py`): each candidate gets eight features (raw cosine, scaled cosine, scaled Jaccard, recency, popularity, best-interest cosine, entity similarity, category match). Within an impression every (clicked, skipped) pair is a training example for a linear model trained with the pairwise logistic loss log(1 + exp(-w · (z(x+) - z(x-)))), features standardized, plain SGD, L2 1e-4, four skipped articles per click. It trains on the latest 12.5% of the training period, with features computed from signals known before that slice, so no test impression is seen.
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
| Entity vectors and labels | `src/t3rec/entities.py` |
| Learning to rank (features, pairwise training) | `src/t3rec/ltr.py` |
| Champion lists, cluster pruning | `src/t3rec/candidates.py` |
| Jaccard, popularity g(d), recency, net score | `src/t3rec/scoring.py` |
| Heap top-K, copy filter, interest quotas, pipeline | `src/t3rec/pipeline.py` |
| MMR diversity | `src/t3rec/diversity.py` |
| Explanations | `src/t3rec/explain.py` |
| Evaluation and tuning | `src/t3rec/evaluate.py`, `src/t3rec/eval_cli.py`, `src/t3rec/tune_cli.py` |
| Interfaces | `src/t3rec/web.py` (+ `web/`), `src/t3rec/demo_ir.py`, `src/t3rec/cli.py` |

## Data credit

MIND: Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. "MIND: A Large-scale Dataset for News Recommendation." ACL 2020. Used under the Microsoft Research License Terms; not redistributed here. `data/synthetic.json` and `data/sample_mind/` are hand-written toy data.
