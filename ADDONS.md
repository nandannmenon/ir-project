# Add-ons to the news recommender

We added two things to the project:

1. **Entity-based interests:** the system now knows which people, places and organizations each article is about.
2. **Learning to rank:** the system now learns how to combine its signals from real clicks, instead of using numbers we chose by hand.

---

## 1. Entity-based interests

### The problem

Our recommender understands an article only through its words. Two articles about the same basketball team can use completely different words, for example "Celtics win again" and "Hayward breaks his hand". To the old system, those two articles looked unrelated.

Our interest labels also came from the most common words, so some were meaningless, like "before · older · hold".

### What we added

The MIND dataset tells us which **entities** each article mentions. An entity is a named person, place, team or organization, such as "Gordon Hayward", "Boston Celtics" or "Ukraine". MIND also gives each entity a list of numbers that describes it. Related entities, like a player and his team, get similar numbers.

We use this in three ways:

1. **Each article gets an entity summary:** the average of the numbers for the entities it mentions. Entities in the title count twice as much as those in the summary. 86% of articles have at least one entity.
2. **Interests get readable names.** Each interest is now labeled with the entity its articles mention most, for example "Gordon Hayward" or "Ukraine · Donald Trump", instead of random top words.
3. **Entity similarity becomes a signal** in the new ranking model (add-on 2): how close an article's entities are to the entities the reader usually reads about.

### One fix we had to make

At first, all entity summaries looked a bit alike, because they shared a common "average" part. Even two random articles scored 0.29 out of 1. We subtracted that shared average. After that, random articles score about 0, and articles on the same topic score clearly higher. This made the entity signal useful.

### What we tried that did not work

We also tried using entities to **decide which interest an article belongs to**. We tested this on older data (never on the final test data):

- It **did not improve accuracy** (0.3750 vs 0.3749).
- In one example it **made things worse**: Gordon Hayward stories were placed into a "Donald Trump · Ukraine" group.

So we kept grouping by words, and used entities only for **labels** and as a **ranking signal**. This is reported as model G in our evaluation.

### Result

- Interest labels are now real names a person can understand.
- Accuracy stayed the same (model G vs model F: no real difference).

---

## 2. Learning to rank

### The problem

Our system scores each article by mixing four signals with weights we picked ourselves:

| Signal | What it means | Our hand-picked weight |
|---|---|---|
| Word match | How many important words the article shares with what you read | 75% |
| Topic overlap | Shared topics and categories | 10% |
| Freshness | How new the article is | 5% |
| Popularity | How many people clicked it | 10% |

We guessed these numbers. We had no proof they were right.

### What we added

We let the system **learn the weights from real clicks**.

1. **Training examples.** For each past session, we take an article the user clicked and an article they saw but skipped.
2. **The learning goal.** The system adjusts its weights so the clicked article scores higher than the skipped one, across thousands of such pairs.
3. **Eight signals instead of four.** The model uses the four above plus four more: entity similarity, how close the article is to the reader's best-matching interest, how much the reader reads that category, and a scaled version of word match.

We trained it only on **older sessions**, the last part of the training period. The final test sessions were never used for training, so the test is fair.

### What the model learned

The signals it relies on most:

1. **Entity similarity**, our new signal from add-on 1
2. **Freshness**
3. **Popularity**

Plain word match ended up less important than we had assumed. This fits what we saw in the data: people mostly click new and trending stories.

### Result

We tested on 31,290 real sessions from 20,654 users:

| Model | Correct clicks in the top 5 (P@5) | Ranking quality (NDCG@10) |
|---|---|---|
| Random order | 0.075 | 0.265 |
| Most popular articles | 0.079 | 0.286 |
| Our hand-weighted system | 0.101 | 0.348 |
| **Learning to rank** | **0.109** | **0.373** |

- Learning to rank is **clearly better** than our hand-picked weights: ranking quality went up by +0.025, and the improvement is statistically significant.
- It is the biggest improvement since we first beat the "most popular" baseline.

### The trade-off

The learned model recommends **less varied** lists (category variety dropped from 0.62 to 0.56), because it focuses on fresh and popular stories. A good next step is to combine it with our diversity and multi-interest features to get both accuracy and variety.

---

## Summary

| Add-on | Did it improve accuracy? | What it adds |
|---|---|---|
| Entity-based interests | No (same accuracy) | Readable interest names, and a strong new signal for ranking |
| Learning to rank | **Yes** (0.348 → 0.373) | Weights learned from real clicks instead of guessed |

## Where it is in the code

| What | File |
|---|---|
| Entity summaries and labels | `src/t3rec/entities.py` |
| Grouping and labeling interests | `src/t3rec/profiles.py` |
| Learning to rank | `src/t3rec/ltr.py` |
| Testing both add-ons (models G and H) | `src/t3rec/evaluate.py` |
