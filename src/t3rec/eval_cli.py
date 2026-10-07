"""Run the complete chronological evaluation from raw or sample MIND TSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .evaluate import run_evaluation, write_evaluation
from .ingest import ingest_mind


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronologically evaluate T3 recommender models")
    parser.add_argument("--raw-dir", default="data/sample_mind",
                        help="Folder containing MIND news.tsv and behaviors.tsv (default: included sample)")
    parser.add_argument("--cache-dir", help="Processed cache folder (default: <output-dir>/cache)")
    parser.add_argument("--output-dir", default="outputs/evaluation",
                        help="Folder for CSV metrics, run summary, and SVG plots")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Chronological tail fraction; timestamp ties stay on train side")
    parser.add_argument("--ks", default="5,10", help="Comma-separated positive cutoffs, e.g. 5,10")
    parser.add_argument("--diversity-lambda", type=float, default=0.75,
                        help="Relevance weight λ for greedy diversity reranking")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ks = tuple(int(part.strip()) for part in args.ks.split(",") if part.strip())
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    dataset = ingest_mind(raw_dir, cache_dir)
    result = run_evaluation(dataset, test_fraction=args.test_fraction, ks=ks,
                            diversity_lambda=args.diversity_lambda)
    write_evaluation(result, output_dir, ks)
    logging.getLogger("t3rec.evaluate").info(
        "Chronological cutoff %s: %d train, %d test, %d evaluable impressions across %d users",
        result.split.cutoff.isoformat(), len(result.split.train), len(result.split.test),
        result.summary["evaluated_impressions"], result.summary["evaluated_users"])
    display_metrics = [metric for k in ks for metric in (f"P@{k}", f"R@{k}")]
    display_metrics += ["Coverage", f"CategoryDiversity@{max(ks)}",
                        f"AvgPairwiseSimilarity@{max(ks)}"]
    print("Model | " + " | ".join(display_metrics))
    for row in result.aggregate:
        values = [f"{float(row.get(name, 0.0)):.4f}" for name in display_metrics]
        print(f"{row['Model']} | " + " | ".join(values))
    print(f"Wrote evaluation artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
