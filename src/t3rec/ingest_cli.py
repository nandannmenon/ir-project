"""Command-line entry point for deterministic MIND ingestion."""

from __future__ import annotations

import argparse
import logging

from .ingest import ingest_mind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/MINDsmall_train",
                        help="Folder containing MIND news.tsv and behaviors.tsv")
    parser.add_argument("--output-dir", default="data/processed/MINDsmall_train",
                        help="Cache directory for cleaned data and sparse TF-IDF rows")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the cache matches")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(name)s: %(message)s")
    dataset = ingest_mind(args.raw_dir, args.output_dir, force=args.force)
    print("MIND ingestion complete")
    for key, value in dataset.validation.items():
        print(f"{key}: {value}")
    print(f"cache: {args.output_dir}/dataset.json.gz")


if __name__ == "__main__":
    main()
