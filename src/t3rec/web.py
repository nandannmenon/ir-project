"""Serve a local browser workbench for the recommender."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .data import load_json_fixture
from .ingest import ingest_mind
from .models import User
from .pipeline import Recommender
from .scoring import ScoreWeights

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent / "web"


class Workbench:
    """Small JSON facade over the recommender, shared by the HTTP routes."""

    def __init__(self, recommender: Recommender, users: list[User]):
        self.recommender = recommender
        self.users = {user.user_id: user for user in users}

    def state(self) -> dict[str, Any]:
        return {
            "article_count": len(self.recommender.articles),
            "user_count": len(self.users),
            "candidate_modes": ["champions", "clusters", "all"],
            "users": [
                {"user_id": user.user_id, "history": list(user.history)}
                for user in sorted(self.users.values(), key=lambda user: user.user_id)[:300]
            ],
        }

    def catalog(self, query: str = "", limit: int = 24) -> list[dict[str, str]]:
        needle = query.strip().casefold()
        rows = (
            article for article in self.recommender.articles
            if not needle or needle in " ".join((article.article_id, article.title,
                                                  article.category, article.subcategory)).casefold()
        )
        return [
            {"article_id": article.article_id, "title": article.title,
             "category": article.category, "subcategory": article.subcategory}
            for article in sorted(rows, key=lambda article: (article.title.casefold(), article.article_id))
            if article.article_id
        ][:max(1, min(limit, 40))]

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id", "custom")).strip() or "custom"
        raw_history = payload.get("history", [])
        if not isinstance(raw_history, list) or len(raw_history) > 200:
            raise ValueError("history must be a list of at most 200 article IDs")
        history = tuple(dict.fromkeys(str(article_id).strip() for article_id in raw_history
                                      if str(article_id).strip()))
        limit = max(1, min(int(payload.get("limit", 5)), 20))
        pool_size = max(limit, min(int(payload.get("candidate_pool_size", 100)), 5000))
        candidate_mode = str(payload.get("candidate_mode", "champions"))
        if candidate_mode not in {"champions", "clusters", "all"}:
            raise ValueError("candidate_mode must be champions, clusters, or all")
        raw_weights = payload.get("weights", {})
        if not isinstance(raw_weights, dict):
            raise ValueError("weights must be an object")
        weights = ScoreWeights(
            alpha=float(raw_weights.get("relevance", 0.75)),
            beta=float(raw_weights.get("jaccard", 0.10)),
            gamma=float(raw_weights.get("recency", 0.05)),
            delta=float(raw_weights.get("popularity", 0.10)),
        )
        results = self.recommender.recommend(
            User(user_id, history), limit=limit, candidate_mode=candidate_mode,
            candidate_pool_size=pool_size, diverse=bool(payload.get("diverse", True)),
            diversity_relevance_weight=float(payload.get("diversity_relevance_weight", 0.75)),
            score_weights=weights, multi_interest=bool(payload.get("multi_interest", False)),
        )
        diagnostics = self.recommender.last_diagnostics
        profile = diagnostics.get("profile_vector", {})
        return {
            "user_id": user_id,
            "history_count": len(history),
            "mode": diagnostics.get("recommendation_mode"),
            "candidate_mode": candidate_mode,
            "candidate_count": diagnostics.get("candidate_count", 0),
            "history_copies_removed": len(diagnostics.get("history_copies_removed", [])),
            "recency_available": diagnostics.get("recency_available", False),
            "interests": [{key: interest[key] for key in ("label", "category", "size")}
                          for interest in diagnostics.get("interests", [])],
            "profile_terms": [
                {"term": term, "weight": score}
                for term, score in sorted(profile.items(), key=lambda row: (-row[1], row[0]))[:12]
            ],
            "recommendations": [
                {
                    "article_id": row.article_id,
                    "title": row.title,
                    "category": row.category,
                    "subcategory": row.article.subcategory,
                    "rank": row.rank,
                    "net_score": row.net_score,
                    "final_score": row.final_score,
                    "cosine": row.cosine_score,
                    "cosine_normalized": row.relevance_normalized,
                    "jaccard": row.jaccard_score,
                    "jaccard_normalized": row.jaccard_normalized,
                    "recency": row.recency_score,
                    "popularity": row.popularity_score,
                    "redundancy": row.redundancy_score,
                    "interest": row.interest,
                    "explanation": row.explanation,
                    "terms": [{"term": term, "contribution": score}
                              for term, score in row.term_contributions],
                }
                for row in results
            ],
        }


def build_handler(workbench: Workbench) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/api/state":
                self._json(workbench.state())
                return
            if request.path == "/api/catalog":
                query = parse_qs(request.query)
                self._json(workbench.catalog(query.get("q", [""])[0]))
                return
            asset = {"/": "index.html", "/styles.css": "styles.css",
                     "/app.js": "app.js"}.get(request.path)
            if asset:
                path = WEB_ROOT / asset
                content_type = "text/html; charset=utf-8" if asset.endswith(".html") else "text/css; charset=utf-8" if asset.endswith(".css") else "text/javascript; charset=utf-8"
                self._send(path.read_bytes(), content_type)
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/recommend":
                self._json({"error": "not found"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 1_000_000:
                    raise ValueError("request body must be between 1 byte and 1 MB")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                self._json(workbench.recommend(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(ROOT / "data/synthetic.json"))
    parser.add_argument("--raw-dir", help="Use a downloaded MIND directory instead of a JSON fixture")
    parser.add_argument("--cache-dir", default=str(ROOT / "data/processed/ui"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.raw_dir:
        dataset = ingest_mind(args.raw_dir, args.cache_dir)
        recommender = Recommender.from_processed(dataset)
        users = list(dataset.users.values())
    else:
        articles, users = load_json_fixture(args.fixture)
        clicks = Counter(article_id for user in users for article_id in user.history)
        recommender = Recommender(articles, click_counts=clicks)

    server = ThreadingHTTPServer((args.host, args.port), build_handler(Workbench(recommender, users)))
    print(f"T3 recommender workbench: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workbench")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()