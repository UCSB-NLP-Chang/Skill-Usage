#!/usr/bin/env python3
"""HTTP server exposing skill search endpoints.

Start:
    python http_server.py              # default port 8742
    python http_server.py --port 9000  # custom port

Endpoints:
    GET /keyword?q=react&top_k=10
    GET /semantic?q=build+CI+pipelines&top_k=10
    GET /hybrid?q=react+components&top_k=10&keyword_weight=0.4&semantic_weight=0.6
    GET /detail/{identifier}           (slug or UUID)
    GET /health
"""

import argparse
import json
import os
import sys

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(SCRIPT_DIR, "index")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "skills")

sys.path.insert(0, SCRIPT_DIR)

from semantic_search import DEFAULT_EMBEDDING_MODEL

app = FastAPI(title="Skill Search")

_bm25 = None
_semantic = None
_embedding_model = DEFAULT_EMBEDDING_MODEL
_index_suffix = ""
_db_suffix = ""
_include_content = False
_content_weight = 0.05


def get_bm25():
    global _bm25
    if _bm25 is None:
        from bm25_search import BM25Search
        _bm25 = BM25Search(os.path.join(INDEX_DIR, f"skills{_db_suffix}.db"))
    return _bm25


def get_semantic():
    global _semantic
    if _semantic is None:
        from semantic_search import SemanticSearch
        content_emb = os.path.join(INDEX_DIR, f"content_embeddings{_index_suffix}.npy")
        _semantic = SemanticSearch(
            os.path.join(INDEX_DIR, f"embeddings{_index_suffix}.npy"),
            os.path.join(INDEX_DIR, f"skill_ids{_index_suffix}.json"),
            os.path.join(INDEX_DIR, f"skills{_db_suffix}.db"),
            model_name=_embedding_model,
            content_embeddings_path=content_emb if _include_content else None,
            content_weight=_content_weight,
        )
    return _semantic


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/keyword")
def keyword_search(q: str, top_k: int = Query(default=10, le=50)):
    """BM25 keyword search. Supports FTS5 syntax: prefix (react*),
    phrases ("react hooks"), boolean (react OR vue)."""
    results = get_bm25().search(q, top_k)
    return results


@app.get("/semantic")
def semantic_search(q: str, top_k: int = Query(default=10, le=50)):
    """Semantic similarity search using dense embeddings."""
    results = get_semantic().search(q, top_k)
    return results


@app.get("/hybrid")
def hybrid_search(
    q: str,
    top_k: int = Query(default=10, le=50),
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
    bm25_content_weight: float | None = None,
    semantic_content_weight: float | None = None,
):
    """Hybrid search combining BM25 and semantic with reciprocal rank fusion.

    Optional per-query content weight overrides:
    - bm25_content_weight: BM25 weight for skill_md_content field (default: 5.0)
    - semantic_content_weight: blend weight for content vs summary embeddings (default: server setting)
    """
    kw_results = get_bm25().search(q, top_k * 2, content_weight=bm25_content_weight)
    sem_results = get_semantic().search(q, top_k * 2, content_weight=semantic_content_weight)

    rrf_scores: dict[str, float] = {}
    skill_data: dict[str, dict] = {}
    k = 60

    for rank, r in enumerate(kw_results):
        sid = r["skill_id"]
        rrf_scores[sid] = rrf_scores.get(sid, 0) + keyword_weight / (k + rank + 1)
        if sid not in skill_data:
            skill_data[sid] = r

    for rank, r in enumerate(sem_results):
        sid = r["skill_id"]
        rrf_scores[sid] = rrf_scores.get(sid, 0) + semantic_weight / (k + rank + 1)
        if sid not in skill_data:
            skill_data[sid] = r

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    results = []
    for sid in sorted_ids:
        r = skill_data[sid]
        r.pop("score", None)
        r["rrf_score"] = rrf_scores[sid]
        results.append(r)

    return results


@app.get("/detail/{identifier}")
def get_detail(identifier: str):
    """Get full skill details by skill_id or UUID."""
    import sqlite3

    db = sqlite3.connect(os.path.join(INDEX_DIR, f"skills{_db_suffix}.db"))
    db.row_factory = sqlite3.Row
    row = (
        db.execute("SELECT * FROM skills WHERE skill_id = ?", (identifier,)).fetchone()
        or db.execute("SELECT * FROM skills WHERE id = ?", (identifier,)).fetchone()
    )
    db.close()
    if not row:
        return JSONResponse({"error": "Skill not found"}, status_code=404)

    result = dict(row)
    for key in ("rowid", "skill_md_snippet"):
        result.pop(key, None)

    owner = result.get("owner", "")
    name = result.get("name", "")
    skill_md_path = os.path.join(DATA_DIR, f"{owner}--{name}", "SKILL.md")
    if os.path.exists(skill_md_path):
        with open(skill_md_path, errors="replace") as f:
            result["skill_md"] = f.read()

    return result


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8742)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--exclude-gt", action="store_true",
        help="Use indexes built with --exclude-gt (skills_no_gt.db, etc.)",
    )
    parser.add_argument(
        "--include-content", action="store_true",
        help="Enable SKILL.md content in semantic search (requires content embeddings)",
    )
    parser.add_argument(
        "--content-weight", type=float, default=0.05,
        help="Weight for content embeddings vs summary (default: 0.05)",
    )
    args = parser.parse_args()

    _embedding_model = args.model
    _include_content = args.include_content or os.environ.get("INCLUDE_CONTENT") == "1"
    _content_weight = float(os.environ.get("CONTENT_WEIGHT", str(args.content_weight)))
    _index_suffix = "_no_gt" if args.exclude_gt else ""
    _db_suffix = ("_full" if _include_content else "") + _index_suffix

    print(f"Starting skill search server on http://localhost:{args.port}")
    print(f"Embedding model: {_embedding_model}")
    print(f"Index suffix: '{_index_suffix}' (DB: '{_db_suffix}')")
    print(f"Include content: {_include_content}")
    if _include_content:
        print(f"Content weight: {_content_weight}")
    print(f"Docs: http://localhost:{args.port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
