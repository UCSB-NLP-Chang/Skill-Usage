#!/usr/bin/env python3
"""Sweep content weights for hybrid retrieval and evaluate.

Uses pre-generated queries (from generate_queries.py) to call the HTTP search
server's /hybrid endpoint with per-query weight overrides. Sweeps are
parallelized across weight combinations using a thread pool.

Requires the HTTP server running with --include-content:
    python search_server/http_server.py --include-content --exclude-gt

Usage:
    python scripts/sweep_retrieval.py --queries data/task_queries.json
    python scripts/sweep_retrieval.py --queries data/task_queries.json \
        --bm25-weights 0.5 1.0 2.0 --semantic-weights 0.0 0.3 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_PATH = REPO_ROOT / "data" / "task_skill_mapping.json"
GT_OWNER = "benchflow-ai"

DEFAULT_BASE_URL = "http://localhost:8742"


def load_gt() -> dict[str, set[str]]:
    raw = json.loads(GT_PATH.read_text())
    return {
        task: {f"{GT_OWNER}--{s}" for s in skills}
        for task, skills in raw.items()
    }


def recall_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(retrieved[:k]) & gt) / len(gt)


def mrr(retrieved: list[str], gt: set[str]) -> float:
    for i, s in enumerate(retrieved):
        if s in gt:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_results(results: dict[str, list[str]], gt: dict[str, set[str]],
                     ks: list[int] = [5, 10, 20]) -> dict[str, float]:
    tasks = sorted(set(results.keys()) & set(gt.keys()))
    if not tasks:
        return {}
    metrics = {}
    for k in ks:
        recalls = [recall_at_k(results[t], gt[t], k) for t in tasks]
        metrics[f"R@{k}"] = float(np.mean(recalls))
    mrrs = [mrr(results[t], gt[t]) for t in tasks]
    metrics["MRR"] = float(np.mean(mrrs))
    metrics["n_tasks"] = len(tasks)
    return metrics


# --------------- HTTP search helpers ---------------

def _http_hybrid(base_url: str, query: str, top_k: int,
                 keyword_weight: float, semantic_weight: float,
                 bm25_content_weight: float | None,
                 semantic_content_weight: float | None) -> list[dict]:
    params = {
        "q": query,
        "top_k": top_k * 2,
        "keyword_weight": keyword_weight,
        "semantic_weight": semantic_weight,
    }
    if bm25_content_weight is not None:
        params["bm25_content_weight"] = bm25_content_weight
    if semantic_content_weight is not None:
        params["semantic_content_weight"] = semantic_content_weight
    url = f"{base_url}/hybrid?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read())


def _interleave(ranked_lists: list[list[str]], top_k: int) -> list[str]:
    """Round-robin interleave across ranked lists, skipping duplicates."""
    result = []
    seen = set()
    max_len = max((len(r) for r in ranked_lists), default=0)
    for rank in range(max_len):
        for r_list in ranked_lists:
            if rank < len(r_list):
                sid = r_list[rank]
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
                    if len(result) >= top_k:
                        return result
    return result


def run_sweep_config(
    queries: dict[str, list[str]],
    base_url: str,
    top_k: int,
    bm25_cw: float,
    sem_cw: float,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> dict[str, list[str]]:
    """Run hybrid search for all tasks with a specific weight config."""
    results = {}
    for task_name, task_queries in queries.items():
        per_query_ranks = []
        for q in task_queries:
            hits = _http_hybrid(
                base_url, q, top_k,
                keyword_weight, semantic_weight,
                bm25_content_weight=bm25_cw,
                semantic_content_weight=sem_cw,
            )
            ranked = [r["skill_id"] for r in hits if "skill_id" in r]
            per_query_ranks.append(ranked)

        if len(per_query_ranks) == 1:
            results[task_name] = per_query_ranks[0][:top_k]
        else:
            results[task_name] = _interleave(per_query_ranks, top_k)
    return results


def fmt_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if k == "n_tasks":
            continue
        parts.append(f"{k}={v:.3f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Sweep content weights for hybrid retrieval")
    parser.add_argument(
        "--queries", required=True,
        help="Path to task_queries.json (from generate_queries.py)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results per search (default: 10)",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"HTTP server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--bm25-weights", nargs="+", type=float,
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="BM25 content weights to sweep (default: 0.0 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--semantic-weights", nargs="+", type=float,
        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        help="Semantic content weights to sweep (default: 0.0 0.2 0.4 0.5 0.6 0.8 1.0)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for sweep results (default: data/sweep/)",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[5, 10, 20],
        help="Values of k for evaluation (default: 5 10 20)",
    )
    parser.add_argument(
        "--keyword-weights", nargs="+", type=float,
        default=[0.5],
        help="Keyword RRF weights to sweep; semantic = 1 - keyword (default: 0.5)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for sweep (default: 4)",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        queries_path = REPO_ROOT / args.queries
    queries = json.loads(queries_path.read_text())
    print(f"Loaded queries for {len(queries)} tasks")

    gt = load_gt()
    print(f"Ground truth: {len(gt)} tasks")

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "data" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build grid of (bm25_cw, sem_cw, kw_w) combinations
    configs = [
        (b, s, kw)
        for b in args.bm25_weights
        for s in args.semantic_weights
        for kw in args.keyword_weights
    ]
    print(f"Sweeping {len(configs)} weight combinations with {args.workers} workers")

    all_results = []

    def _run_one(bm25_cw: float, sem_cw: float, kw_w: float):
        results = run_sweep_config(
            queries, args.base_url, args.top_k, bm25_cw, sem_cw,
            keyword_weight=kw_w,
            semantic_weight=1.0 - kw_w,
        )
        metrics = evaluate_results(results, gt, args.ks)
        out_file = output_dir / f"hybrid_bm25cw{bm25_cw}_scw{sem_cw}_kw{kw_w}.json"
        out_file.write_text(json.dumps(results, indent=2) + "\n")
        return bm25_cw, sem_cw, kw_w, metrics

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, *cfg): cfg
            for cfg in configs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweep"):
            bm25_cw, sem_cw, kw_w, metrics = future.result()
            label = f"bm25_cw={bm25_cw}, sem_cw={sem_cw}, kw={kw_w}"
            tqdm.write(f"  {label}  {fmt_metrics(metrics)}")
            all_results.append({
                "method": "hybrid",
                "bm25_content_weight": bm25_cw,
                "semantic_content_weight": sem_cw,
                "keyword_weight": kw_w,
                **metrics,
            })

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by R@10)")
    print("=" * 70)

    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")

    header = f"{'Config':<30}"
    for k in args.ks:
        header += f"{'R@'+str(k):>8}"
    header += f"{'MRR':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    sorted_results = sorted(all_results, key=lambda x: x.get("R@10", 0), reverse=True)
    for r in sorted_results:
        config = f"bm25={r['bm25_content_weight']}, sem={r['semantic_content_weight']}, kw={r['keyword_weight']}"
        line = f"{config:<30}"
        for k in args.ks:
            line += f"{r.get(f'R@{k}', 0):>8.3f}"
        line += f"{r.get('MRR', 0):>8.3f}"
        print(line)

    print(f"\nSaved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
