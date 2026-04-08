#!/usr/bin/env python3
"""Evaluate skill retrieval performance (recall@k, precision@k, MRR).

Usage:
    # From a folder of tasks with found_skills.json in each:
    python scripts/eval_retrieval.py tasks-local-found-skills

    # From a JSON file mapping task names to retrieved skill lists:
    python scripts/eval_retrieval.py results/keyword_results.json

    # Custom k values:
    python scripts/eval_retrieval.py tasks-local-found-skills --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_PATH = REPO_ROOT / "data" / "task_skill_mapping.json"
GT_OWNER = "benchflow-ai"


def load_gt() -> dict[str, set[str]]:
    """Load ground truth mapping. Values become skill_ids with owner prefix."""
    raw = json.loads(GT_PATH.read_text())
    return {
        task: {f"{GT_OWNER}--{s}" for s in skills}
        for task, skills in raw.items()
    }


def load_retrieved_from_folder(folder: Path) -> dict[str, list[str]]:
    """Load retrieved skills from found_skills.json in each task subfolder."""
    retrieved = {}
    for task_dir in sorted(folder.iterdir()):
        if not task_dir.is_dir():
            continue
        found = task_dir / "found_skills.json"
        if found.exists():
            try:
                skills = json.loads(found.read_text())
                if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
                    retrieved[task_dir.name] = skills
            except json.JSONDecodeError:
                pass
    return retrieved


def load_retrieved(path: Path) -> dict[str, list[str]]:
    """Load retrieved skills from a folder or JSON file."""
    if path.is_dir():
        return load_retrieved_from_folder(path)
    # JSON file: dict mapping task name -> list of skill_ids
    raw = json.loads(path.read_text())
    return {task: skills for task, skills in raw.items() if isinstance(skills, list)}


def recall_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & gt) / len(gt)


def precision_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & gt) / len(top_k)


def reciprocal_rank(retrieved: list[str], gt: set[str]) -> float:
    for i, skill in enumerate(retrieved):
        if skill in gt:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(retrieved: dict[str, list[str]], gt: dict[str, set[str]], ks: list[int]):
    # Only evaluate tasks present in both GT and retrieved
    tasks = sorted(set(retrieved.keys()) & set(gt.keys()))
    gt_only = sorted(set(gt.keys()) - set(retrieved.keys()))
    ret_only = sorted(set(retrieved.keys()) - set(gt.keys()))

    if gt_only:
        print(f"Tasks in GT but not retrieved ({len(gt_only)}): {', '.join(gt_only[:5])}{'...' if len(gt_only) > 5 else ''}")
    if ret_only:
        print(f"Tasks retrieved but not in GT ({len(ret_only)}): {', '.join(ret_only[:5])}{'...' if len(ret_only) > 5 else ''}")
    print(f"Evaluating on {len(tasks)} tasks\n")

    if not tasks:
        print("No overlapping tasks to evaluate.")
        return

    # Per-task metrics
    per_task = []
    for task in tasks:
        r = retrieved[task]
        g = gt[task]
        row = {"task": task, "gt_count": len(g)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(r, g, k)
            row[f"precision@{k}"] = precision_at_k(r, g, k)
        row["mrr"] = reciprocal_rank(r, g)
        per_task.append(row)

    # Aggregate
    max_k = max(ks)
    header_parts = ["Task"]
    for k in ks:
        header_parts.extend([f"R@{k}", f"P@{k}"])
    header_parts.append("MRR")

    col_w = 8
    task_w = 45
    header = f"{'Task':<{task_w}}" + "".join(f"{h:>{col_w}}" for h in header_parts[1:])
    print(header)
    print("-" * len(header))

    for row in per_task:
        line = f"{row['task']:<{task_w}}"
        for k in ks:
            line += f"{row[f'recall@{k}']:>{col_w}.3f}"
            line += f"{row[f'precision@{k}']:>{col_w}.3f}"
        line += f"{row['mrr']:>{col_w}.3f}"
        print(line)

    print("-" * len(header))

    # Averages
    avg_line = f"{'AVERAGE':<{task_w}}"
    for k in ks:
        avg_r = sum(r[f"recall@{k}"] for r in per_task) / len(per_task)
        avg_p = sum(r[f"precision@{k}"] for r in per_task) / len(per_task)
        avg_line += f"{avg_r:>{col_w}.3f}{avg_p:>{col_w}.3f}"
    avg_mrr = sum(r["mrr"] for r in per_task) / len(per_task)
    avg_line += f"{avg_mrr:>{col_w}.3f}"
    print(avg_line)


def main():
    parser = argparse.ArgumentParser(description="Evaluate skill retrieval performance")
    parser.add_argument(
        "input",
        help="Folder with task subdirs containing found_skills.json, or a JSON file",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="Values of k for recall@k and precision@k (default: 5 10 20)",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help=f"Path to ground truth JSON (default: {GT_PATH})",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt) if args.gt else GT_PATH
    if not gt_path.exists():
        print(f"Error: GT file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_dir():
        input_path = REPO_ROOT / args.input
    if not input_path.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    gt = load_gt() if not args.gt else {
        task: {f"{GT_OWNER}--{s}" for s in skills}
        for task, skills in json.loads(gt_path.read_text()).items()
    }
    retrieved = load_retrieved(input_path)

    print(f"GT: {len(gt)} tasks, Retrieved: {len(retrieved)} tasks")
    evaluate(retrieved, gt, sorted(args.k))


if __name__ == "__main__":
    main()
