#!/usr/bin/env python3
"""Calculate aggregated results from harbor run(s).

Usage:
    # Single job:
    python scripts/calculate_results.py <jobs_dir>/<job_name>

    # All jobs in a folder (saves summary JSON):
    python scripts/calculate_results.py <jobs_dir> --batch

Groups trials by task, randomly splits into runs, computes per-run averages,
then reports mean and std across runs.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_trial_id(trial_id: str) -> tuple[str, str]:
    """Parse 'taskname__trialid' into (task_name, trial_suffix)."""
    parts = trial_id.rsplit("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected trial ID format: {trial_id}")
    return parts[0], parts[1]


def calculate_job_results(job_dir: Path, seed: int = 42, verbose: bool = True) -> dict:
    """Calculate results for a single job directory.

    Returns dict with keys: overall_mean, overall_std, n_runs, n_tasks, n_trials, run_avgs
    """
    result_path = job_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"{result_path} not found")

    with open(result_path) as f:
        data = json.load(f)

    n_total_trials = data["n_total_trials"]

    # Collect all trials with their rewards
    task_trials = defaultdict(list)  # task_name -> list of (trial_id, reward)
    total_counted = 0

    for eval_name, eval_data in data["stats"]["evals"].items():
        for reward_str, trial_ids in eval_data["reward_stats"]["reward"].items():
            reward = float(reward_str)
            for trial_id in trial_ids:
                task_name, suffix = parse_trial_id(trial_id)
                task_trials[task_name].append((trial_id, reward))
                total_counted += 1

    n_errors_total = sum(e["n_errors"] for e in data["stats"]["evals"].values())

    if verbose:
        print(f"Total trials in result: {n_total_trials}")
        print(f"Trials with rewards: {total_counted}")
        print(f"Errors (from stats): {n_errors_total}")
        print()

    # Check trial counts per task
    trial_counts = {task: len(trials) for task, trials in task_trials.items()}
    most_common_count = max(set(trial_counts.values()), key=list(trial_counts.values()).count)

    if verbose:
        print(f"Number of tasks: {len(task_trials)}")
        print(f"Most common trial count per task: {most_common_count}")

    # Print tasks with different trial counts
    irregular_tasks = {t: c for t, c in trial_counts.items() if c != most_common_count}
    if irregular_tasks and verbose:
        print(f"\nTasks with different trial counts ({len(irregular_tasks)}):")
        for task, count in sorted(irregular_tasks.items()):
            print(f"  {task}: {count} trials (expected {most_common_count})")
    elif verbose:
        print("All tasks have the same number of trials.")

    if verbose:
        print()

    # Assert total trials match
    n_trials_in_evals = sum(e["n_trials"] for e in data["stats"]["evals"].values())
    assert total_counted == n_trials_in_evals, (
        f"Trial count mismatch: {total_counted} in reward_stats != {n_trials_in_evals} in evals"
    )
    n_missing = n_total_trials - n_trials_in_evals

    if verbose:
        print(f"✓ Trial count verified: {total_counted} scored trials match eval n_trials")
        if n_missing > 0:
            print(f"  ({n_missing} trials failed before evaluation, out of {n_total_trials} total)")
        print()

    # Randomly group trials into runs
    n_runs = most_common_count
    if verbose:
        print(f"Grouping into {n_runs} runs (based on most common trial count)")
        print()

    rng = random.Random(seed)
    run_scores = defaultdict(list)

    for task_name, trials in sorted(task_trials.items()):
        shuffled = trials.copy()
        rng.shuffle(shuffled)

        if len(shuffled) < n_runs:
            for i, (tid, reward) in enumerate(shuffled):
                run_scores[i].append(reward)
        else:
            for i, (tid, reward) in enumerate(shuffled):
                run_idx = i % n_runs
                run_scores[run_idx].append(reward)

    # Calculate per-run averages
    run_avgs = []
    for run_idx in range(n_runs):
        scores = run_scores[run_idx]
        avg = float(np.mean(scores))
        run_avgs.append(avg)
        if verbose:
            print(f"  Run {run_idx}: avg={avg:.4f} (n_tasks={len(scores)})")

    overall_mean = float(np.mean(run_avgs))
    overall_std = float(np.std(run_avgs, ddof=1)) if len(run_avgs) > 1 else 0.0

    if verbose:
        print()
        print(f"=== Final Results ===")
        print(f"Mean across {n_runs} runs: {overall_mean:.4f}")
        print(f"Std across {n_runs} runs:  {overall_std:.4f}")
        print(f"({overall_mean*100:.2f} ± {overall_std*100:.2f}%)")

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "n_runs": n_runs,
        "n_tasks": len(task_trials),
        "n_trials": total_counted,
        "run_avgs": run_avgs,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate results from SkillsBench run(s)")
    parser.add_argument("path", help="Path to a job directory or parent folder (with --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all subdirs with result.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouping trials into runs")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path (default: <path>/summary.json in batch mode)")
    args = parser.parse_args()

    path = Path(args.path)

    if not args.batch:
        calculate_job_results(path, seed=args.seed, verbose=True)
        return

    # Batch mode: iterate through subdirectories
    job_dirs = sorted([
        d for d in path.iterdir()
        if d.is_dir() and (d / "result.json").exists()
    ])

    if not job_dirs:
        print(f"No subdirectories with result.json found in {path}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    for job_dir in job_dirs:
        job_name = job_dir.name
        print(f"{'='*60}")
        print(f"Job: {job_name}")
        print(f"{'='*60}")
        try:
            result = calculate_job_results(job_dir, seed=args.seed, verbose=True)
            summary[job_name] = {
                "overall_mean": result["overall_mean"],
                "overall_std": result["overall_std"],
                "n_runs": result["n_runs"],
                "n_tasks": result["n_tasks"],
                "n_trials": result["n_trials"],
            }
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            summary[job_name] = {"error": str(e)}
        print()

    # Save summary
    output_path = Path(args.output) if args.output else path / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    # Print compact table
    print(f"\n{'Job':<60} {'Mean':>8} {'Std':>8} {'Runs':>5} {'Tasks':>6}")
    print("-" * 90)
    for job_name, result in summary.items():
        if "error" in result:
            print(f"{job_name:<60} ERROR: {result['error']}")
        else:
            print(f"{job_name:<60} {result['overall_mean']*100:>7.2f}% {result['overall_std']*100:>7.2f}% {result['n_runs']:>5} {result['n_tasks']:>6}")


if __name__ == "__main__":
    main()
