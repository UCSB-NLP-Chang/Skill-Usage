#!/usr/bin/env python3
"""Analyze skill usage in agent trajectories from harbor runs.

Usage:
    # Single job:
    python scripts/check_skill_usage.py <jobs_dir>/<job_name>

    # All jobs in a folder:
    python scripts/check_skill_usage.py <jobs_dir> --batch

Detects skill usage via:
1. Skill tool calls (function_name == "Skill")
2. Read tool calls on skill-related file paths
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


SKILL_PATH_PATTERNS = [
    re.compile(r"/skills/", re.IGNORECASE),
]


def resolve_skill_name(skill_name: str, task_name: str, tasks_dir: Path) -> str:
    """Resolve skill_name to the actual name from SKILL.md YAML frontmatter."""
    skill_md = tasks_dir / task_name / "environment" / "skills" / skill_name / "SKILL.md"
    if not skill_md.is_file():
        return skill_name
    try:
        content = skill_md.read_text()
    except OSError:
        return skill_name
    if not content.startswith("---"):
        return skill_name
    parts = content.split("---", 2)
    if len(parts) < 3:
        return skill_name
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return skill_name
    if isinstance(meta, dict) and "name" in meta:
        return meta["name"]
    return skill_name


def is_skill_read(file_path: str) -> bool:
    return any(p.search(file_path) for p in SKILL_PATH_PATTERNS)


def _is_step_error(step: dict) -> bool:
    """Check if a trajectory.json step's tool result was an error."""
    extra = step.get("extra", {})
    if extra.get("tool_result_is_error"):
        return True
    metadata = extra.get("tool_result_metadata", {})
    raw = metadata.get("raw_tool_result", {})
    if raw.get("is_error"):
        return True
    return False


def _extract_skill_name_from_path(file_path: str) -> str | None:
    """Extract skill name from a skill file path like /skills/<name>/SKILL.md."""
    m = re.search(r"/skills/([^/]+)/", file_path)
    return m.group(1) if m else None


def analyze_trajectory_json(traj_path: str) -> dict:
    """Analyze a Claude Code or kimi-cli trajectory.json file."""
    with open(traj_path) as f:
        data = json.load(f)

    skill_tool_calls = []
    skill_reads = []
    total_steps = len(data.get("steps", []))

    for step in data.get("steps", []):
        step_id = step.get("step_id", "?")
        if "tool_calls" not in step:
            continue
        is_error = _is_step_error(step)
        for tc in step["tool_calls"]:
            fn = tc.get("function_name", "")
            args = tc.get("arguments", {})

            if fn == "Skill":
                if is_error:
                    continue
                skill_name = args.get("skill", "unknown")
                skill_tool_calls.append({
                    "step_id": step_id,
                    "skill": skill_name,
                    "args": args.get("args", ""),
                })
            elif fn in ("Read", "ReadFile"):
                fp = args.get("file_path", "") or args.get("path", "")
                if is_skill_read(fp):
                    skill_reads.append({
                        "step_id": step_id,
                        "file_path": fp,
                    })

    return _build_result(total_steps, skill_tool_calls, skill_reads)


def analyze_trajectory_qwen(traj_path: str) -> dict:
    """Analyze a qwen-code JSONL trajectory (.txt) file."""
    skill_tool_calls = []
    skill_reads = []
    total_steps = 0

    # First pass: collect tool_result error status by tool_use_id
    tool_result_errors: dict[str, bool] = {}
    all_lines = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_lines.append(obj)
            if obj.get("type") == "user":
                for item in obj.get("message", {}).get("content", []):
                    if item.get("type") == "tool_result" and "tool_use_id" in item:
                        tool_result_errors[item["tool_use_id"]] = bool(item.get("is_error"))

    # Second pass: process assistant tool_use messages
    for obj in all_lines:
        if obj.get("type") != "assistant":
            continue
        content = obj.get("message", {}).get("content", [])
        has_tool_use = False
        for item in content:
            if item.get("type") != "tool_use":
                continue
            has_tool_use = True
            name = item.get("name", "")
            inp = item.get("input", {})
            tool_id = item.get("id", "")

            if name == "skill":
                # Skip if the tool result was an error (skill not found / not loaded)
                if tool_result_errors.get(tool_id, False):
                    continue
                skill_name = inp.get("skill", "unknown")
                skill_tool_calls.append({
                    "step_id": obj.get("uuid", "?"),
                    "skill": skill_name,
                    "args": inp.get("args", ""),
                })
            elif name == "read_file":
                fp = inp.get("absolute_path", "") or inp.get("file_path", "")
                if is_skill_read(fp):
                    skill_reads.append({
                        "step_id": obj.get("uuid", "?"),
                        "file_path": fp,
                    })
        if has_tool_use:
            total_steps += 1

    return _build_result(total_steps, skill_tool_calls, skill_reads)


def analyze_trajectory_terminus(traj_path: str) -> dict:
    """Analyze a terminus-2 trajectory.json file.

    Terminus-2 uses bash_command tool calls with keystrokes to interact.
    Skills are offered via <location> tags in the initial prompt, and the agent
    reads them via `cat /path/to/SKILL.md` keystrokes.
    """
    with open(traj_path) as f:
        data = json.load(f)

    skill_tool_calls = []
    skill_reads = []
    total_steps = 0

    for step in data.get("steps", []):
        if step.get("source") != "agent":
            continue
        total_steps += 1
        step_id = step.get("step_id", "?")
        for tc in step.get("tool_calls", []):
            keystrokes = tc.get("arguments", {}).get("keystrokes", "")
            # Detect cat commands on SKILL.md files
            if re.search(r"cat\s+\S*/skills/\S*SKILL\.md", keystrokes):
                # Extract the file path
                m = re.search(r"cat\s+(\S*/skills/\S*SKILL\.md)", keystrokes)
                if m:
                    skill_reads.append({
                        "step_id": step_id,
                        "file_path": m.group(1),
                    })

    return _build_result(total_steps, skill_tool_calls, skill_reads)


def _build_result(total_steps, skill_tool_calls, skill_reads):
    used_skill = len(skill_tool_calls) > 0 or len(skill_reads) > 0
    # Collect skill names from both Skill tool calls and file-path reads
    skills_from_calls = {s["skill"] for s in skill_tool_calls}
    skills_from_reads = {
        name for sr in skill_reads
        if (name := _extract_skill_name_from_path(sr["file_path"])) is not None
    }
    return {
        "total_steps": total_steps,
        "used_skill": used_skill,
        "skill_tool_calls": skill_tool_calls,
        "skill_reads": skill_reads,
        "num_skill_calls": len(skill_tool_calls),
        "num_skill_reads": len(skill_reads),
        "skills_invoked": sorted(skills_from_calls | skills_from_reads),
    }


def find_trajectory(entry_path: Path):
    """Find trajectory file in an entry directory.

    Returns (path, format) where format is 'json', 'terminus', or 'qwen',
    or (None, None).
    """
    agent_dir = entry_path / "agent"
    traj_json = agent_dir / "trajectory.json"
    if traj_json.is_file():
        # Terminus-2 leaves a terminus_2.pane file in the agent directory
        if (agent_dir / "terminus_2.pane").exists():
            return traj_json, "terminus"
        return traj_json, "json"
    # Qwen-code format: agent/qwen-code.txt (JSONL)
    for candidate in sorted(agent_dir.glob("*.txt")) if agent_dir.is_dir() else []:
        return candidate, "qwen"
    return None, None


def analyze_trajectory(traj_path: str, fmt: str = "json") -> dict:
    if fmt == "qwen":
        return analyze_trajectory_qwen(traj_path)
    if fmt == "terminus":
        return analyze_trajectory_terminus(traj_path)
    return analyze_trajectory_json(traj_path)


def analyze_job(job_dir: Path, verbose: bool = False) -> dict:
    """Analyze skill usage for a single job directory.

    Returns dict with keys: used_count, total_count, usage_pct, avg_gt_pct, results
    """
    is_skillsbench = "skillsbench-trajectories" in str(job_dir)
    repo_root = Path(__file__).resolve().parent.parent
    # GT tasks dir for GT skill coverage comparison
    gt_tasks_dir = repo_root / "tasks" if is_skillsbench else None
    # Actual tasks dir used by the job (for resolving skill folder names)
    actual_tasks_dir = None
    if is_skillsbench:
        config_path = job_dir / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text())
                for ds in config.get("datasets", []):
                    ds_path = ds.get("path", "")
                    if ds_path:
                        candidate = Path(ds_path)
                        if not candidate.is_absolute():
                            candidate = repo_root / candidate
                        if candidate.is_dir():
                            actual_tasks_dir = candidate
                        break
            except (json.JSONDecodeError, OSError):
                pass
        if actual_tasks_dir is None:
            actual_tasks_dir = gt_tasks_dir

    results = {}
    task_names = set()

    for entry in sorted(os.listdir(job_dir)):
        entry_path = job_dir / entry
        if not entry_path.is_dir():
            continue
        traj_path, traj_fmt = find_trajectory(entry_path)
        if traj_path is None:
            continue

        # Parse task name (everything before the last __hash)
        parts = entry.rsplit("__", 1)
        task_name = parts[0] if len(parts) == 2 else entry
        run_id = parts[1] if len(parts) == 2 else ""

        try:
            info = analyze_trajectory(str(traj_path), fmt=traj_fmt)
        except Exception as e:
            print(f"  Error reading {traj_path}: {e}", file=sys.stderr)
            continue

        # Resolve skill names from SKILL.md for skillsbench-trajectories
        if actual_tasks_dir:
            is_found_job = "found" in str(job_dir).lower()

            # Build map: resolved_name -> folder_name using the actual tasks dir
            resolved_to_folder: dict[str, str] = {}
            task_skills_dir = actual_tasks_dir / task_name / "environment" / "skills"
            if task_skills_dir.is_dir():
                for skill_folder in task_skills_dir.iterdir():
                    if skill_folder.is_dir():
                        resolved = resolve_skill_name(skill_folder.name, task_name, actual_tasks_dir)
                        resolved_to_folder[resolved] = skill_folder.name
                        resolved_to_folder[skill_folder.name] = skill_folder.name

            def _maybe_non_gt(name: str) -> str:
                """For found jobs, replace non-benchflow-ai skills with 'non-gt'."""
                if not is_found_job:
                    return name
                folder = resolved_to_folder.get(name, name)
                if folder.startswith("benchflow-ai"):
                    return name
                return "non-gt"

            for sc in info["skill_tool_calls"]:
                sc["skill"] = _maybe_non_gt(resolve_skill_name(sc["skill"], task_name, actual_tasks_dir))
            skills_from_calls = {s["skill"] for s in info["skill_tool_calls"]}
            skills_from_reads = set()
            for sr in info["skill_reads"]:
                folder = _extract_skill_name_from_path(sr["file_path"])
                if folder is None:
                    continue
                if is_found_job and not folder.startswith("benchflow-ai"):
                    skills_from_reads.add("non-gt")
                else:
                    skills_from_reads.add(resolve_skill_name(folder, task_name, actual_tasks_dir))
            info["skills_invoked"] = sorted(skills_from_calls | skills_from_reads)

        info["task_name"] = task_name
        info["run_id"] = run_id
        info["dir_name"] = entry
        results[entry] = info
        task_names.add(task_name)

    # Print summary table
    print(f"\n{'Task Directory':<55} {'Used Skill':<12} {'#Skill Calls':<14} {'#Skill Reads':<14} {'Skills Invoked'}")
    print("-" * 140)

    used_count = 0
    total_count = 0
    for entry, info in sorted(results.items()):
        total_count += 1
        if info["used_skill"]:
            used_count += 1
        skills_str = ", ".join(info["skills_invoked"]) if info["skills_invoked"] else "-"
        print(
            f"{entry:<55} {'YES' if info['used_skill'] else 'NO':<12} "
            f"{info['num_skill_calls']:<14} {info['num_skill_reads']:<14} {skills_str}"
        )

    usage_pct = used_count / total_count * 100 if total_count > 0 else 0.0
    print("-" * 140)
    print(f"Total: {total_count} runs, {used_count} used skills ({usage_pct:.1f}%), {total_count - used_count} did not")

    avg_gt_pct = None
    if is_skillsbench:
        # Load GT skill mapping
        mapping_path = Path(__file__).resolve().parent.parent / "data" / "task_skill_mapping.json"
        gt_skills = {}
        if mapping_path.is_file():
            with open(mapping_path) as f:
                gt_skills = json.load(f)
        else:
            print(f"\nWarning: GT skill mapping not found at {mapping_path}", file=sys.stderr)

        # Per-task summary (3 runs) with GT skill coverage
        header = f"{'Task':<45} {'Run 1':<12} {'Run 2':<12} {'Run 3':<12}"
        if gt_skills:
            header += f" {'GT Skills':<10} {'Run1 %':<10} {'Run2 %':<10} {'Run3 %':<10}"
        print(f"\n{header}")
        print("-" * len(header))

        all_gt_pcts = []

        for task_name in sorted(task_names):
            runs = sorted(
                [info for info in results.values() if info["task_name"] == task_name],
                key=lambda r: r["run_id"],
            )
            cols = []
            for r in runs:
                cols.append("YES" if r["used_skill"] else "NO")
            while len(cols) < 3:
                cols.append("-")
            line = f"{task_name:<45} {cols[0]:<12} {cols[1]:<12} {cols[2]:<12}"

            if gt_skills and task_name in gt_skills:
                gt_set = set(gt_skills[task_name])
                line += f" {len(gt_set):<10}"
                for r in runs:
                    invoked = set(r["skills_invoked"])
                    covered = sum(1 for g in gt_set if g in invoked or re.sub(r"-\d+$", "", g) in invoked)
                    pct = covered / len(gt_set) * 100 if gt_set else 0.0
                    all_gt_pcts.append(pct)
                    line += f" {pct:>5.1f}%    "
                for _ in range(3 - len(runs)):
                    line += f" {'-':<10}"
            elif gt_skills:
                line += f" {'?':<10} {'?':<10} {'?':<10} {'?':<10}"

            print(line)

        if all_gt_pcts:
            avg_gt_pct = sum(all_gt_pcts) / len(all_gt_pcts)
            print(f"\nAverage GT skill coverage: {avg_gt_pct:.1f}% across {len(all_gt_pcts)} trials")

    if verbose:
        print("\n=== Detailed skill usage ===\n")
        for entry, info in sorted(results.items()):
            if not info["used_skill"]:
                continue
            print(f"\n{entry}:")
            for sc in info["skill_tool_calls"]:
                print(f"  Step {sc['step_id']}: Skill('{sc['skill']}'{', args=' + repr(sc['args']) if sc['args'] else ''})")
            for sr in info["skill_reads"]:
                print(f"  Step {sr['step_id']}: Read('{sr['file_path']}')")

    return {
        "used_count": used_count,
        "total_count": total_count,
        "usage_pct": usage_pct,
        "avg_gt_pct": avg_gt_pct,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check skill usage in trajectory files")
    parser.add_argument(
        "path",
        help="Path to a job directory or parent folder (with --batch)",
    )
    parser.add_argument("--batch", action="store_true", help="Process all subdirs with trajectory data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details per task")
    parser.add_argument("--json-output", "-j", type=str, help="Write results to JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path for batch summary (default: <path>/skill_usage_summary.json)")
    args = parser.parse_args()

    path = Path(args.path)

    if not args.batch:
        job_dir = path
        if not job_dir.is_dir():
            print(f"Error: {job_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        result = analyze_job(job_dir, verbose=args.verbose)
        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(result["results"], f, indent=2)
            print(f"\nJSON results written to {args.json_output}")
        return

    # Batch mode: iterate through subdirectories
    def _has_trajectory(d):
        for entry in os.listdir(d):
            ep = d / entry
            if ep.is_dir() and find_trajectory(ep)[0] is not None:
                return True
        return False

    job_dirs = sorted([
        d for d in path.iterdir()
        if d.is_dir() and _has_trajectory(d)
    ])

    if not job_dirs:
        print(f"No subdirectories with trajectory data found in {path}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    for job_dir in job_dirs:
        job_name = job_dir.name
        print(f"\n{'='*60}")
        print(f"Job: {job_name}")
        print(f"{'='*60}")
        try:
            result = analyze_job(job_dir, verbose=args.verbose)
            summary[job_name] = {
                "usage_pct": result["usage_pct"],
                "avg_gt_pct": result["avg_gt_pct"],
                "used_count": result["used_count"],
                "total_count": result["total_count"],
            }
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            summary[job_name] = {"error": str(e)}

    # Save summary
    output_path = Path(args.output) if args.output else path / "skill_usage_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    # Print compact table
    print(f"\n{'Job':<60} {'Usage%':>8} {'GT Skill%':>10} {'Used':>6} {'Total':>6}")
    print("-" * 95)
    for job_name, result in summary.items():
        if "error" in result:
            print(f"{job_name:<60} ERROR: {result['error']}")
        else:
            gt_str = f"{result['avg_gt_pct']:.1f}%" if result["avg_gt_pct"] is not None else "N/A"
            print(f"{job_name:<60} {result['usage_pct']:>7.1f}% {gt_str:>10} {result['used_count']:>6} {result['total_count']:>6}")


if __name__ == "__main__":
    main()
