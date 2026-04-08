#!/usr/bin/env python3
"""Query-agnostic skill refinement: improve skills independently using skill-creator.

Unlike query_specific_refinement.py which refines per-task, this script:
1. Finds unique skills across all tasks (deduplicating by name)
2. Improves each unique skill independently using skill-creator
3. Copies improved skills back to all tasks that use them

A per-model cache avoids re-refining skills that have already been improved.

Usage:
    # 1. Prepare temp task dirs and harbor config
    python scripts/query_agnostic_refinement.py prepare [options]

    # 2. Run harbor
    harbor run -c <config path printed by prepare>

    # 3. Collect results back to original task dirs
    python scripts/query_agnostic_refinement.py collect [options]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent

SKILL_CREATOR_REL = ".claude/skills/skill-creator"

_REFINED_SKILLS_PATH = {
    "qwen-coder": "/root/refined_skill",
    "terminus-2": "/root/refined_skill",
}
_DEFAULT_REFINED_SKILLS_PATH = "/agent-logs/refined_skill"

# Agent-specific skill paths inside Docker (same as find_skills_for_tasks.py)
SKILL_COPY_LINES = [
    "COPY skills /root/.claude/skills",
    "COPY skills /root/.codex/skills",
    "COPY skills /root/.opencode/skill",
    "COPY skills /root/.goose/skills",
    "COPY skills /root/.factory/skills",
    "COPY skills /root/.agents/skills",
    "COPY skills /root/.github/skills",
    "COPY skills /root/.gemini/skills",
    "COPY skills /root/.qwen/skills",
]

TASK_TOML = """\
version = "1.0"

[metadata]
author_name = "skill-refiner"
difficulty = "easy"
category = "skill-refinement"
tags = ["skill-refinement"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 2000.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 4096
storage_mb = 10240
"""


def get_refined_skill_path(agent: str) -> str:
    return _REFINED_SKILLS_PATH.get(agent, _DEFAULT_REFINED_SKILLS_PATH)


def get_cache_dir(workspace: Path, model: str) -> Path:
    """Return the cache directory for refined skills for a given model."""
    return workspace / "refined-skills-cache" / model


MINIMAL_DOCKERFILE = """\
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY skill_to_improve /root/skill_to_improve
{skill_copy_lines}
"""

INSTRUCTION_PROMPT = """\
You are a skill improvement agent. Your job is to improve a single skill.

The skill to improve is at /root/skill_to_improve/ (contains SKILL.md and possibly \
supporting files). A guide on how to create and improve skills (skill-creator) is \
available in your skills.

Read the skill to improve and the skill-creator guide. Follow the skill-creator \
methodology to generate sample test queries, then evaluate the skill using A/B testing \
as described in the guide. Finally, improve the skill based on what you find.

## Important Guidelines

- **No user interaction.** Work autonomously.
- **Single iteration.** One round of evaluation and improvement — do not loop multiple times.

## Output

Save the improved skill to {refined_skill_path}/. The skill should have a SKILL.md file \
and optional supporting files:

```
{refined_skill_path}/
├── SKILL.md
└── (optional supporting files like scripts/, references/, assets/)
```
"""


def _build_agent_kwargs(agent: str, port: int) -> dict:
    """Build the kwargs dict for the harbor agent config."""
    if agent == "terminus-2":
        return {
            "api_base": f"http://localhost:{port}/v1",
            "model_info": {"max_input_tokens": 253952},
        }
    elif agent == "qwen-coder":
        return {
            "version": "0.12.3",
            "api_key": "empty",
            "base_url": f"http://host.docker.internal:{port}/v1",
        }
    else:
        return {
            "version": {"claude-code": "2.1.19"}.get(agent, "latest"),
        }


def find_task_dirs(tasks_dir: Path) -> list[Path]:
    """Return sorted list of task directories containing instruction.md."""
    dirs = []
    for d in sorted(tasks_dir.iterdir()):
        if d.is_dir() and (d / "instruction.md").exists():
            dirs.append(d)
    return dirs


def find_unique_skills(
    task_dirs: list[Path], tasks_filter: set[str] | None = None
) -> dict[str, dict]:
    """Find unique skills across all tasks, deduplicating by name.

    Returns:
        dict: {skill_name: {"path": Path, "tasks": [task_name, ...]}}
    """
    unique: dict[str, dict] = {}

    for task_dir in task_dirs:
        if tasks_filter and task_dir.name not in tasks_filter:
            continue
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.is_dir():
            continue
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                continue
            name = skill_dir.name
            if name not in unique:
                unique[name] = {
                    "path": skill_dir,
                    "tasks": [],
                }
            unique[name]["tasks"].append(task_dir.name)

    return unique


def create_temp_skill_dir(
    skill_name: str,
    skill_info: dict,
    skill_creator_path: Path,
    temp_tasks_dir: Path,
    agent: str,
) -> Path:
    """Create a harbor-compatible temp task dir for improving a single skill."""
    task_dir = temp_tasks_dir / skill_name
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)

    # Copy the skill to improve into environment/
    shutil.copytree(skill_info["path"], env_dir / "skill_to_improve")

    # Copy skill-creator into environment/skills/ (agent picks it up naturally)
    skills_dir = env_dir / "skills" / "skill-creator"
    shutil.copytree(skill_creator_path, skills_dir)

    # Write Dockerfile: copy skill_to_improve + skills to all agent skill paths
    dockerfile_content = MINIMAL_DOCKERFILE.format(
        skill_copy_lines="\n".join(SKILL_COPY_LINES)
    )
    (env_dir / "Dockerfile").write_text(dockerfile_content)

    # Write task.toml (add skills_dir for terminus-2)
    task_toml = TASK_TOML
    if agent == "terminus-2":
        task_toml = task_toml.replace(
            "[environment]",
            '[environment]\nskills_dir = "/root/.claude/skills"',
        )
    (task_dir / "task.toml").write_text(task_toml)

    # Write instruction
    refined_skill_path = get_refined_skill_path(agent)
    (task_dir / "instruction.md").write_text(
        INSTRUCTION_PROMPT.format(refined_skill_path=refined_skill_path) + "\n"
    )

    # No-op test
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        '#!/bin/bash\necho "1" > /logs/verifier/reward.txt\n'
    )

    return task_dir


def generate_harbor_config(
    temp_tasks_dir: Path,
    jobs_dir: Path,
    agent: str,
    model: str,
    max_workers: int,
    skill_names: list[str] | None,
    port: int = 30000,
    split_index: int | None = None,
) -> Path:
    """Generate a harbor YAML config file and return its path."""
    import yaml

    config = {
        "job_name": "refine-skills-agnostic",
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "timeout_multiplier": 1.0,
        "debug": False,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": max_workers,
            "quiet": False,
            "retry": {
                "max_retries": 1,
                "exclude_exceptions": ["VerifierTimeoutError"],
                "wait_multiplier": 1.0,
                "min_wait_sec": 1.0,
                "max_wait_sec": 60.0,
            },
        },
        "environment": {
            "type": "docker",
            "force_build": False,
            "delete": True,
        },
        "artifacts": [
            get_refined_skill_path(agent),
        ],
        "verifier": {
            "disable": True,
        },
        "agents": [
            {
                "name": agent,
                "import_path": None,
                "model_name": model,
                "kwargs": _build_agent_kwargs(agent, port),
            },
        ],
        "datasets": [
            {
                "path": str(temp_tasks_dir),
                "task_names": skill_names,
            },
        ],
    }

    suffix = f"-{split_index}" if split_index is not None else ""
    config_path = temp_tasks_dir.parent / f"harbor-refine-skills-agnostic{suffix}.yaml"
    config_path.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )
    return config_path


def collect_results(
    jobs_dir: Path,
    unique_skills: dict[str, dict],
    original_tasks_dir: Path,
    cache_dir: Path,
) -> None:
    """Copy improved skills from harbor output back to all tasks and the cache."""
    job_dir = jobs_dir / "refine-skills-agnostic"
    if not job_dir.is_dir():
        print(
            f"Warning: harbor job output not found at {job_dir}", file=sys.stderr
        )
        return

    results = []

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        # Determine skill name from result.json or directory name
        result_file = trial_dir / "result.json"
        result_data = None
        if result_file.exists():
            try:
                result_data = json.loads(result_file.read_text())
                skill_name = result_data["task_name"]
            except (json.JSONDecodeError, KeyError):
                skill_name = trial_dir.name.rsplit("__", 1)[0]
        else:
            skill_name = trial_dir.name.rsplit("__", 1)[0]

        if skill_name not in unique_skills:
            continue

        skill_info = unique_skills[skill_name]
        error_msg = ""
        if result_data and result_data.get("exception_info"):
            exc = result_data["exception_info"]
            error_msg = (
                f"{exc.get('exception_type', 'Unknown')}: "
                f"{exc.get('exception_message', '')}"
            )

        # Check for refined skill in artifacts
        refined_src = trial_dir / "artifacts" / "refined_skill"
        if not refined_src.exists() or not refined_src.is_dir():
            results.append(
                {
                    "skill": skill_name,
                    "tasks": skill_info["tasks"],
                    "status": "kept_original",
                    "error": error_msg,
                }
            )
            continue

        # Resolve the refined skill root (may be directly or in a subdirectory)
        refined_root = None
        if (refined_src / "SKILL.md").exists():
            refined_root = refined_src
        else:
            for sub in sorted(refined_src.iterdir()):
                if sub.is_dir() and (sub / "SKILL.md").exists():
                    refined_root = sub
                    break

        if refined_root is None:
            results.append(
                {
                    "skill": skill_name,
                    "tasks": skill_info["tasks"],
                    "status": "no_skill_found",
                    "error": error_msg,
                }
            )
            continue

        # Save to cache
        cache_dest = cache_dir / skill_name
        if cache_dest.exists():
            shutil.rmtree(cache_dest)
        shutil.copytree(refined_root, cache_dest)

        # Copy improved skill back to all tasks that use it
        updated_tasks = []
        for task_name in skill_info["tasks"]:
            dest = original_tasks_dir / task_name / "environment" / "skills" / skill_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(refined_root, dest)
            updated_tasks.append(task_name)

        results.append(
            {
                "skill": skill_name,
                "tasks": updated_tasks,
                "status": "success",
                "error": error_msg,
            }
        )

    # Report missing skills
    found = {r["skill"] for r in results}
    for name, info in unique_skills.items():
        if name not in found:
            results.append(
                {"skill": name, "tasks": info["tasks"], "status": "missing", "error": ""}
            )

    # Print summary
    results.sort(key=lambda r: r["skill"])
    ok = sum(1 for r in results if r["status"] == "success")

    print(f"\n{'Skill':<40} {'Tasks':>5} {'Status'}")
    print("-" * 65)
    for r in results:
        line = f"{r['skill']:<40} {len(r['tasks']):>5} {r['status']}"
        if r.get("error"):
            line += f"  [{r['error']}]"
        print(line)
    print("-" * 65)
    print(f"Total: {len(results)} unique skills")
    print(f"Status: {Counter([r['status'] for r in results])}")
    print(f"Cache: {cache_dir}")


DEFAULT_WORKSPACE = ".local-workspace"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both subcommands."""
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="Directory containing tasks with environment/skills/",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Workspace directory for temp files and jobs (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help="Where harbor writes job output (default: <workspace>/refine-skills-jobs)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help="Only process skills from these task names",
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        metavar="SKILL",
        help="Only process these skill names (e.g., --skills docker-networking python-debugging)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Model to use for skill refinement (default: claude-opus-4-6)",
    )


def resolve_common_args(args):
    """Resolve tasks_dir, task_dirs, jobs_dir, and unique_skills from parsed args."""
    tasks_dir = REPO_ROOT / args.tasks_dir
    if not tasks_dir.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_dirs = find_task_dirs(tasks_dir)
    tasks_filter = set(args.tasks) if args.tasks else None
    unique_skills = find_unique_skills(task_dirs, tasks_filter)

    # Filter by skill name if requested
    if args.skills:
        requested = set(args.skills)
        unique_skills = {
            k: v for k, v in unique_skills.items() if k in requested
        }
        missing = requested - set(unique_skills.keys())
        if missing:
            print(
                f"Warning: skills not found: {', '.join(sorted(missing))}",
                file=sys.stderr,
            )

    if not unique_skills:
        print(f"No skills found in {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    workspace = REPO_ROOT / args.workspace
    jobs_dir = (
        Path(args.jobs_dir) if args.jobs_dir else workspace / "refine-skills-jobs"
    )

    return tasks_dir, task_dirs, unique_skills, jobs_dir, workspace


def cmd_prepare(args):
    """Prepare temp task directories and harbor config."""
    tasks_dir, task_dirs, unique_skills, jobs_dir, workspace = resolve_common_args(args)

    # Locate skill-creator
    skill_creator_path = REPO_ROOT / SKILL_CREATOR_REL
    if not skill_creator_path.exists() or not (skill_creator_path / "SKILL.md").exists():
        print(
            f"Error: cannot locate skill-creator at {skill_creator_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check cache for already-refined skills
    cache_dir = get_cache_dir(workspace, args.model)
    cached = []
    if not args.no_cache and cache_dir.is_dir():
        to_remove = []
        for name in unique_skills:
            cached_skill = cache_dir / name
            if cached_skill.is_dir() and (cached_skill / "SKILL.md").exists():
                cached.append(name)
                to_remove.append(name)
        for name in to_remove:
            del unique_skills[name]

    temp_base = workspace / "refine-skills-tmp"
    temp_tasks_dir = temp_base / "tasks"

    print(f"Found {len(unique_skills) + len(cached)} unique skills across tasks in {tasks_dir}")
    if cached:
        print(f"Skipping {len(cached)} already-refined skills (cached for model {args.model}):")
        for name in sorted(cached):
            print(f"  {name} (cached)")
    print(f"To refine: {len(unique_skills)} skills")
    print(f"Agent: {args.agent}, Model: {args.model}, Workers: {args.max_workers}")
    print(f"Jobs dir: {jobs_dir}")
    print(f"Cache dir: {cache_dir}")
    print()

    if not unique_skills:
        print("All skills already refined. Nothing to do.")
        return

    # Create temp task directories
    if temp_tasks_dir.exists():
        shutil.rmtree(temp_tasks_dir)
    temp_tasks_dir.mkdir(parents=True)

    for skill_name, skill_info in sorted(unique_skills.items()):
        create_temp_skill_dir(
            skill_name, skill_info, skill_creator_path, temp_tasks_dir, args.agent
        )
        task_list = ", ".join(skill_info["tasks"])
        print(f"  {skill_name} (used by: {task_list})")

    # Split skills into N folders for parallel processing
    n_splits = args.n_splits
    sorted_skills = sorted(unique_skills.keys())

    if n_splits <= 1:
        # Single config, no splitting
        skill_names_filter = list(unique_skills.keys()) if args.skills or args.tasks else None
        config_path = generate_harbor_config(
            temp_tasks_dir,
            jobs_dir,
            args.agent,
            args.model,
            args.max_workers,
            skill_names_filter,
            port=args.port,
        )
        print(f"\nHarbor config: {config_path}")
        print(f"\nNow run: harbor run -c {config_path}")
    else:
        # Split into N groups and create N config files with separate jobs_dirs
        chunks = [[] for _ in range(n_splits)]
        for i, name in enumerate(sorted_skills):
            chunks[i % n_splits].append(name)
        # Remove empty chunks (if n_splits > number of skills)
        chunks = [c for c in chunks if c]

        config_paths = []
        for idx, chunk in enumerate(chunks):
            split_jobs_dir = jobs_dir.parent / f"{jobs_dir.name}-{idx}"
            config_path = generate_harbor_config(
                temp_tasks_dir,
                split_jobs_dir,
                args.agent,
                args.model,
                args.max_workers,
                chunk,
                port=args.port,
                split_index=idx,
            )
            config_paths.append(config_path)

        print(f"\n{len(config_paths)} harbor configs created:")
        for cp in config_paths:
            print(f"  {cp}")
        print(f"\nRun each in parallel:")
        for cp in config_paths:
            print(f"  harbor run -c {cp}")


def cmd_collect(args):
    """Collect results from harbor output back to original task dirs."""
    tasks_dir, task_dirs, unique_skills, jobs_dir, workspace = resolve_common_args(args)

    cache_dir = get_cache_dir(workspace, args.model)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: For each skill, check if it already exists in cache and copy from there
    cached_skills = []
    remaining_skills = {}
    for skill_name, skill_info in unique_skills.items():
        cached_skill = cache_dir / skill_name
        if cached_skill.is_dir() and (cached_skill / "SKILL.md").exists():
            # Copy from cache to all tasks that use this skill
            for task_name in skill_info["tasks"]:
                dest = tasks_dir / task_name / "environment" / "skills" / skill_name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(cached_skill, dest)
            cached_skills.append(skill_name)
        else:
            remaining_skills[skill_name] = skill_info

    if cached_skills:
        print(f"Copied {len(cached_skills)} skills from cache:")
        for name in sorted(cached_skills):
            print(f"  {name} (cached)")

    # Phase 2: For remaining skills not in cache, collect from job_dir
    if remaining_skills:
        split_dirs = sorted(jobs_dir.parent.glob(f"{jobs_dir.name}-[0-9]*"))
        if split_dirs:
            for split_dir in split_dirs:
                print(f"Collecting from {split_dir} ...")
                collect_results(split_dir, remaining_skills, tasks_dir, cache_dir)
        else:
            collect_results(jobs_dir, remaining_skills, tasks_dir, cache_dir)
    else:
        print("All skills found in cache. Nothing to collect from job dirs.")

    # Cleanup temp dirs
    temp_base = workspace / "refine-skills-tmp"
    if not args.keep_temp and temp_base.exists():
        shutil.rmtree(temp_base)
        print(f"\nCleaned up temp dirs at {temp_base}")
    elif temp_base.exists():
        print(f"\nTemp dirs kept at {temp_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Refine skills using skill-creator methodology (per-skill, deduplicated by name)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare subcommand
    prepare_parser = subparsers.add_parser(
        "prepare", help="Create temp task dirs and harbor config"
    )
    add_common_args(prepare_parser)
    prepare_parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Max parallel containers (default: 5)",
    )
    prepare_parser.add_argument(
        "--agent",
        default="claude-code",
        help="Harbor agent name (default: claude-code)",
    )
    prepare_parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port for local LLM endpoint (default: 30000). Used for qwen-coder and terminus-2.",
    )
    prepare_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and re-refine all skills",
    )
    prepare_parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Split skills into N groups with separate config files for parallel harbor runs (default: 1)",
    )

    # collect subcommand
    collect_parser = subparsers.add_parser(
        "collect", help="Collect results from harbor output"
    )
    add_common_args(collect_parser)
    collect_parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Don't delete temp task directories after collection",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
