#!/usr/bin/env python3
"""Find skills for tasks by running agents inside Docker containers via harbor.

Usage:
    # 1. Prepare temp task dirs and harbor config
    python scripts/find_skills_for_tasks.py prepare [options]

    # 2. Run harbor yourself
    harbor run -c <config path printed by prepare>

    # 3. Collect results back to original task dirs
    python scripts/find_skills_for_tasks.py collect [options]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "skill-finder-task-template"

_FOUND_SKILLS_PATH = {
    "qwen-coder": "/root/found_skills.json",
    "kimi-cli": "/root/found_skills.json",
    "terminus-2": "/root/found_skills.json",
}
_DEFAULT_FOUND_SKILLS_PATH = "/agent-logs/found_skills.json"


def get_instruction_prompt(agent: str) -> str:
    path = _FOUND_SKILLS_PATH.get(agent, _DEFAULT_FOUND_SKILLS_PATH)
    return (
        "Read the task description in task_instruction.md. Find relevant skills for "
        "completing this task. Aim for 10 relevant skills.\n\n"
        f"Finally, list all found skill_id in {path} as a JSON "
        "array of strings, sorted by relevance (most relevant first)."
    )

# Files to exclude when copying task environment data
EXCLUDED_ENV_FILES = {"Dockerfile", "docker-compose.yaml", "docker-compose.yml", "skills"}

# Agent-specific skill paths inside Docker
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


def _build_agent_kwargs(agent: str, port: int) -> dict:
    """Build the kwargs dict for the harbor agent config."""
    if agent == "terminus-2":
        return {
            "api_base": f"http://localhost:{port}/v1",
            "model_info": {"max_input_tokens": 253952},
        }
    elif agent in ("qwen-coder", "kimi-cli"):
        return {
            "version": {"qwen-coder": "0.12.3", "kimi-cli": "1.27.0"}[agent],
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


SKILL_VARIANTS = {
    "hybrid": "find-skills",
    "keyword": "find-skills-keyword",
    "semantic": "find-skills-semantic",
}


def find_skill_source(variant: str = "hybrid") -> Path:
    """Return the path to the find-skills SKILL.md source file for the given variant."""
    skill_dir_name = SKILL_VARIANTS.get(variant)
    if not skill_dir_name:
        print(f"Error: unknown skill variant '{variant}', choose from {list(SKILL_VARIANTS)}", file=sys.stderr)
        sys.exit(1)
    for rel in ("data",):
        candidate = REPO_ROOT / rel / skill_dir_name / "SKILL.md"
        if candidate.exists():
            return candidate
    print(f"Error: cannot locate {skill_dir_name}/SKILL.md", file=sys.stderr)
    sys.exit(1)


def create_temp_task_dir(
    task_dir: Path, skill_source: Path, temp_tasks_dir: Path, agent: str = "claude-code"
) -> Path:
    """Create a harbor-compatible temp task directory for skill finding."""
    task_name = task_dir.name
    temp_dir = temp_tasks_dir / task_name
    temp_env = temp_dir / "environment"
    temp_env.mkdir(parents=True, exist_ok=True)

    # Copy template files (add skills_dir for terminus-2)
    task_toml = (TEMPLATE_DIR / "task.toml").read_text()
    if agent == "terminus-2":
        task_toml = task_toml.replace(
            "[environment]",
            '[environment]\nskills_dir = "/root/.claude/skills"',
        )
    (temp_dir / "task.toml").write_text(task_toml)
    tests_dir = temp_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "tests" / "test.sh", tests_dir / "test.sh")

    # Copy original instruction.md as task_instruction.md in environment
    shutil.copy2(
        task_dir / "instruction.md", temp_env / "task_instruction.md"
    )

    # Copy task-specific data files from environment/ (excluding Docker/skills)
    src_env = task_dir / "environment"
    data_files = []  # relative paths of copied data files/dirs
    if src_env.is_dir():
        for item in sorted(src_env.iterdir()):
            if item.name in EXCLUDED_ENV_FILES:
                continue
            dest = temp_env / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
            data_files.append(item.name)

    # Copy find-skills SKILL.md, rewriting localhost for macOS Docker
    skills_dest = temp_env / "skills" / "find-skills"
    skills_dest.mkdir(parents=True, exist_ok=True)
    skill_content = skill_source.read_text()
    skill_content = skill_content.replace(
        "localhost:8742", "host.docker.internal:8742"
    )
    if agent == "kimi-cli":
        skill_content += (
            "\n\n## URL Format Reminder\n\n"
            "**Important:** Always use `?` to start query parameters, not `&`.\n"
            "- Correct: `curl -s \"http://host.docker.internal:8742/hybrid?q=react+hooks&top_k=10\"`\n"
            "- Wrong: `curl -s \"http://host.docker.internal:8742/hybrid&q=react+hooks&top_k=10\"`\n"
        )
    (skills_dest / "SKILL.md").write_text(skill_content)

    # Generate Dockerfile
    dockerfile_lines = [
        "FROM ubuntu:24.04",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y curl jq && rm -rf /var/lib/apt/lists/*",
        "WORKDIR /root",
        "COPY task_instruction.md /root/task_instruction.md",
    ]
    for name in data_files:
        dockerfile_lines.append(f'COPY ["{name}", "/root/{name}"]')
    dockerfile_lines.extend(SKILL_COPY_LINES)
    (temp_env / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n")

    # Create instruction.md with the skill-finding prompt
    (temp_dir / "instruction.md").write_text(get_instruction_prompt(agent) + "\n")

    return temp_dir


def generate_harbor_config(
    temp_tasks_dir: Path,
    jobs_dir: Path,
    agent: str,
    model: str,
    max_workers: int,
    task_names: list[str] | None,
    port: int = 30000,
) -> Path:
    """Generate a harbor YAML config file and return its path."""
    import yaml

    config = {
        "job_name": "find-skills",
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
        "artifacts": [_FOUND_SKILLS_PATH.get(agent, _DEFAULT_FOUND_SKILLS_PATH)],
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
                "task_names": task_names,
            },
        ],
    }

    config_path = temp_tasks_dir.parent / "harbor-find-skills.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    return config_path


def collect_results(
    jobs_dir: Path, task_dirs: list[Path], original_tasks_dir: Path
) -> None:
    """Copy found_skills.json from harbor job output back to original task dirs."""
    find_skills_job_dir = jobs_dir / "find-skills"
    if not find_skills_job_dir.is_dir():
        print(f"Warning: harbor job output not found at {find_skills_job_dir}", file=sys.stderr)
        return

    # Build a mapping from task name to original task dir
    task_map = {d.name: d for d in task_dirs}

    results = []
    # Scan job output directories for found_skills.json
    for job_dir in sorted(find_skills_job_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        # Harbor may truncate long task names in dir names, so read
        # the actual task_name from result.json when available.
        result_file = job_dir / "result.json"
        if result_file.exists():
            try:
                task_name = json.loads(result_file.read_text())["task_name"]
            except (json.JSONDecodeError, KeyError):
                task_name = job_dir.name.rsplit("__", 1)[0]
        else:
            task_name = job_dir.name.rsplit("__", 1)[0]

        if task_name not in task_map:
            continue

        found_skills_src = job_dir / "artifacts" / "found_skills.json"
        if not found_skills_src.exists():
            results.append({"task": task_name, "status": "no_output", "skills_count": 0})
            shutil.rmtree(job_dir)
            print(f"  Removed failed job dir: {job_dir.name}")
            continue

        # Copy to original task dir
        dest = task_map[task_name] / "found_skills.json"
        shutil.copy2(found_skills_src, dest)

        try:
            skills = json.loads(found_skills_src.read_text())
            count = len(skills) if isinstance(skills, list) else 0
            results.append({"task": task_name, "status": "success", "skills_count": count})
        except json.JSONDecodeError:
            results.append({"task": task_name, "status": "bad_json", "skills_count": 0})

    # Report tasks with no job output at all
    found_tasks = {r["task"] for r in results}
    for task_name in sorted(task_map.keys()):
        if task_name not in found_tasks:
            results.append({"task": task_name, "status": "missing", "skills_count": 0})

    # Print summary
    results.sort(key=lambda r: r["task"])
    ok = sum(1 for r in results if r["status"] == "success")
    fail = len(results) - ok

    print(f"\n{'Task':<50} {'Skills':>6} {'Status'}")
    print("-" * 72)
    for r in results:
        print(f"{r['task']:<50} {r['skills_count']:>6} {r['status']}")
    print("-" * 72)
    print(f"Total: {len(results)} tasks, {ok} succeeded, {fail} failed")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both subcommands."""
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="Directory containing tasks",
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help="Where harbor writes job output (default: .local-workspace/find-skills-jobs)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help="Only process these task names (e.g., --tasks fix-build-google-auto lean4-proof)",
    )


def resolve_common_args(args):
    """Resolve tasks_dir, task_dirs, and jobs_dir from parsed args."""
    tasks_dir = REPO_ROOT / args.tasks_dir
    if not tasks_dir.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_dirs = find_task_dirs(tasks_dir)

    if args.tasks:
        requested = set(args.tasks)
        task_dirs = [d for d in task_dirs if d.name in requested]
        missing = requested - {d.name for d in task_dirs}
        if missing:
            print(f"Warning: tasks not found: {', '.join(sorted(missing))}", file=sys.stderr)
    if not task_dirs:
        print(f"No task directories found in {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else REPO_ROOT / ".local-workspace" / "find-skills-jobs"

    return tasks_dir, task_dirs, jobs_dir


def cmd_prepare(args):
    """Prepare temp task directories and harbor config."""
    tasks_dir, task_dirs, jobs_dir = resolve_common_args(args)
    skill_source = find_skill_source(args.skill_variant)
    temp_base = REPO_ROOT / ".local-workspace" / "find-skills-tmp"
    temp_tasks_dir = temp_base / "tasks"

    print(f"Found {len(task_dirs)} tasks in {tasks_dir}")
    print(f"find-skills source: {skill_source}")
    print(f"Agent: {args.agent}, Model: {args.model}, Workers: {args.max_workers}")
    print(f"Jobs dir: {jobs_dir}")
    print()

    # Create temp task directories
    if temp_tasks_dir.exists():
        shutil.rmtree(temp_tasks_dir)
    temp_tasks_dir.mkdir(parents=True)

    for task_dir in task_dirs:
        create_temp_task_dir(task_dir, skill_source, temp_tasks_dir, agent=args.agent)
        print(f"  Created temp task: {task_dir.name}")

    # Generate harbor config
    task_names_filter = args.tasks if args.tasks else None
    config_path = generate_harbor_config(
        temp_tasks_dir, jobs_dir, args.agent, args.model, args.max_workers, task_names_filter,
        port=args.port,
    )
    print(f"\nHarbor config: {config_path}")
    print(f"\nNow run: harbor run -c {config_path}")


def cmd_collect(args):
    """Collect results from harbor output back to original task dirs."""
    tasks_dir, task_dirs, jobs_dir = resolve_common_args(args)

    collect_results(jobs_dir, task_dirs, tasks_dir)

    # Cleanup temp dirs
    temp_base = REPO_ROOT / ".local-workspace" / "find-skills-tmp"
    if not args.keep_temp and temp_base.exists():
        shutil.rmtree(temp_base)
        print(f"\nCleaned up temp dirs at {temp_base}")
    elif temp_base.exists():
        print(f"\nTemp dirs kept at {temp_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Find skills for tasks using harbor-managed Docker containers"
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
        "--model",
        default="claude-opus-4-6",
        help="Model to use for skill finding (default: claude-opus-4-6)",
    )
    prepare_parser.add_argument(
        "--agent",
        default="claude-code",
        help="Harbor agent name (default: claude-code)",
    )
    prepare_parser.add_argument(
        "--skill-variant",
        choices=list(SKILL_VARIANTS),
        default="hybrid",
        help="Search variant: hybrid (default), keyword, or semantic",
    )
    prepare_parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port for local LLM endpoint (default: 30000). Used for qwen-coder, kimi-cli, and terminus-2.",
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
