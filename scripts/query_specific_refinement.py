#!/usr/bin/env python3
"""Refine retrieved skills for tasks by running agents inside task Docker containers via harbor.

Unlike retrieve_skills.py which uses a minimal Docker container for search,
this script uses the task's own Dockerfile so the agent can explore and test skills
in the same environment used to complete the task.

Usage:
    # 1. Prepare temp task dirs and harbor config
    python scripts/query_specific_refinement.py prepare [options]

    # 2. Run harbor yourself
    harbor run -c <config path printed by prepare>

    # 3. Collect results back to original task dirs
    python scripts/query_specific_refinement.py collect [options]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Skills to include for refinement guidance
SKILL_CREATOR_REL = ".claude/skills/skill-creator"

_REFINED_SKILLS_PATH = {
    "qwen-coder": "/root/refined_skills",
    "terminus-2": "/root/refined_skills",
}
_DEFAULT_REFINED_SKILLS_PATH = "/agent-logs/refined_skills"

_AGENT_SKILLS_PATH = {
    "qwen-coder": "/root/.qwen/skills",
    "terminus-2": "/root/.claude/skills",
}
_DEFAULT_AGENT_SKILLS_PATH = "/root/.claude/skills"


def get_refined_skills_path(agent: str) -> str:
    return _REFINED_SKILLS_PATH.get(agent, _DEFAULT_REFINED_SKILLS_PATH)


def get_agent_skills_path(agent: str) -> str:
    return _AGENT_SKILLS_PATH.get(agent, _DEFAULT_AGENT_SKILLS_PATH)


INSTRUCTION_PROMPT = """\
You are a skill refinement agent. Your goal is to attempt the target task \
using the retrieved skills, observe which parts of the skills help and which \
don't, and then create improved refined skills based on that experience.

## Your Task

### Phase 1: Understand the task and skills

1. Read the task description in /root/task_instruction.md to understand what the task requires.
2. Read ALL the retrieved skills in /root/retrieved_skills/. Each subdirectory contains a \
skill with a SKILL.md and possibly supporting files (scripts, references, etc.).

### Phase 2: Attempt the task using the retrieved skills

3. Try to solve the task while actively consulting the retrieved skills. This is the most \
important step. As you work through the task:
   - Refer to the retrieved skills for guidance, code snippets, API patterns, and domain knowledge.
   - When a skill suggests an approach, try it. Note whether it works, partially works, or is wrong.
   - When you get stuck, check if any skill covers the issue. Note gaps where no skill helps.
   - Keep track of which specific parts of which skills were useful, misleading, or irrelevant.

   IMPORTANT: If you delegate any part of the exploration to a subagent, you MUST give that \
subagent access to the retrieved skills at /root/retrieved_skills/ and instruct it to \
consult them during its work. The goal is to test the skills in practice, not to solve the \
task from scratch independently.

### Phase 3: Reflect and create refined skills

4. Based on your experience attempting the task with the retrieved skills, reflect on:
   - Which skills or parts of skills were directly useful?
   - Which skills had errors, outdated information, or misleading guidance?
   - What knowledge was missing that you had to figure out on your own?
   - What would have made the task easier if you had known it upfront?

5. Use the skill-creator skill at {agent_skills_path}/skill-creator/ \
as guidance for creating and writing skills.

6. Create refined skills that incorporate what you learned. The refined skills should:
   - Keep the parts that actually worked when you tried them.
   - Fix or remove parts that were wrong or misleading.
   - Add knowledge you discovered during exploration that was missing from the original skills.
   - Combine related information from multiple skills into coherent, task-appropriate guides.

## Important Guidelines

- **Focus on this task only.** You are preparing skills specifically for this given task. \
There is no need to create additional test queries — just test and evaluate against the \
task in /root/task_instruction.md. You do not have access to the ground-truth verifier; \
judge quality based on your own knowledge and exploration of the task.
- **Single iteration of improvement.** Do one round of exploration and refinement — do \
not iterate multiple times.
- **No user interaction.** Do not ask any questions. Self-explore the target task and \
create improved skills based on your exploration trajectory.
- **Compose, don't copy.** The refined skills do not need to cover all information in the \
retrieved skills. Instead, extract and compose the useful, relevant parts and combine them \
into coherent skills. There is no limit on the number of skills you create — you can \
create more or fewer than the number of retrieved skills. Focus on quality and relevance.

## Output

Save your refined skills to {refined_skills_path}/. Each skill should be in its own \
subdirectory with a SKILL.md file (and optional supporting files like scripts or references):

```
{refined_skills_path}/
├── skill-name-1/
│   ├── SKILL.md
│   └── (optional supporting files)
├── skill-name-2/
│   ├── SKILL.md
│   └── (optional supporting files)
└── ...
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


def find_guide_skills() -> dict[str, Path]:
    """Return a dict of {skill-name: path} for guide skills to include."""
    candidate = REPO_ROOT / SKILL_CREATOR_REL
    if candidate.exists() and (candidate / "SKILL.md").exists():
        return {"skill-creator": candidate}
    print(f"Error: cannot locate skill-creator at {candidate}", file=sys.stderr)
    sys.exit(1)


def find_existing_skills(task_dir: Path) -> list[str]:
    """Return list of skill directory names already present in environment/skills/."""
    skills_dir = task_dir / "environment" / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(d.name for d in skills_dir.iterdir() if d.is_dir())


def create_temp_task_dir(
    task_dir: Path,
    guide_skills: dict[str, Path],
    temp_tasks_dir: Path,
    agent: str = "claude-code",
) -> Path:
    """Create a harbor-compatible temp task directory for skill refinement.

    Copies the entire original task, then layers on:
    - task_instruction.md (copy of instruction.md)
    - retrieved_skills/ (copy of the existing skills)
    - guide skills added into environment/skills/
    - Dockerfile appended with extra COPY lines
    - instruction.md replaced with the refinement prompt
    - tests/ replaced with a no-op verifier
    """
    task_name = task_dir.name
    temp_dir = temp_tasks_dir / task_name
    temp_env = temp_dir / "environment"

    # Copy the entire task directory as-is
    shutil.copytree(task_dir, temp_dir)

    # Copy original instruction.md as task_instruction.md in environment
    shutil.copy2(task_dir / "instruction.md", temp_env / "task_instruction.md")

    # Move existing skills into retrieved_skills/ (agent reads these as input)
    src_skills = temp_env / "skills"
    retrieved_dest = temp_env / "retrieved_skills"
    if src_skills.is_dir():
        src_skills.rename(retrieved_dest)

    # Create environment/skills/ with only the guide skills
    src_skills.mkdir(parents=True, exist_ok=True)
    for name, path in guide_skills.items():
        shutil.copytree(path, src_skills / name)

    # Append extra COPY lines to the Dockerfile
    dockerfile = temp_env / "Dockerfile"
    if dockerfile.exists():
        extra_lines = [
            "",
            "# Refinement agent: task instruction and retrieved skills",
            "COPY task_instruction.md /root/task_instruction.md",
            "COPY retrieved_skills /root/retrieved_skills",
        ]
        with open(dockerfile, "a") as f:
            f.write("\n".join(extra_lines) + "\n")

    # Replace instruction.md with the refinement prompt
    refined_skills_path = get_refined_skills_path(agent)
    agent_skills_path = get_agent_skills_path(agent)
    formatted_prompt = INSTRUCTION_PROMPT.format(
        refined_skills_path=refined_skills_path,
        agent_skills_path=agent_skills_path,
    )
    (temp_dir / "instruction.md").write_text(formatted_prompt + "\n")

    # For terminus-2, set skills_dir in task.toml so the agent discovers skills
    if agent == "terminus-2":
        task_toml = temp_dir / "task.toml"
        if task_toml.exists():
            toml_text = task_toml.read_text()
            if "skills_dir" not in toml_text:
                agent_skills_path = get_agent_skills_path(agent)
                toml_text = toml_text.replace(
                    "[environment]",
                    f'[environment]\nskills_dir = "{agent_skills_path}"',
                )
                task_toml.write_text(toml_text)

    # Replace tests/ with a no-op verifier
    tests_dir = temp_dir / "tests"
    if tests_dir.exists():
        shutil.rmtree(tests_dir)
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text('#!/bin/bash\necho "1" > /logs/verifier/reward.txt\n')

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
        "job_name": "refine-skills",
        "jobs_dir": str(jobs_dir),
        "n_attempts": 1,
        "timeout_multiplier": 2.0,
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
            get_refined_skills_path(agent),
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
                "task_names": task_names,
                "exclude_task_names": [
                    "mhc-layer-impl",
                    "scheduling-email-assistant",
                    "fix-visual-stability",
                ],
            },
        ],
    }

    config_path = temp_tasks_dir.parent / "harbor-refine-skills.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    return config_path


def collect_results(
    jobs_dir: Path, task_dirs: list[Path], original_tasks_dir: Path
) -> None:
    """Copy refined skills from harbor job output back to original task dirs."""
    refine_skills_job_dir = jobs_dir / "refine-skills"
    if not refine_skills_job_dir.is_dir():
        print(f"Warning: harbor job output not found at {refine_skills_job_dir}", file=sys.stderr)
        return

    task_map = {d.name: d for d in task_dirs}

    results = []
    for job_dir in sorted(refine_skills_job_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        # Determine task name and exception info from result.json or directory name
        result_file = job_dir / "result.json"
        result_data = None
        if result_file.exists():
            try:
                result_data = json.loads(result_file.read_text())
                task_name = result_data["task_name"]
            except (json.JSONDecodeError, KeyError):
                task_name = job_dir.name.rsplit("__", 1)[0]
        else:
            task_name = job_dir.name.rsplit("__", 1)[0]

        if task_name not in task_map:
            continue

        # Extract error info if present
        error_msg = ""
        if result_data and result_data.get("exception_info"):
            exc = result_data["exception_info"]
            error_msg = f"{exc.get('exception_type', 'Unknown')}: {exc.get('exception_message', '')}"

        # Check for refined skills directory
        refined_src = job_dir / "artifacts" / "refined_skills"
        dest_skills = task_map[task_name] / "environment" / "skills"

        if not refined_src.exists() or not refined_src.is_dir():
            # No refined skills directory at all — keep original skills as-is
            skill_count = sum(
                1 for d in dest_skills.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
            ) if dest_skills.is_dir() else 0
            results.append({"task": task_name, "status": "kept_original", "skills_count": skill_count, "error": error_msg})
            continue

        # Replace environment/skills/ with refined skills
        if dest_skills.exists():
            shutil.rmtree(dest_skills)
        shutil.copytree(refined_src, dest_skills)

        # Count skills
        skill_count = sum(1 for d in dest_skills.iterdir() if d.is_dir() and (d / "SKILL.md").exists())
        results.append({"task": task_name, "status": "success", "skills_count": skill_count, "error": error_msg})

    # Report tasks with no job output
    found_tasks = {r["task"] for r in results}
    for task_name in sorted(task_map.keys()):
        if task_name not in found_tasks:
            results.append({"task": task_name, "status": "missing", "skills_count": 0, "error": ""})

    # Print summary
    results.sort(key=lambda r: r["task"])
    ok = sum(1 for r in results if r["status"] == "success")
    fail = len(results) - ok

    print(f"\n{'Task':<50} {'Skills':>6} {'Status'}")
    print("-" * 72)
    for r in results:
        line = f"{r['task']:<50} {r['skills_count']:>6} {r['status']}"
        if r.get("error"):
            line += f"  [{r['error']}]"
        print(line)
    print("-" * 72)
    print(f"Total: {len(results)} tasks, {ok} succeeded, {fail} failed")


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
        help="Only process these task names (e.g., --tasks fix-build-google-auto lean4-proof)",
    )


def resolve_common_args(args):
    """Resolve tasks_dir, task_dirs, and jobs_dir from parsed args."""
    tasks_dir = REPO_ROOT / args.tasks_dir
    if not tasks_dir.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_dirs = find_task_dirs(tasks_dir)

    # Filter to only tasks that have environment/skills/
    task_dirs = [d for d in task_dirs if (d / "environment" / "skills").is_dir()]

    if args.tasks:
        requested = set(args.tasks)
        task_dirs = [d for d in task_dirs if d.name in requested]
        missing = requested - {d.name for d in task_dirs}
        if missing:
            print(f"Warning: tasks not found (or missing environment/skills/): {', '.join(sorted(missing))}", file=sys.stderr)
    if not task_dirs:
        print(f"No task directories with environment/skills/ found in {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    workspace = REPO_ROOT / args.workspace
    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else workspace / "refine-skills-jobs"

    return tasks_dir, task_dirs, jobs_dir, workspace


def cmd_prepare(args):
    """Prepare temp task directories and harbor config."""
    tasks_dir, task_dirs, jobs_dir, workspace = resolve_common_args(args)
    guide_skills = find_guide_skills()
    temp_base = workspace / "refine-skills-tmp"
    temp_tasks_dir = temp_base / "tasks"

    print(f"Found {len(task_dirs)} tasks with environment/skills/ in {tasks_dir}")
    print(f"Guide skills: {', '.join(f'{n} ({p})' for n, p in guide_skills.items())}")
    print(f"Agent: {args.agent}, Model: {args.model}, Workers: {args.max_workers}")
    print(f"Jobs dir: {jobs_dir}")
    print()

    # Create temp task directories
    if temp_tasks_dir.exists():
        shutil.rmtree(temp_tasks_dir)
    temp_tasks_dir.mkdir(parents=True)

    for task_dir in task_dirs:
        existing_skills = find_existing_skills(task_dir)
        create_temp_task_dir(task_dir, guide_skills, temp_tasks_dir, agent=args.agent)
        print(f"  Created temp task: {task_dir.name} ({len(existing_skills)} skills)")

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
    tasks_dir, task_dirs, jobs_dir, workspace = resolve_common_args(args)

    collect_results(jobs_dir, task_dirs, tasks_dir)

    # Cleanup temp dirs
    temp_base = workspace / "refine-skills-tmp"
    if not args.keep_temp and temp_base.exists():
        shutil.rmtree(temp_base)
        print(f"\nCleaned up temp dirs at {temp_base}")
    elif temp_base.exists():
        print(f"\nTemp dirs kept at {temp_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Refine retrieved skills for tasks using harbor-managed Docker containers"
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
        help="Model to use for skill refinement (default: claude-opus-4-6)",
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
