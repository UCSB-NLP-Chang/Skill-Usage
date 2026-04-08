#!/usr/bin/env python3
"""Judge how well the combined set of skills covers a task.

For each task, sends the task instruction and ALL skills together to a model
and asks it to rate overall coverage on a 1-5 scale:
  5 = Complete coverage — skills cover all steps/aspects needed to solve the task.
  4 = High coverage — skills cover most aspects, minor gaps remain.
  3 = Moderate coverage — skills cover some key aspects but miss significant parts.
  2 = Low coverage — skills touch on the topic but miss most of the task's needs.
  1 = No coverage — nothing in the skills is relevant to the task.

Results are saved as a single JSON file (coverage_judgments.json) in the top-level
tasks directory.

Usage:
    python scripts/judge_skill_coverage.py <tasks_dir>
    python scripts/judge_skill_coverage.py <tasks_dir> --tasks task1 task2
    python scripts/judge_skill_coverage.py <tasks_dir> --model gpt-5.4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "gpt-5.4"
MAX_HELPER_CHARS = 20000  # Per helper file
MAX_SKILL_MD_CHARS = 200000  # Max chars for SKILL.md
MAX_PER_SKILL_CHARS = 400000  # Max chars for a single skill's content
MAX_SKILL_TOTAL_CHARS = 2000000  # Total chars across all skills for one task

SYSTEM_PROMPT = """\
You are an expert evaluator assessing how well a set of skill documents \
collectively covers a specific task. A "skill" is a reusable knowledge document \
(with optional helper scripts/references) that an AI agent can consult while \
working on a task.

You will be given a task instruction and a set of skills. Evaluate how well \
the skills TOGETHER cover what is needed to complete the task.

Rate overall coverage on this scale:
  5 = Complete coverage — the skills together cover all steps and aspects \
needed to solve the task. An agent with these skills has everything it needs.
  4 = High coverage — the skills cover most aspects of the task, but minor \
gaps remain that the agent would need to figure out on its own.
  3 = Moderate coverage — the skills cover some key aspects but miss \
significant parts of the task. The agent would need substantial independent work.
  2 = Low coverage — the skills touch on the topic but miss most of the \
task's specific needs. Only marginally helpful.
  1 = No coverage — nothing in the skills is relevant to the task.

Respond with ONLY a JSON object, no explanation:
{"score": <1|2|3|4|5>, "covered": "<what the skills cover>", "gaps": "<what is missing>"}"""

DEFAULT_EXCLUDE_TASKS = {"mhc-layer-impl", "scheduling-email-assistant", "fix-visual-stability"}


def find_task_dirs(
    base_dir: Path,
    task_filter: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[Path]:
    """Find task directories that have both instruction.md and skills/."""
    if exclude is None:
        exclude = set()
    dirs = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in exclude:
            continue
        if task_filter and entry.name not in task_filter:
            continue
        instruction = entry / "instruction.md"
        skills_dir = entry / "environment" / "skills"
        if instruction.exists() and skills_dir.is_dir():
            dirs.append(entry)
    return dirs


def find_skills(task_dir: Path) -> list[Path]:
    """Find skill directories (containing SKILL.md) within a task."""
    skills_dir = task_dir / "environment" / "skills"
    if not skills_dir.is_dir():
        return []
    results = []
    for entry in sorted(skills_dir.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").exists():
            results.append(entry)
    return results


def _fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    elif b < 1024 * 1024:
        return f"{b/1024:.1f}KB"
    else:
        return f"{b/(1024*1024):.1f}MB"


def build_skill_content(skill_dir: Path, max_helper_chars: int, max_per_skill: int, remaining: int) -> str:
    """Build a textual representation of a single skill's content."""
    budget = min(max_per_skill, remaining)
    parts: list[str] = []

    # File tree
    tree_lines = []
    for root, dirs, files in os.walk(skill_dir):
        level = len(Path(root).relative_to(skill_dir).parts)
        indent = "  " * level
        if level == 0:
            tree_lines.append(f"{skill_dir.name}/")
        else:
            tree_lines.append(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            fp = Path(root) / f
            tree_lines.append(f"{indent}  {f}  ({_fmt_size(fp.stat().st_size)})")
    parts.append("### File Structure\n```\n" + "\n".join(tree_lines) + "\n```\n")

    # SKILL.md
    skill_md = skill_dir / "SKILL.md"
    skill_md_text = skill_md.read_text(errors="replace")
    if len(skill_md_text) > MAX_SKILL_MD_CHARS:
        skill_md_text = skill_md_text[:MAX_SKILL_MD_CHARS] + f"\n... (truncated, {_fmt_size(skill_md.stat().st_size)} total)"
    parts.append(f"### SKILL.md\n```\n{skill_md_text}\n```\n")

    # Helper files
    chars_used = sum(len(p) for p in parts)
    hlp_remaining = budget - chars_used

    for root, _dirs, files in os.walk(skill_dir):
        for f in sorted(files):
            if f == "SKILL.md":
                continue
            if hlp_remaining <= 200:
                parts.append(f"\n### {f}\n(truncated — content limit reached)\n")
                continue
            fp = Path(root) / f
            rel_path = fp.relative_to(skill_dir)
            try:
                content = fp.read_text(errors="replace")
            except Exception:
                parts.append(f"\n### {rel_path}\n(binary or unreadable file)\n")
                continue
            per_file_limit = min(max_helper_chars, hlp_remaining)
            if len(content) > per_file_limit:
                content = content[:per_file_limit] + f"\n... (truncated, {_fmt_size(fp.stat().st_size)} total)"
            entry = f"\n### {rel_path}\n```\n{content}\n```\n"
            parts.append(entry)
            hlp_remaining -= len(entry)

    result = "".join(parts)
    if len(result) > budget:
        result = result[:budget] + f"\n... (skill truncated at {_fmt_size(budget)})"
    return result


def build_all_skills_content(
    task_dir: Path, max_helper_chars: int, max_per_skill: int, max_total_chars: int,
) -> str:
    """Build combined content of all skills for a task."""
    skills = find_skills(task_dir)
    if not skills:
        return "(no skills found)"

    parts: list[str] = []
    remaining = max_total_chars

    for i, skill_dir in enumerate(skills):
        header = f"\n## Skill {i+1}: {skill_dir.name}\n\n"
        parts.append(header)
        remaining -= len(header)

        if remaining <= 500:
            parts.append("(remaining skills truncated — total content limit reached)\n")
            break

        skill_content = build_skill_content(skill_dir, max_helper_chars, max_per_skill, remaining)
        parts.append(skill_content)
        remaining -= len(skill_content)

    return "".join(parts)


def judge_task(
    client: openai.OpenAI,
    model: str,
    task_instruction: str,
    task_name: str,
    all_skills_content: str,
) -> dict:
    """Call the LLM to judge overall skill coverage for a task."""
    user_message = (
        f"# Task Instruction\n\n{task_instruction}\n\n"
        f"---\n\n"
        f"# Skills Provided\n\n{all_skills_content}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=10240,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as e:
        print(f"    API error for {task_name}: {e}", file=sys.stderr)
        print(f"    model={model}, user_message length={len(user_message)} chars", file=sys.stderr)
        raise

    text = resp.choices[0].message.content.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict) and "score" in result:
            result["score"] = int(result["score"])
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to extract score from text
    for s in [5, 4, 3, 2, 1]:
        if str(s) in text:
            return {"score": s, "reason": text, "_parse_fallback": True}

    return {"score": None, "reason": text, "_parse_error": True}


def process_task(
    client: openai.OpenAI | None,
    model: str,
    task_dir: Path,
    max_helper_chars: int,
    max_per_skill_chars: int,
    max_total_chars: int,
    all_results: dict,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Judge combined skill coverage for a single task."""
    task_name = task_dir.name

    # Skip if already judged (unless --force)
    existing = all_results.get(task_name, {})
    if existing.get("score") is not None and not force:
        return existing

    instruction = (task_dir / "instruction.md").read_text(errors="replace").strip()
    n_skills = len(find_skills(task_dir))
    all_skills_content = build_all_skills_content(task_dir, max_helper_chars, max_per_skill_chars, max_total_chars)

    if dry_run:
        print(f"  [{task_name}] {n_skills} skills, {len(all_skills_content)} chars (dry run)")
        return {"task": task_name, "model": model, "n_skills": n_skills, "score": None}

    try:
        judgment = judge_task(client, model, instruction, task_name, all_skills_content)
        judgment["task"] = task_name
        judgment["model"] = model
        judgment["n_skills"] = n_skills
        print(f"  [{task_name}] score={judgment.get('score')} ({n_skills} skills)")
        return judgment
    except openai.RateLimitError:
        print(f"  [{task_name}] rate limited, waiting 30s...")
        time.sleep(30)
        judgment = judge_task(client, model, instruction, task_name, all_skills_content)
        judgment["task"] = task_name
        judgment["model"] = model
        judgment["n_skills"] = n_skills
        print(f"  [{task_name}] score={judgment.get('score')} ({n_skills} skills)")
        return judgment
    except Exception as e:
        print(f"  [{task_name}] ERROR — {e}", file=sys.stderr)
        return {"task": task_name, "model": model, "n_skills": n_skills, "score": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Judge overall skill coverage per task using an LLM (OpenAI-compatible API).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tasks_dir", type=Path, help="Directory containing task subdirectories")
    parser.add_argument("--tasks", nargs="+", help="Filter to specific task names")
    parser.add_argument("--exclude-tasks", nargs="+", default=list(DEFAULT_EXCLUDE_TASKS),
                        help=f"Tasks to exclude (default: {' '.join(sorted(DEFAULT_EXCLUDE_TASKS))})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (default: OPENAI_BASE_URL env var, or OpenAI default)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--max-helper-chars", type=int, default=MAX_HELPER_CHARS,
                        help=f"Max chars per helper file (default: {MAX_HELPER_CHARS})")
    parser.add_argument("--max-per-skill-chars", type=int, default=MAX_PER_SKILL_CHARS,
                        help=f"Max chars per individual skill (default: {MAX_PER_SKILL_CHARS})")
    parser.add_argument("--max-skill-chars", type=int, default=MAX_SKILL_TOTAL_CHARS,
                        help=f"Max total chars across all skills per task (default: {MAX_SKILL_TOTAL_CHARS})")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt sizes without calling API")
    parser.add_argument("--force", action="store_true", help="Re-judge even if results already exist")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent tasks (default: 1, sequential)")

    args = parser.parse_args()

    if not args.tasks_dir.is_dir():
        print(f"Error: {args.tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_dirs = find_task_dirs(args.tasks_dir, args.tasks, set(args.exclude_tasks))
    if not task_dirs:
        print(f"No tasks found in {args.tasks_dir}", file=sys.stderr)
        sys.exit(1)

    total_skills = sum(len(find_skills(d)) for d in task_dirs)
    print(f"Found {len(task_dirs)} tasks with {total_skills} total skills")
    print(f"Model: {args.model}")
    print()

    client = None
    if not args.dry_run:
        client_kwargs = {}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        if args.api_key:
            client_kwargs["api_key"] = args.api_key
        client = openai.OpenAI(**client_kwargs)

    output_path = args.tasks_dir / "coverage_judgments.json"

    # Load existing results to support incremental runs
    all_results = {}
    if output_path.exists() and not args.force:
        try:
            all_results = json.load(open(output_path))
        except json.JSONDecodeError:
            all_results = {}

    save_lock = threading.Lock()

    def run_task(task_dir: Path) -> None:
        result = process_task(
            client, args.model, task_dir,
            args.max_helper_chars, args.max_per_skill_chars, args.max_skill_chars,
            all_results,
            dry_run=args.dry_run, force=args.force,
        )
        with save_lock:
            all_results[task_dir.name] = result
            if not args.dry_run:
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)

    if args.concurrency <= 1:
        for task_dir in task_dirs:
            run_task(task_dir)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_task, td): td for td in task_dirs}
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    td = futures[future]
                    print(f"  [{td.name}] unexpected error: {exc}", file=sys.stderr)

    if not args.dry_run:
        print(f"Results saved → {output_path}")

    # Print summary
    scores = [
        r.get("score")
        for r in all_results.values()
        if r.get("score") is not None
    ]
    if scores:
        print("=" * 60)
        print(f"Summary: {len(scores)} tasks judged")
        for s in [5, 4, 3, 2, 1]:
            print(f"  Score {s}: {scores.count(s)} ({100*scores.count(s)/len(scores):.0f}%)")
        print(f"  Avg score: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()
