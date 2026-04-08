#!/usr/bin/env python3
"""Rewrite Dockerfiles to use prebuilt images as base and only add skills on top.

For each task that has a docker_image in task.toml:
1. Rewrites the Dockerfile to: FROM <prebuilt-image> + skills COPY lines
2. Removes the docker_image line from task.toml

This lets Docker pull the prebuilt image as a cached base layer and only add
a thin skills layer on top — fast build, identical environment.

Usage:
    python scripts/use_prebuilt_with_skills.py terminal-bench-2-claude-skills-5/
    python scripts/use_prebuilt_with_skills.py terminal-bench-2-claude-skills-5/ --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKILLS_BLOCK = """\
# Copy skills to agent-specific locations
COPY skills /root/.claude/skills
COPY skills /root/.codex/skills
COPY skills /root/.opencode/skill
COPY skills /root/.goose/skills
COPY skills /root/.factory/skills
COPY skills /root/.agents/skills
COPY skills /root/.gemini/skills
COPY skills /root/.qwen/skills
"""


def extract_docker_image(task_toml: Path) -> str | None:
    """Extract docker_image value from task.toml."""
    text = task_toml.read_text()
    match = re.search(r'^docker_image\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def remove_docker_image(task_toml: Path, dry_run: bool) -> None:
    """Remove the docker_image line from task.toml."""
    text = task_toml.read_text()
    new_text = re.sub(r'^docker_image\s*=\s*"[^"]+"\n?', "", text, flags=re.MULTILINE)
    if not dry_run:
        task_toml.write_text(new_text)


def extract_canary_comment(dockerfile: Path) -> str:
    """Extract canary/comment lines before FROM."""
    lines = dockerfile.read_text().splitlines()
    comments = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            comments.append(line)
        else:
            break
    return "\n".join(comments) + "\n" if comments else ""


def rewrite_dockerfile(dockerfile: Path, prebuilt_image: str, dry_run: bool) -> None:
    """Rewrite Dockerfile to use prebuilt image + skills COPY."""
    canary = extract_canary_comment(dockerfile)
    new_content = f"{canary}FROM {prebuilt_image}\n\n{SKILLS_BLOCK}"
    if not dry_run:
        dockerfile.write_text(new_content)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite Dockerfiles to use prebuilt images with skills"
    )
    parser.add_argument("tasks_dir", help="Directory containing task subdirectories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without modifying files",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    modified = 0
    skipped = 0

    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_toml = task_dir / "task.toml"
        dockerfile = task_dir / "environment" / "Dockerfile"

        if not task_toml.exists() or not dockerfile.exists():
            continue

        prebuilt_image = extract_docker_image(task_toml)
        if not prebuilt_image:
            skipped += 1
            continue

        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"  {prefix}{task_dir.name}: FROM {prebuilt_image}")

        rewrite_dockerfile(dockerfile, prebuilt_image, args.dry_run)
        remove_docker_image(task_toml, args.dry_run)
        modified += 1

    print(f"\nModified: {modified}, Skipped (no docker_image): {skipped}")


if __name__ == "__main__":
    main()