#!/usr/bin/env python3
"""Copy tasks-found-skills/ to tasks-found-skills-k/ keeping only the top-k skills per task.

Usage:
    python scripts/select_top_k_skills.py 5
    python scripts/select_top_k_skills.py 10 --tasks-dir tasks-found-skills --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy tasks keeping only top-k skills per task"
    )
    parser.add_argument("k", type=int, help="Number of top skills to keep per task")
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="Source tasks directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / args.tasks_dir
    dst_dir = repo_root / f"{args.tasks_dir}-{args.k}"

    if not src_dir.is_dir():
        print(f"Error: {src_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if dst_dir.exists():
        if args.dry_run:
            print(f"[dry-run] Would remove existing {dst_dir}")
        else:
            print(f"Removing existing {dst_dir}")
            shutil.rmtree(dst_dir)

    # Step 1: Copy entire source to destination
    print(f"Copying {src_dir} -> {dst_dir}")
    if not args.dry_run:
        shutil.copytree(src_dir, dst_dir)

    # Step 2 & 3: Process each task in the destination
    task_dirs = sorted(d for d in (dst_dir if not args.dry_run else src_dir).iterdir() if d.is_dir())
    ok = 0
    skipped = 0

    for task_dir in task_dirs:
        # When dry-run, we look at src but describe changes to dst
        display_name = task_dir.name
        actual_dir = dst_dir / display_name if args.dry_run else task_dir

        # Remove artifacts
        for artifact in ("found_skills.json", "find_skills.log", "skills-lock.json"):
            artifact_path = actual_dir / artifact if not args.dry_run else task_dir / artifact
            if (task_dir / artifact).exists():
                if args.dry_run:
                    print(f"  [dry-run] {display_name}: would remove {artifact}")
                else:
                    (actual_dir / artifact).unlink()

        # Read found_skills.json from the source to determine ordering
        found_skills_file = (src_dir / display_name / "found_skills.json") if not args.dry_run else (task_dir / "found_skills.json")
        skills_dir = actual_dir / "environment" / "skills" if not args.dry_run else task_dir / "environment" / "skills"

        if not found_skills_file.exists():
            print(f"  {display_name}: no found_skills.json, skipping skill filtering")
            skipped += 1
            continue

        try:
            skills_list = json.loads(found_skills_file.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  {display_name}: bad found_skills.json ({e}), skipping")
            skipped += 1
            continue

        if not isinstance(skills_list, list):
            print(f"  {display_name}: found_skills.json is not a list, skipping")
            skipped += 1
            continue

        # Determine if entries are strings (skill folder names) or dicts
        is_string_list = all(isinstance(entry, str) for entry in skills_list)

        # Top-k skill names (ordered by relevance from found_skills.json)
        top_k_names = []
        for entry in skills_list[:args.k]:
            if is_string_list:
                top_k_names.append(entry)
            else:
                name = entry.get("name") if isinstance(entry, dict) else None
                if name:
                    top_k_names.append(name)
        top_k_set = set(top_k_names)

        # In the destination's environment/skills/, handle skill setup
        dst_skills_dir = dst_dir / display_name / "environment" / "skills"
        skills_source = repo_root / "skills"

        if is_string_list:
            # Copy skills from external source
            if args.dry_run:
                print(f"  [dry-run] {display_name}: would copy {len(top_k_names)} skills from {skills_source}")
                for s in top_k_names:
                    print(f"           copy: {s}")
            else:
                dst_skills_dir.mkdir(parents=True, exist_ok=True)
                copied = 0
                for skill_name in top_k_names:
                    src_skill = skills_source / skill_name
                    dst_skill = dst_skills_dir / skill_name
                    if src_skill.is_dir():
                        shutil.copytree(src_skill, dst_skill, dirs_exist_ok=True)
                        metadata_file = dst_skill / "_metadata.json"
                        if metadata_file.exists():
                            metadata_file.unlink()
                        copied += 1
                    else:
                        print(f"  {display_name}: skill {skill_name} not found at {src_skill}!!!!!!!!!!")
                print(f"  {display_name}: copied {copied}/{len(top_k_names)} skills")
        elif args.dry_run:
            # Check source to report what would happen
            if skills_dir.is_dir():
                all_skills = [d.name for d in skills_dir.iterdir() if d.is_dir()]
                keep = [s for s in all_skills if s in top_k_set]
                remove = [s for s in all_skills if s not in top_k_set]
                print(f"  [dry-run] {display_name}: keep {len(keep)}/{len(all_skills)} skills, remove {len(remove)}")
                if remove:
                    print(f"           remove: {', '.join(sorted(remove))}")
        else:
            if dst_skills_dir.is_dir():
                all_skills = [d for d in dst_skills_dir.iterdir() if d.is_dir()]
                removed = 0
                for skill_dir in all_skills:
                    if skill_dir.name not in top_k_set:
                        shutil.rmtree(skill_dir)
                        removed += 1
                kept = len(all_skills) - removed
                print(f"  {display_name}: kept {kept}/{len(all_skills)} skills, removed {removed}")
            else:
                print(f"  {display_name}: no environment/skills/ directory")
                skipped += 1
                continue

        ok += 1

    print(f"\nDone: {ok} tasks processed, {skipped} skipped (k={args.k})")
    if not args.dry_run:
        print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
