#!/usr/bin/env python3
"""Download data from HuggingFace and prepare task folders + config for a given experiment setting.

Usage:
    python scripts/prepare_experiment.py --benchmark skillsbench --setting curated --model claude
    python scripts/prepare_experiment.py --benchmark terminalbench --setting retrieved --model qwen
    python scripts/prepare_experiment.py --benchmark skillsbench --setting retrieved_w_curated_query_specific --model claude

Settings for skillsbench:
    curated, curated_w_distractors, force_loaded, no_skills,
    retrieved_w_curated, retrieved_wo_curated,
    retrieved_w_curated_query_agnostic, retrieved_w_curated_query_specific,
    retrieved_wo_curated_query_agnostic, retrieved_wo_curated_query_specific

Settings for terminalbench:
    base, retrieved, retrieved_query_agnostic, retrieved_query_specific

Models: claude, kimi, qwen
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO = "Shiyu-Lab/Skill-Usage"

# Settings that are per-task (no model subfolder in HF)
SKILLSBENCH_FLAT_SETTINGS = {
    "curated",
    "curated_terminus",
    "curated_w_distractors",
    "curated_w_distractors_terminus",
    "force_loaded",
    "force_loaded_terminus",
    "no_skills",
}

# For kimi on skillsbench flat settings, use the _terminus variant
KIMI_TERMINUS_SUFFIX_SETTINGS = {
    "curated",
    "curated_w_distractors",
    "force_loaded",
}

SKILLSBENCH_SETTINGS = [
    "curated",
    "curated_w_distractors",
    "force_loaded",
    "no_skills",
    "retrieved_w_curated",
    "retrieved_wo_curated",
    "retrieved_w_curated_query_agnostic",
    "retrieved_w_curated_query_specific",
    "retrieved_wo_curated_query_agnostic",
    "retrieved_wo_curated_query_specific",
]

TERMINALBENCH_SETTINGS = [
    "base",
    "retrieved",
    "retrieved_query_agnostic",
    "retrieved_query_specific",
]

ALL_SETTINGS = SKILLSBENCH_SETTINGS + TERMINALBENCH_SETTINGS

# Base task directories
SKILLSBENCH_BASE = "tasks"
TERMINALBENCH_BASE = "terminal-bench-2"

# Excluded tasks for skillsbench
SKILLSBENCH_EXCLUDE = [
    "mhc-layer-impl",
    "scheduling-email-assistant",
    "fix-visual-stability",
]

# Model -> agent config mapping
MODEL_CONFIGS = {
    "claude": {
        "agent_name": "claude-code",
        "model_name": "claude-opus-4-6",
        "kwargs": {"version": "2.1.19"},
    },
    "kimi": {
        "agent_name": "terminus-2",
        "model_name": "openai/moonshotai/Kimi-K2.5",
        "extra_kwargs": {"model_info": {"max_input_tokens": 253952}},
    },
    "qwen": {
        "agent_name": "qwen-coder",
        "model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
        "kwargs": {"version": "0.12.3", "api_key": "empty"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HF data, build task folders, and generate harbor config."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["skillsbench", "terminalbench"],
        help="Which benchmark to prepare.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=ALL_SETTINGS,
        help="Evaluation setting. "
        "skillsbench: {%(sb)s}. "
        "terminalbench: {%(tb)s}."
        % {
            "sb": ", ".join(SKILLSBENCH_SETTINGS),
            "tb": ", ".join(TERMINALBENCH_SETTINGS),
        },
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["claude", "kimi", "qwen"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prepared tasks. Default: auto-generated name.",
    )
    parser.add_argument(
        "--config-dir",
        default="experiments/configs/generated",
        help="Directory to write the harbor config YAML.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent trials (default: 8).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port for local model API server (default: 30000). Used by kimi and qwen.",
    )
    return parser.parse_args()


def resolve_hf_path(benchmark: str, setting: str, model: str) -> str:
    """Return the HF subfolder path containing per-task folders to download.

    For flat settings (no model subfolder), returns e.g. 'skillsbench/curated'.
    For per-model settings, returns e.g. 'skillsbench/retrieved_w_curated/qwen'.
    """
    if benchmark == "skillsbench":
        if model == "kimi" and setting in KIMI_TERMINUS_SUFFIX_SETTINGS:
            hf_setting = f"{setting}_terminus"
        else:
            hf_setting = setting
        is_flat = hf_setting in SKILLSBENCH_FLAT_SETTINGS
        base = f"skillsbench/{hf_setting}"
        return base if is_flat else f"{base}/{model}"
    else:  # terminalbench
        base = f"terminalbench/{setting}"
        return base if setting == "base" else f"{base}/{model}"


def download_from_hf(hf_subfolder: str, download_dir: Path) -> None:
    """Download a subfolder from the HF dataset repo using the Python API."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {HF_REPO}/{hf_subfolder} -> {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=f"{hf_subfolder}/*",
        local_dir=str(download_dir),
    )
    print("  Download complete.")



def copy_base_tasks(
    base_dir: Path, output_dir: Path, exclude: list[str] | None = None
) -> list[str]:
    """Copy all tasks from the base directory to output, returning task names."""
    exclude = exclude or []
    output_dir.mkdir(parents=True, exist_ok=True)
    task_names = []
    for task_path in sorted(base_dir.iterdir()):
        if not task_path.is_dir():
            continue
        if task_path.name in exclude:
            continue
        dest = output_dir / task_path.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(task_path, dest)
        task_names.append(task_path.name)
    print(f"Copied {len(task_names)} base tasks from {base_dir} -> {output_dir}")
    return task_names


def overlay_downloaded_files(downloaded_tasks_dir: Path, output_dir: Path) -> int:
    """Overlay downloaded files onto base tasks. Returns the number of tasks overlaid."""
    count = 0
    if not downloaded_tasks_dir.exists():
        raise FileNotFoundError(
            f"Downloaded tasks dir does not exist: {downloaded_tasks_dir}"
        )

    for task_dir in sorted(downloaded_tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        dest_task = output_dir / task_dir.name
        if not dest_task.exists():
            print(f"  Skipping {task_dir.name}: not in output tasks")
            continue

        # Overlay instruction.md
        src_instruction = task_dir / "instruction.md"
        if src_instruction.exists():
            shutil.copy2(src_instruction, dest_task / "instruction.md")

        # Overlay task.toml
        src_toml = task_dir / "task.toml"
        if src_toml.exists():
            shutil.copy2(src_toml, dest_task / "task.toml")

        # Overlay environment/Dockerfile
        src_env = task_dir / "environment"
        if src_env.exists():
            dest_env = dest_task / "environment"
            src_dockerfile = src_env / "Dockerfile"
            if src_dockerfile.exists():
                shutil.copy2(src_dockerfile, dest_env / "Dockerfile")

        # Overlay skills folder
        src_skills = task_dir / "skills"
        if src_skills.exists():
            dest_skills = dest_task / "environment" / "skills"
            if dest_skills.exists():
                shutil.rmtree(dest_skills)
            shutil.copytree(src_skills, dest_skills)

        count += 1
    print(f"Overlaid files for {count} tasks from {downloaded_tasks_dir}")
    return count


def generate_config(
    benchmark: str,
    setting: str,
    model: str,
    output_dir: Path,
    config_dir: Path,
    concurrency: int,
    port: int,
) -> Path:
    """Generate a harbor run YAML config file."""
    mc = MODEL_CONFIGS[model]

    # Timeout / retry rules
    if benchmark == "skillsbench":
        timeout_multiplier = 1.5
        max_retries = 3
        jobs_dir = "skillsbench-trajectories/jobs"
    else:
        timeout_multiplier = 1.0 if model == "claude" else 2.0
        max_retries = 0
        jobs_dir = "terminalbench-trajectories/jobs"

    job_name = f"{setting}-{model}"

    # Build agent kwargs
    agent_kwargs: dict = dict(mc.get("kwargs", {}))
    if model == "kimi":
        agent_kwargs["api_base"] = f"http://localhost:{port}/v1"
        agent_kwargs.update(mc.get("extra_kwargs", {}))
    elif model == "qwen":
        agent_kwargs["base_url"] = f"http://host.docker.internal:{port}/v1"

    # Build config dict
    config = {
        "job_name": job_name,
        "jobs_dir": jobs_dir,
        "n_attempts": 3,
        "timeout_multiplier": timeout_multiplier,
        "debug": False,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": concurrency,
            "quiet": False,
            "retry": {
                "max_retries": max_retries,
                "include_exceptions": None,
                "exclude_exceptions": [
                    "VerifierTimeoutError",
                    "BadRequestError",
                    "RateLimitError",
                    "AgentTimeoutError",
                ],
                "wait_multiplier": 1.0,
                "min_wait_sec": 1.0,
                "max_wait_sec": 60.0,
            },
            "kwargs": {},
        },
        "environment": {
            "type": "docker",
            "import_path": None,
            "force_build": False,
            "delete": True,
            "override_cpus": None,
            "override_memory_mb": None,
            "override_storage_mb": None,
            "override_gpus": None,
            "kwargs": {},
        },
        "verifier": {
            "override_timeout_sec": None,
            "max_timeout_sec": None,
            "disable": False,
        },
        "metrics": [],
        "agents": [
            {
                "name": mc["agent_name"],
                "import_path": None,
                "model_name": mc["model_name"],
                "override_timeout_sec": None,
                "override_setup_timeout_sec": None,
                "max_timeout_sec": None,
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [
            {
                "task_names": None,
                **(
                    {"exclude_task_names": SKILLSBENCH_EXCLUDE}
                    if benchmark == "skillsbench"
                    else {}
                ),
                "path": str(output_dir),
            }
        ],
    }

    config_dir = Path(config_dir) / benchmark
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to {config_path}")
    return config_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    benchmark: str = args.benchmark
    setting: str = args.setting
    model: str = args.model

    # Validate setting matches benchmark
    valid = SKILLSBENCH_SETTINGS if benchmark == "skillsbench" else TERMINALBENCH_SETTINGS
    if setting not in valid:
        sys.exit(
            f"Error: setting '{setting}' is not valid for {benchmark}.\n"
            f"Valid settings: {', '.join(valid)}"
        )

    # Output directory for prepared tasks
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / f"{benchmark}-{setting}-{model}"

    # Base tasks directory
    if benchmark == "skillsbench":
        base_dir = repo_root / SKILLSBENCH_BASE
        exclude = SKILLSBENCH_EXCLUDE
    else:
        base_dir = repo_root / TERMINALBENCH_BASE
        exclude = []

    # For terminalbench/base and skillsbench/curated (non-kimi), the base
    # tasks already have everything — just copy, no download needed.
    skip_overlay = (benchmark == "terminalbench" and setting == "base") or (
        benchmark == "skillsbench" and setting == "curated" and model != "kimi"
    )

    if not skip_overlay:
        # Step 1: Download from HF (huggingface_hub handles caching automatically)
        hf_path = resolve_hf_path(benchmark, setting, model)
        download_dir = repo_root / ".hf_download"
        download_from_hf(hf_path, download_dir)

    # Step 2: Copy base tasks
    print(f"\nCopying base tasks from {base_dir} -> {output_dir}")
    copy_base_tasks(base_dir, output_dir, exclude=exclude)

    if not skip_overlay:
        # Step 3: Overlay downloaded files
        downloaded_tasks_dir = download_dir / hf_path
        print(f"\nOverlaying downloaded files from {downloaded_tasks_dir}")
        overlay_downloaded_files(downloaded_tasks_dir, output_dir)
    else:
        print(f"\nNo overlay needed for {benchmark}/{setting} — base tasks are sufficient.")

    # Step 4: Generate config
    print("\nGenerating harbor config...")
    config_path = generate_config(
        benchmark=benchmark,
        setting=setting,
        model=model,
        output_dir=output_dir,
        config_dir=repo_root / args.config_dir,
        concurrency=args.concurrency,
        port=args.port,
    )

    print(f"\n{'='*60}")
    print("Preparation complete!")
    print(f"  Task directory: {output_dir}")
    print(f"  Config file:    {config_path}")
    print(f"\nTo run:")
    print(f"  harbor run -c {config_path}")


if __name__ == "__main__":
    main()
