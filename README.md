# How Well Do Agentic Skills Work in the Wild

Code and data for **["How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings"](https://arxiv.org/abs/2604.04323)**.

[![arXiv](https://img.shields.io/badge/arXiv-2604.04323-b31b1b.svg)](https://arxiv.org/abs/2604.04323) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage)

## Overview

This repository provides a framework for evaluating skill utility under progressively realistic conditions: from hand-curated skills to retrieval from a noisy pool of 34k real-world skills.

Key findings:
- Skill benefits **degrade consistently** as settings become more realistic
- Agents struggle to **select**, **retrieve**, and **adapt** skills
- **Query-specific refinement** substantially recovers lost performance

We evaluate on [SkillsBench](https://github.com/benchflow-ai/skillsbench) and [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) across three models: Claude Opus 4.6, Kimi K2.5, and Qwen3.5-397B.

## Setup

```bash
git clone https://github.com/UCSB-NLP-Chang/Skill-Usage.git
cd Skill-Usage

# Create environment
conda create -n skillusage python=3.12 -y
conda activate skillusage

# Install dependencies
pip install uv huggingface_hub pyyaml numpy
uv tool install --force harbor@git+https://github.com/yujianll/harbor@main
```

Requirements:
- Docker (for running tasks in isolated containers)
- API access for the model you want to evaluate

Copy `.env.example` to `.envrc` and fill in your keys. Use [direnv](https://direnv.net/) to load them automatically.

## Repository Structure

```
.
├── tasks/                      # SkillsBench base tasks (84 tasks)
├── terminal-bench-2/           # Terminal-Bench 2.0 base tasks (89 tasks)
├── skills/                     # 34k real-world skill collection (download from HF)
├── search_server/              # Skill search engine (BM25 + semantic + hybrid)
│   └── index/                  # Pre-built search index (download from HF)
├── scripts/                    # Experiment scripts
│   ├── prepare_experiment.py           # Download data & generate configs
│   ├── retrieve_skills.py              # Agentic skill retrieval
│   ├── query_specific_refinement.py    # Query-specific skill refinement
│   ├── query_agnostic_refinement.py    # Query-agnostic skill refinement
│   ├── calculate_results.py            # Aggregate results
│   ├── check_skill_usage.py            # Analyze skill usage in trajectories
│   └── ...
├── experiments/configs/        # Harbor run configurations
└── data/                       # Retrieval data and mappings
```

---

## Option 1: Reproduce Paper Results

Use `scripts/prepare_experiment.py` to download pre-computed experiment data from our [HuggingFace dataset](https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage), build task folders, and generate a Harbor config — all in one command.

```bash
python scripts/prepare_experiment.py \
    --benchmark skillsbench \
    --setting curated \
    --model claude
```

The script prints the Harbor command to run at the end, e.g.:

```bash
harbor run -c experiments/configs/generated/skillsbench/curated-claude.yaml
```

### Available Settings

**SkillsBench** (from most idealized to most realistic):

| Setting | Description |
|---------|-------------|
| `force_loaded` | All curated skills force-loaded into context (upper bound) |
| `curated` | Original curated skills, agent decides which to load |
| `curated_w_distractors` | Curated skills + distractor skills from 34k pool |
| `no_skills` | No skills provided (baseline) |
| `retrieved_w_curated` | Agent retrieves top-5 from 34k pool (curated included) |
| `retrieved_wo_curated` | Agent retrieves top-5 from 34k pool (curated removed) |
| `retrieved_w_curated_query_specific` | Retrieved + query-specific refinement |
| `retrieved_w_curated_query_agnostic` | Retrieved + query-agnostic refinement |
| `retrieved_wo_curated_query_specific` | Retrieved (w/o curated) + query-specific refinement |
| `retrieved_wo_curated_query_agnostic` | Retrieved (w/o curated) + query-agnostic refinement |

**Terminal-Bench 2.0:**

| Setting | Description |
|---------|-------------|
| `base` | No skills (baseline) |
| `retrieved` | Agent retrieves top-5 from 34k pool |
| `retrieved_query_specific` | Retrieved + query-specific refinement |
| `retrieved_query_agnostic` | Retrieved + query-agnostic refinement |

**Models:** `claude` (Claude Code), `kimi` (Terminus-2), `qwen` (Qwen-Code)

### Additional Options

```bash
python scripts/prepare_experiment.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | auto | Output directory for prepared tasks |
| `--config-dir` | `experiments/configs/generated` | Where to write the config YAML |
| `--concurrency` | 8 | Number of concurrent trials |
| `--port` | 30000 | Port for local model API server (kimi/qwen) |

### Calculate Results

After a Harbor run completes:

```bash
# Single job
python scripts/calculate_results.py skillsbench-trajectories/jobs/<job_name>

# All jobs in a folder
python scripts/calculate_results.py skillsbench-trajectories/jobs --batch
```

---

## Option 2: Full Pipeline Walkthrough

Run the complete pipeline yourself — retrieval, task completion, refinement, and evaluation — instead of using pre-computed data.

### Step 1: Download Skills and Start the Search Server

The search server provides keyword, semantic, and hybrid search over the 34k skill collection.

**Download the skill collection and pre-built search index:**

```bash
# Download skills and metadata (needed for /detail endpoint and building index from scratch)
hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip skills-34k/skills_meta.jsonl --repo-type dataset --local-dir .
unzip skills-34k/skills.zip -d skills/
cp skills-34k/skills_meta.jsonl skills/

# Download pre-built search index
hf download Shiyu-Lab/Skill-Usage search_index/search_index.zip --repo-type dataset --local-dir .
unzip search_index/search_index.zip -d search_server/index/
```

Using the pre-built index is preferred. Alternatively, you can build the index from scratch (requires a running [vLLM](https://github.com/vllm-project/vllm) instance serving `Qwen/Qwen3-Embedding-4B` and the full skill collection):

```bash
python search_server/index_builder.py
```

**Install search server dependencies and start:**

```bash
pip install -r search_server/requirements.txt
python search_server/http_server.py --include-content
```

**Server arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8742 | Port to serve on |
| `--model` | `Qwen/Qwen3-Embedding-4B` | Embedding model for semantic search (requires vLLM) |
| `--exclude-gt` | false | Use indexes built without ground-truth skills |
| `--include-content` | false | Include full SKILL.md content in semantic search |
| `--content-weight` | 0.05 | Blend weight for content vs metadata embeddings |

**Endpoints:**
- `GET /keyword?q=...&top_k=10` — BM25 keyword search
- `GET /semantic?q=...&top_k=10` — Dense embedding search
- `GET /hybrid?q=...&top_k=10` — Reciprocal rank fusion of keyword + semantic
- `GET /detail/{skill_id}` — Full SKILL.md content for a given skill

### Step 2: Retrieve Skills for Tasks

First, copy the base tasks into a working directory:

```bash
# For SkillsBench
cp -r tasks skillsbench-retrieved-claude

# For Terminal-Bench 2.0
cp -r terminal-bench-2 terminalbench-retrieved-claude
```

Use agentic search where an agent iteratively formulates queries, inspects candidates, and selects the most relevant skills. The `--tasks-dir` argument points to the working task directory created above.

```bash
# Prepare Docker containers for skill retrieval
python scripts/retrieve_skills.py prepare \
    --tasks-dir skillsbench-retrieved-claude \
    --agent claude-code \
    --model claude-opus-4-6 \
    --max-workers 5

# For Kimi:  --agent terminus-2 --model openai/moonshotai/Kimi-K2.5
# For Qwen:  --agent qwen-coder --model Qwen/Qwen3.5-397B-A17B-FP8

# Run retrieval (config path is printed by the prepare step)
harbor run -c <config_path>

# Collect results — each task gets a found_skills.json with skill IDs sorted by relevance
python scripts/retrieve_skills.py collect \
    --tasks-dir skillsbench-retrieved-claude \
    --keep-temp
```

Then, copy the actual skill files into each task's `environment/skills/` using `select_top_k_skills.py`. This reads each task's `found_skills.json`, copies the top-k skills from the `skills/` directory, and produces a new task directory ready to run:

```bash
# Select top-5 skills per task (creates skillsbench-retrieved-claude-5/)
python scripts/select_top_k_skills.py 5 --tasks-dir skillsbench-retrieved-claude
```

For **Terminal-Bench 2.0** tasks, you also need to rewrite the Dockerfiles to include skill COPY lines. Terminal-Bench tasks use prebuilt Docker images specified in `task.toml`, which bypasses the Dockerfile. This script removes the prebuilt image from `task.toml` and rewrites the Dockerfile to use it as the base with skills added on top:

```bash
python scripts/rewrite_dockerfile.py terminalbench-retrieved-claude-5
```

### Step 3: Run Tasks

The output directory (e.g., `skillsbench-retrieved-claude-5/`) now contains tasks with skills in `environment/skills/`. To run the benchmark, use one of the example configs in `experiments/configs/` and update the `datasets.path` field to point to your task directory:

```bash
# Edit the config to set your task directory path
# Then run:
harbor run -c <your_config>.yaml
```

Base configs are at `experiments/configs/{skillsbench,terminalbench}/{claude,kimi,qwen}.yaml`.

### Step 4: Refine Skills (Optional)

> **Note:** Both refinement scripts use `.local-workspace/` for temp files and jobs by default. To run query-specific and query-agnostic refinement concurrently, use `--workspace` to separate them, e.g., `--workspace .workspace-specific` and `--workspace .workspace-agnostic`.

#### Query-Specific Refinement

The agent explores the target task, attempts a solution using retrieved skills, reflects on what helped or hurt, and composes improved skills. The `--tasks-dir` should point to a task directory that already has skills (e.g., from Step 2).

```bash
# Prepare (uses the task's own Dockerfile so the agent can test skills)
python scripts/query_specific_refinement.py prepare \
    --tasks-dir skillsbench-retrieved-claude-5 \
    --agent claude-code \
    --model claude-opus-4-6
# For Kimi:  --agent terminus-2 --model openai/moonshotai/Kimi-K2.5
# For Qwen:  --agent qwen-coder --model Qwen/Qwen3.5-397B-A17B-FP8

# Run refinement
harbor run -c <config_path>

# Copy task directory before collecting (collect overwrites environment/skills/)
cp -r skillsbench-retrieved-claude-5 skillsbench-retrieved-claude-5-query-specific

# Collect refined skills into the copy
python scripts/query_specific_refinement.py collect \
    --tasks-dir skillsbench-retrieved-claude-5-query-specific \
    --keep-temp
```

#### Query-Agnostic Refinement

Each skill is improved independently using the `skill-creator` methodology, without knowledge of the target task:

```bash
# Prepare per-skill refinement containers
python scripts/query_agnostic_refinement.py prepare \
    --tasks-dir skillsbench-retrieved-claude-5 \
    --agent claude-code \
    --model claude-opus-4-6
# For Kimi:  --agent terminus-2 --model openai/moonshotai/Kimi-K2.5
# For Qwen:  --agent qwen-coder --model Qwen/Qwen3.5-397B-A17B-FP8

# Run refinement
harbor run -c <config_path>

# Copy task directory before collecting (collect overwrites environment/skills/)
cp -r skillsbench-retrieved-claude-5 skillsbench-retrieved-claude-5-query-agnostic

# Collect and distribute refined skills into the copy
python scripts/query_agnostic_refinement.py collect \
    --tasks-dir skillsbench-retrieved-claude-5-query-agnostic \
    --keep-temp
```

### Step 5: Evaluate

```bash
# Aggregate pass rates (mean +/- std across runs)
python scripts/calculate_results.py skillsbench-trajectories/jobs/<job_name>

# Analyze skill usage in agent trajectories
python scripts/check_skill_usage.py skillsbench-trajectories/jobs/<job_name>

# Judge skill coverage with an LLM (requires OpenAI-compatible API)
python scripts/judge_skill_coverage.py <tasks-dir> --model gpt-5.4
```

---

## Skill Collection

The skill collection contains 34,198 real-world skills sourced from [skillhub.club](https://www.skillhub.club/) and [skills.sh](https://skills.sh/), filtered by permissive licenses (MIT and Apache 2.0), deduplicated, and cleaned. Each skill is a folder containing a `SKILL.md` file with structured metadata and content.

To download:

```bash
hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip --repo-type dataset --local-dir .
unzip skills-34k/skills.zip -d skills/
```

## Citation

```bibtex
@misc{liu2026agenticskillsworkwild,
    title={How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings}, 
    author={Yujian Liu and Jiabao Ji and Li An and Tommi Jaakkola and Yang Zhang and Shiyu Chang},
    year={2026},
    eprint={2604.04323},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2604.04323}, 
}
```

## Acknowledgments

This work builds on and is grateful to the following projects:

- [SkillsBench](https://github.com/benchflow-ai/skillsbench) — the benchmark for evaluating how well AI agents use skills
- [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) — a benchmark for hard, realistic tasks in command line interfaces
- [Harbor](https://github.com/laude-institute/harbor) — the framework for evaluating and optimizing agents in container environments

## License

MIT
