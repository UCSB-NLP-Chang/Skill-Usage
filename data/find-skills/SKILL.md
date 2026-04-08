---
name: finding-skills
description: Discovers relevant agent skills from a local index of skills for a given task. Breaks complex tasks into sub-tasks and finds 10 skills across keyword, semantic, and hybrid search. Use when starting a new task, looking for specialized capabilities, or wanting to find best practices for a domain.
---

# Finding Skills

Searches a local index of agent skills to find the most relevant ones for a given task. Skills are pre-downloaded — no installation needed.

## When to Use

- Starting a new task that may benefit from specialized skills
- Looking for best practices, patterns, or workflows for a specific domain
- Wanting to find tools or templates for a task (testing, deployment, design, etc.)

## Search API

Three search endpoints are available. Use `curl -s` to query them.

### Keyword search

Best for exact term matching when you know specific skill names or technologies.

```bash
curl -s "http://localhost:8742/keyword?q=QUERY&top_k=10"
```

Supports FTS5 syntax:
- Prefix: `react*`
- Phrase: `"code review"`
- Boolean: `react OR vue`

### Semantic search

Best for conceptual queries where you describe what you need in natural language.

```bash
curl -s "http://localhost:8742/semantic?q=QUERY&top_k=10"
```

Example: `q=help me build and deploy containerized applications`

### Hybrid search

Combines keyword and semantic search with reciprocal rank fusion. Best general-purpose option.

```bash
curl -s "http://localhost:8742/hybrid?q=QUERY&top_k=10&keyword_weight=0.5&semantic_weight=0.5"
```

- `keyword_weight`: How much BM25 keyword matches contribute to the final ranking (default: 0.5)
- `semantic_weight`: How much semantic similarity contributes to the final ranking (default: 0.5)

### Response format

All search endpoints return a JSON array. Each result contains:

- `name`: skill name (may be duplicated across authors)
- `description`: what the skill does
- `skill_md_snippet`: first 100 words of the skill's documentation
- `skill_id`: unique identifier (author--name format), use with the detail endpoint
- `github_stars`: popularity of the source repository
- `score`: relevance score. For keyword search, more negative = better match. For semantic search, 0-1 where higher = better match. Hybrid search returns `rrf_score` instead (higher = better match).

### Skill detail

Fetches full metadata and complete SKILL.md content for a skill. Pass the `skill_id` from search results.

```bash
curl -s "http://localhost:8742/detail/SKILL_ID"
```

## Workflow

### Step 1: Analyze the task

Break the task into concrete sub-tasks. For example, "build a REST API with auth and tests" becomes:
- Design API endpoints and routing
- Implement authentication
- Write tests

### Step 2: Search for each sub-task

For each sub-task, run search queries to find skills.

```bash
# Sub-task: implement authentication
curl -s "http://localhost:8742/hybrid?q=implement+authentication+JWT&top_k=10"
# Keyword search
curl -s "http://localhost:8742/keyword?q=JWT&top_k=10"

# Sub-task: write tests
curl -s "http://localhost:8742/hybrid?q=writing+unit+tests&top_k=10"
```

Refine queries if initial results are too broad or miss the mark. Try different phrasing or switch between keyword and semantic search.

### Step 3: Review and select

From the search results, select 10 skills total across all sub-tasks. Prioritize:
- High relevance to the target task
- Higher `github_stars` when multiple skills cover the same topic
- Skills with informative `skill_md_snippet` content

If needed, fetch full details of a skill to confirm relevance:

```bash
curl -s "http://localhost:8742/detail/SKILL_ID"
```

### Step 4: Record results

Record the selected skills as a structured list:

```
## Found Skills

- **[skill-name]** (skill_id: [skill_id]) - [one-line summary from description]
- **[skill-name]** (skill_id: [skill_id]) - [one-line summary from description]
- ...
```
