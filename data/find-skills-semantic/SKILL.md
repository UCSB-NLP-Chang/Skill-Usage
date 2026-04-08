---
name: finding-skills-semantic
description: Discovers relevant agent skills using semantic (embedding) search. Breaks complex tasks into sub-tasks and finds 10 skills via natural language similarity. Use when starting a new task, looking for specialized capabilities, or wanting to find best practices for a domain.
---

# Finding Skills

Searches a local index of agent skills using semantic search to find the most relevant ones for a given task. Skills are pre-downloaded — no installation needed.

## When to Use

- Starting a new task that may benefit from specialized skills
- Looking for best practices, patterns, or workflows for a specific domain
- Wanting to find tools or templates for a task (testing, deployment, design, etc.)

## Search API

Use `curl -s` to query the semantic search endpoint.

### Semantic search

Best for conceptual queries where you describe what you need in natural language.

```bash
curl -s "http://localhost:8742/semantic?q=QUERY&top_k=10"
```

Example: `q=help me build and deploy containerized applications`

### Response format

The search endpoint returns a JSON array. Each result contains:

- `name`: skill name (may be duplicated across authors)
- `description`: what the skill does
- `skill_md_snippet`: first 100 words of the skill's documentation
- `skill_id`: unique identifier (author--name format), use with the detail endpoint
- `github_stars`: popularity of the source repository
- `score`: relevance score, 0-1 where higher = better match.

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

For each sub-task, run semantic search queries to find skills.

```bash
# Sub-task: implement authentication
curl -s "http://localhost:8742/semantic?q=implement+authentication+with+JWT+tokens&top_k=10"

# Sub-task: write tests
curl -s "http://localhost:8742/semantic?q=writing+unit+tests+for+web+applications&top_k=10"
```

Refine queries if initial results are too broad or miss the mark. Try different natural language descriptions.

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
