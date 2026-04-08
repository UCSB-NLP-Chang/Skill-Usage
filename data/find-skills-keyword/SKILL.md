---
name: finding-skills-keyword
description: Discovers relevant agent skills using keyword (BM25) search. Breaks complex tasks into sub-tasks and finds 10 skills via term matching. Use when starting a new task, looking for specialized capabilities, or wanting to find best practices for a domain.
---

# Finding Skills

Searches a local index of agent skills using keyword search to find the most relevant ones for a given task. Skills are pre-downloaded — no installation needed.

## When to Use

- Starting a new task that may benefit from specialized skills
- Looking for best practices, patterns, or workflows for a specific domain
- Wanting to find tools or templates for a task (testing, deployment, design, etc.)

## Search API

Use `curl -s` to query the keyword search endpoint.

### Keyword search

Best for exact term matching when you know specific skill names or technologies.

```bash
curl -s "http://localhost:8742/keyword?q=QUERY&top_k=10"
```

Supports FTS5 syntax:
- Prefix: `react*`
- Phrase: `"code review"`
- Boolean: `react OR vue`

### Response format

The search endpoint returns a JSON array. Each result contains:

- `name`: skill name (may be duplicated across authors)
- `description`: what the skill does
- `skill_md_snippet`: first 100 words of the skill's documentation
- `skill_id`: unique identifier (author--name format), use with the detail endpoint
- `github_stars`: popularity of the source repository
- `score`: relevance score. More negative = better match.

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
curl -s "http://localhost:8742/keyword?q=JWT+authentication&top_k=10"

# Sub-task: write tests
curl -s "http://localhost:8742/keyword?q=unit+tests&top_k=10"
```

Refine queries if initial results are too broad or miss the mark. Try different phrasing, prefixes, or boolean operators.

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
