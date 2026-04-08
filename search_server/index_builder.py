#!/usr/bin/env python3
"""Build FTS5 and embedding indexes from skills_listings.jsonl.

Usage:
    python3 index_builder.py                    # Build all indexes
    python3 index_builder.py --mode fts5        # Build only FTS5 index
    python3 index_builder.py --mode embeddings  # Build only embeddings
    python3 index_builder.py --limit 100        # Limit number of skills
"""

import argparse
import json
import os
import re
import sqlite3
import sys

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(SCRIPT_DIR, "..", "skills")
JSONL_PATH = os.path.join(SKILLS_DIR, "skills_meta.jsonl")
INDEX_DIR = os.path.join(SCRIPT_DIR, "index")

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

ALLOWED_LICENSES = {"MIT", "Apache-2.0", "Apache 2.0"}


GT_PREFIX = "benchflow-ai--"


def load_records(limit: int | None = None, exclude_gt: bool = False) -> list[dict]:
    records = []
    with open(JSONL_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("github_license") not in ALLOWED_LICENSES:
                continue
            if exclude_gt and rec.get("skill_id", "").startswith(GT_PREFIX):
                continue
            records.append(rec)
            if limit and len(records) >= limit:
                break
    return records


def _read_skill_md(owner: str, name: str) -> str:
    """Read SKILL.md and strip YAML front matter. Returns full text or empty string."""
    skill_dir = os.path.join(SKILLS_DIR, f"{owner}--{name}")
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.exists(skill_md):
        return ""
    try:
        with open(skill_md, errors="replace") as f:
            content = f.read()
    except Exception:
        return ""

    # Strip YAML front matter
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)
    return content.strip()


def extract_skill_md_snippet(owner: str, name: str, max_words: int = 100) -> str:
    """Read SKILL.md, strip YAML front matter, return first max_words words."""
    content = _read_skill_md(owner, name)
    words = content.split()
    return " ".join(words[:max_words])


def extract_skill_md_content(owner: str, name: str) -> str:
    """Read SKILL.md, strip YAML front matter, return full text."""
    return _read_skill_md(owner, name)


def _index_suffix(exclude_gt: bool) -> str:
    suffix = ""
    if exclude_gt:
        suffix += "_no_gt"
    return suffix


def build_fts_index(records: list[dict], exclude_gt: bool = False,
                    include_content: bool = False):
    """Create SQLite DB with skills table and FTS5 index on name+description.

    If include_content is True, also indexes full SKILL.md content as a third
    FTS5 field.
    """
    suffix = _index_suffix(exclude_gt)
    full = "_full" if include_content else ""
    db_path = os.path.join(INDEX_DIR, f"skills{full}{suffix}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = sqlite3.connect(db_path)

    if include_content:
        db.execute("""CREATE TABLE skills (
            rowid INTEGER PRIMARY KEY,
            name TEXT, skill_id TEXT, owner TEXT,
            description TEXT,
            github_stars INTEGER,
            skill_md_snippet TEXT,
            skill_md_content TEXT
        )""")
        db.execute("""CREATE VIRTUAL TABLE skills_fts USING fts5(
            name, description, skill_md_content,
            content=skills, content_rowid=rowid
        )""")
    else:
        db.execute("""CREATE TABLE skills (
            rowid INTEGER PRIMARY KEY,
            name TEXT, skill_id TEXT, owner TEXT,
            description TEXT,
            github_stars INTEGER,
            skill_md_snippet TEXT
        )""")
        db.execute("""CREATE VIRTUAL TABLE skills_fts USING fts5(
            name, description,
            content=skills, content_rowid=rowid
        )""")

    for i, r in enumerate(records):
        owner = r.get("owner", "")
        name = r.get("name", "")
        snippet = extract_skill_md_snippet(owner, name)
        if include_content:
            content = extract_skill_md_content(owner, name)
            db.execute(
                "INSERT INTO skills VALUES (?,?,?,?,?,?,?,?)",
                (
                    i, r["name"], r["skill_id"], r["owner"],
                    r["description"], r.get("github_stars", 0),
                    snippet, content,
                ),
            )
        else:
            db.execute(
                "INSERT INTO skills VALUES (?,?,?,?,?,?,?)",
                (
                    i, r["name"], r["skill_id"], r["owner"],
                    r["description"], r.get("github_stars", 0),
                    snippet,
                ),
            )

    db.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
    db.commit()
    db.close()
    print(f"FTS5 index built: {db_path} (include_content={include_content})")


def build_embeddings(records: list[dict], model_name: str = DEFAULT_EMBEDDING_MODEL,
                     exclude_gt: bool = False,
                     batch_size: int = 5120,
                     vllm_base_url: str = DEFAULT_VLLM_BASE_URL):
    """Compute dense embeddings for name+description via vLLM OpenAI-compatible server."""
    from openai import OpenAI

    client = OpenAI(base_url=vllm_base_url, api_key="unused")

    texts = []
    skill_ids = []
    for r in records:
        name = r.get("name") or ""
        desc = r.get("description") or ""
        text = f"{name}: {desc}".strip() or name
        texts.append(text)
        skill_ids.append(r.get("id", ""))

    print(f"Encoding {len(texts)} skills via vLLM at {vllm_base_url}...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        batch_embs = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embs)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    suffix = _index_suffix(exclude_gt)
    emb_path = os.path.join(INDEX_DIR, f"embeddings{suffix}.npy")
    ids_path = os.path.join(INDEX_DIR, f"skill_ids{suffix}.json")

    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(skill_ids, f)

    print(f"Embeddings saved: {emb_path} (shape: {embeddings.shape})")
    print(f"Skill IDs saved: {ids_path}")


def _prepare_content_texts(records: list[dict], model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Prepare truncated content texts for embedding. Returns (texts, skill_ids)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_tokens = 32768

    def truncate_to_tokens(text: str) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return tokenizer.decode(ids[:max_tokens])

    texts = []
    skill_ids = []
    for r in tqdm(records):
        owner = r.get("owner", "")
        name = r.get("name") or ""
        desc = r.get("description") or ""

        content = extract_skill_md_content(owner, name)
        if not content:
            content = f"{name}: {desc}".strip() or name

        texts.append(truncate_to_tokens(content))
        skill_ids.append(r.get("id", ""))

    return texts, skill_ids


def build_content_embeddings(records: list[dict], model_name: str = DEFAULT_EMBEDDING_MODEL,
                             exclude_gt: bool = False,
                             batch_size: int = 512,
                             vllm_base_url: str = DEFAULT_VLLM_BASE_URL):
    """Compute dense embeddings from full SKILL.md content via vLLM OpenAI-compatible server.

    Expects a running vLLM server, e.g.:
        vllm serve Qwen/Qwen3-Embedding-4B --task embed --tensor-parallel-size 4
    """
    from openai import OpenAI

    client = OpenAI(base_url=vllm_base_url, api_key="unused")

    print("Preparing content texts (tokenizing + truncating)...")
    texts, skill_ids = _prepare_content_texts(records, model_name)
    print(f"Encoding {len(texts)} skills (content embeddings) via vLLM at {vllm_base_url}...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        batch_embs = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embs)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    suffix = _index_suffix(exclude_gt)
    emb_path = os.path.join(INDEX_DIR, f"content_embeddings{suffix}.npy")
    ids_path = os.path.join(INDEX_DIR, f"content_skill_ids{suffix}.json")

    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(skill_ids, f)

    print(f"Content embeddings saved: {emb_path} (shape: {embeddings.shape})")
    print(f"Content skill IDs saved: {ids_path}")


def main():
    parser = argparse.ArgumentParser(description="Build skill search indexes")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of skills to index (for testing)",
    )
    parser.add_argument(
        "--mode", choices=["all", "fts5", "embeddings", "content_embeddings"],
        default="all",
        help="Which index to build (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--exclude-gt", action="store_true",
        help="Exclude ground-truth skills (benchflow-ai--*)",
    )
    parser.add_argument(
        "--include-content", action="store_true",
        help="Index full SKILL.md content in FTS5 and build content embeddings",
    )
    parser.add_argument(
        "--vllm-base-url", type=str, default=DEFAULT_VLLM_BASE_URL,
        help=f"vLLM OpenAI-compatible server URL (default: {DEFAULT_VLLM_BASE_URL})",
    )
    args = parser.parse_args()

    os.makedirs(INDEX_DIR, exist_ok=True)

    records = load_records(args.limit, exclude_gt=args.exclude_gt)
    print(f"Loaded {len(records)} records (licenses: {', '.join(sorted(ALLOWED_LICENSES))})")

    if args.mode in ("all", "fts5"):
        print("\n--- Building FTS5 index ---")
        build_fts_index(records, exclude_gt=args.exclude_gt,
                        include_content=args.include_content)

    if args.mode in ("all", "embeddings"):
        print("\n--- Building embeddings ---")
        build_embeddings(records, model_name=args.model, exclude_gt=args.exclude_gt,
                         vllm_base_url=args.vllm_base_url)

    if args.include_content and args.mode in ("all", "content_embeddings"):
        print("\n--- Building content embeddings ---")
        build_content_embeddings(records, model_name=args.model,
                                 exclude_gt=args.exclude_gt,
                                 vllm_base_url=args.vllm_base_url)

    print("\nDone!")


if __name__ == "__main__":
    main()
