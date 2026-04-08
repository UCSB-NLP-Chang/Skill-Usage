import re
import sqlite3
import json
import threading

_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}


def _sanitize_fts5_query(query: str) -> str:
    """Quote tokens that contain special characters to prevent FTS5 parse errors."""
    tokens = query.split()
    result = []
    for token in tokens:
        if token in _FTS5_OPERATORS:
            result.append(token)
        elif re.search(r'[^a-zA-Z0-9_*]', token):
            result.append('"' + token.replace('"', '') + '"')
        else:
            result.append(token)
    return " ".join(result)


class BM25Search:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._has_content = None  # auto-detected on first query

    def _get_db(self):
        if not hasattr(self._local, 'db'):
            self._local.db = sqlite3.connect(self.db_path)
            self._local.db.row_factory = sqlite3.Row
        return self._local.db

    def _detect_content_field(self):
        if self._has_content is None:
            db = self._get_db()
            cols = [r[1] for r in db.execute("PRAGMA table_info(skills)").fetchall()]
            self._has_content = "skill_md_content" in cols
        return self._has_content

    def search(self, query: str, top_k: int = 10,
               content_weight: float | None = None) -> list[dict]:
        """FTS5 search with BM25 ranking. Weights: name=10, description=5, content=content_weight.

        Args:
            content_weight: BM25 weight for skill_md_content field (default 5.0).
                            Only used when the index includes the content field.
        """
        query = _sanitize_fts5_query(query)
        has_content = self._detect_content_field()
        cw = content_weight if content_weight is not None else 5.0
        bm25_expr = f"bm25(skills_fts, 10.0, 5.0, {cw})" if has_content else "bm25(skills_fts, 10.0, 5.0)"
        rows = self._get_db().execute(
            f"""
            SELECT s.name, s.description, s.skill_md_snippet,
                   s.skill_id, s.github_stars,
                   {bm25_expr} AS score
            FROM skills_fts f
            JOIN skills s ON s.rowid = f.rowid
            WHERE skills_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if hasattr(self._local, 'db'):
            self._local.db.close()
            del self._local.db
