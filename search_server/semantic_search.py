import json
import os
import sqlite3
import threading

import numpy as np


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

QUERY_INSTRUCTION = "Find skills matching this query: "


class SemanticSearch:
    def __init__(self, embeddings_path: str, skill_ids_path: str, db_path: str,
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 content_embeddings_path: str | None = None,
                 content_weight: float = 0.05):
        self.embeddings = np.load(embeddings_path)  # (N, D), normalized
        with open(skill_ids_path) as f:
            self.skill_ids = json.load(f)
        self.db_path = db_path
        self._model_name = model_name
        self._local = threading.local()
        self.content_weight = content_weight

        # Load content embeddings if available
        if content_embeddings_path and os.path.exists(content_embeddings_path):
            self.content_embeddings = np.load(content_embeddings_path)  # (N, D), normalized
        else:
            self.content_embeddings = None

        # Lazy-load model on first query
        self._model = None

    def _get_db(self):
        if not hasattr(self._local, 'db'):
            self._local.db = sqlite3.connect(self.db_path)
            self._local.db.row_factory = sqlite3.Row
        return self._local.db

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def search(self, query: str, top_k: int = 10,
               content_weight: float | None = None) -> list[dict]:
        """Semantic search. content_weight overrides self.content_weight if given."""
        model = self._get_model()
        q_emb = model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
        )  # (1, D)

        summary_scores = (self.embeddings @ q_emb.T).flatten()
        if self.content_embeddings is not None:
            content_scores = (self.content_embeddings @ q_emb.T).flatten()
            w = content_weight if content_weight is not None else self.content_weight
            scores = (1 - w) * summary_scores + w * content_scores
        else:
            scores = summary_scores
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        db = self._get_db()
        results = []
        for idx in top_indices:
            row = db.execute(
                """SELECT name, description, skill_md_snippet,
                          skill_id, github_stars
                   FROM skills WHERE rowid = ?""",
                (int(idx),),
            ).fetchone()
            if row:
                r = dict(row)
                r["score"] = float(scores[idx])
                results.append(r)
        return results

    def close(self):
        if hasattr(self._local, 'db'):
            self._local.db.close()
            del self._local.db
