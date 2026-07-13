from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

try:
    # Reuse your project's stopwords if available.
    from pipeline.schema_text_utils import STOPWORDS as PROJECT_STOPWORDS
except Exception:
    PROJECT_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
        "create", "do", "does", "for", "from", "get", "give", "if", "in", "into",
        "is", "it", "let", "make", "me", "need", "of", "on", "or", "run", "set",
        "show", "that", "the", "this", "to", "try", "use", "using", "with", "you",
        "your", "ensure", "present", "running", "condition", "file", "job",
    }


TOKEN_RE = re.compile(r"[a-z0-9_./-]+")


def _normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def tokenize(text: str) -> List[str]:
    toks = TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if t not in PROJECT_STOPWORDS and len(t) > 1]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_as_str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {_as_str(v)}" for k, v in value.items())
    return str(value)


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _module_fqn_from_doc(doc: dict) -> str:
    module = _as_str(doc.get("module", "")).strip()
    collection = _as_str(doc.get("collection", "")).strip()

    if module and "." in module:
        return module

    if collection and module:
        return f"{collection}.{module}"

    # fallback for docs that only expose a path or title
    title = _as_str(doc.get("short_description", "")).strip()
    if title:
        return title

    return module or collection or ""


def _doc_text(doc: dict) -> str:
    parts = [
        _as_str(doc.get("module", "")),
        _as_str(doc.get("collection", "")),
        _as_str(doc.get("short_description", "")),
        _as_str(doc.get("description", "")),
        _as_str(doc.get("examples", "")),
        _as_str(doc.get("text", "")),
        _as_str(doc.get("body", "")),
        _as_str(doc.get("content", "")),
        _as_str(doc.get("parameters", "")),
    ]
    return " ".join(p for p in parts if p.strip())


@dataclass(frozen=True)
class BM25Candidate:
    score: float
    path: str
    module: str
    module_fqn: str
    collection: str
    short_description: str


class BM25Retriever:
    """
    Standalone BM25 retriever over your Ansible docs corpus.

    It loads docs once, builds a BM25Okapi index, and returns top-k
    candidates in a shape that matches your existing pipeline.
    """

    def __init__(
        self,
        docs_dir: str | Path,
        *,
        doc_map_path: str | Path | None = None,
        max_docs: Optional[int] = None,
    ):
        self.docs_dir = Path(docs_dir).resolve()
        self.doc_map_path = Path(doc_map_path).resolve() if doc_map_path else None
        self.max_docs = max_docs

        self.docs: List[dict] = []
        self.doc_texts: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self._build_index()

    def _load_docs_from_doc_map(self, doc_map_path: Path) -> List[dict]:
        with doc_map_path.open("r", encoding="utf-8") as f:
            doc_map = json.load(f)

        if not isinstance(doc_map, list):
            raise ValueError(f"Expected doc_map to be a list, got {type(doc_map)}")

        docs: List[dict] = []
        for item in doc_map:
            if not isinstance(item, dict):
                continue
            docs.append(item)

        return docs

    def _load_docs_from_json_dir(self, docs_dir: Path) -> List[dict]:
        docs: List[dict] = []
        for path in sorted(docs_dir.rglob("*.json")):
            obj = _load_json(path)
            if obj is None:
                continue

            # Keep the original path so the retriever can hand back file refs.
            obj.setdefault("source", str(path))
            obj.setdefault("path", str(path))
            docs.append(obj)

        return docs

    def _canonicalize_doc(self, doc: dict) -> dict:
        out = dict(doc)

        source = _as_str(out.get("source", "")).strip() or _as_str(out.get("path", "")).strip()
        if not source and out.get("module"):
            # fallback for very old docs
            source = str(self.docs_dir / f"{_as_str(out.get('module')).strip()}.json")
        out["source"] = source
        out["path"] = source

        module_fqn = _module_fqn_from_doc(out)
        out["module_fqn"] = module_fqn

        # If module is not present, use the last part of module_fqn.
        module = _as_str(out.get("module", "")).strip()
        if not module and module_fqn:
            module = module_fqn.split(".")[-1]
        out["module"] = module or module_fqn

        collection = _as_str(out.get("collection", "")).strip()
        if not collection and module_fqn and "." in module_fqn:
            collection = ".".join(module_fqn.split(".")[:-1])
        out["collection"] = collection

        short_description = _as_str(out.get("short_description", "")).strip()
        if not short_description:
            short_description = _as_str(out.get("title", "")).strip()
        out["short_description"] = short_description

        return out

    def _build_index(self) -> None:
        if self.doc_map_path and self.doc_map_path.exists():
            raw_docs = self._load_docs_from_doc_map(self.doc_map_path)
        else:
            raw_docs = self._load_docs_from_json_dir(self.docs_dir)

        if self.max_docs is not None:
            raw_docs = raw_docs[: self.max_docs]

        self.docs = [self._canonicalize_doc(doc) for doc in raw_docs]

        self.doc_texts = [_doc_text(doc) for doc in self.docs]
        self.doc_tokens = [tokenize(text) for text in self.doc_texts]

        # BM25 requires tokenized corpus.
        self.bm25 = BM25Okapi(self.doc_tokens)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.docs:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)

        if scores.size == 0:
            return []

        k = min(top_k, len(scores))
        # argpartition is faster than full sort for large corpora.
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        candidates: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_idx, start=1):
            doc = self.docs[int(idx)]
            score = float(scores[int(idx)])

            candidates.append(
                {
                    "score": score,
                    "bm25_score": score,
                    "rank": rank,
                    "path": doc.get("path", ""),
                    "source": doc.get("source", ""),
                    "module": doc.get("module", ""),
                    "module_fqn": doc.get("module_fqn", ""),
                    "collection": doc.get("collection", ""),
                    "short_description": doc.get("short_description", ""),
                    "text": _doc_text(doc),
                }
            )

        return candidates

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k=top_k)