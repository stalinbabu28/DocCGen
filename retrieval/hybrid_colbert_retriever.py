from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from colbert import Searcher
from colbert.infra import Run, RunConfig

from retrieval.hybrid_retriever import hybrid_retrieve


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


def _normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


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


class HybridColBERTRetriever:
    """
    Fusion retriever:
    - existing hybrid retriever (MiniLM + sparse lexical)
    - ColBERT
    - normalized weighted score fusion
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        docs_dir: str | Path | None = None,
        experiment: str = "phase3_colbert",
        index_name: str = "phase3_colbert_ft",
        doc_map_path: str | Path | None = None,
        hybrid_weight: float = 0.60,
        colbert_weight: float = 0.40,
        candidate_pool: int = 100,
    ):
        self.project_root = Path(project_root).resolve()
        self.index_root = self.project_root / "colbert_data"
        self.experiment = experiment
        self.index_name = index_name
        self.hybrid_weight = float(hybrid_weight)
        self.colbert_weight = float(colbert_weight)
        self.candidate_pool = int(candidate_pool)

        if self.hybrid_weight < 0 or self.colbert_weight < 0:
            raise ValueError("hybrid_weight and colbert_weight must be non-negative")
        if self.hybrid_weight + self.colbert_weight == 0:
            raise ValueError("At least one fusion weight must be positive")

        self.docs_dir = Path(
            docs_dir
            or os.getenv("ANSIBLE_DOCS_DIR", str(self.project_root / "docs"))
        ).resolve()

        self.doc_map_path = Path(
            doc_map_path
            or (self.index_root / "data" / "doc_map.json")
        ).resolve()

        with self.doc_map_path.open("r", encoding="utf-8") as f:
            doc_map = json.load(f)

        if not isinstance(doc_map, list):
            raise ValueError(f"Expected doc_map to be a list, got {type(doc_map)}")

        self.doc_map: List[dict] = sorted(doc_map, key=lambda d: int(d["pid"]))
        self.docs_by_pid: Dict[int, dict] = {int(doc["pid"]): doc for doc in self.doc_map}
        self.docs_by_source: Dict[str, dict] = {}
        for doc in self.doc_map:
            src = str(doc.get("source", "")).strip()
            if src:
                self.docs_by_source[_normpath(src)] = doc

        with Run().context(
            RunConfig(
                root=str(self.index_root),
                experiment=self.experiment,
                nranks=1,
            )
        ):
            self.searcher = Searcher(index=self.index_name)

    @staticmethod
    def _normalize_scores(results: List[Tuple[float, str]]) -> Dict[str, float]:
        if not results:
            return {}

        scores = [float(s) for s, _ in results]
        mn = min(scores)
        mx = max(scores)

        if mx == mn:
            return {_normpath(path): 1.0 for _, path in results}

        out: Dict[str, float] = {}
        for score, path in results:
            out[_normpath(path)] = (float(score) - mn) / (mx - mn)
        return out

    @staticmethod
    def _module_fqn(doc: dict) -> str:
        module = str(doc.get("module", "")).strip()
        collection = str(doc.get("collection", "")).strip()
        return f"{collection}.{module}" if collection else module

    def _candidate_from_doc(
        self,
        doc: dict,
        *,
        score: float = 0.0,
        hybrid_score: float = 0.0,
        colbert_score: float = 0.0,
        rank: Optional[int] = None,
    ) -> dict:
        source = str(doc.get("source", "")).strip()
        module = str(doc.get("module", "")).strip()
        collection = str(doc.get("collection", "")).strip()
        module_fqn = self._module_fqn(doc)

        out = {
            "score": float(score),
            "fusion_score": float(score),
            "hybrid_score": float(hybrid_score),
            "colbert_score": float(colbert_score),
            "path": source,
            "source": source,
            "module": module,
            "module_fqn": module_fqn,
            "collection": collection,
            "short_description": doc.get("short_description", ""),
            "text": _doc_text(doc),
            "retriever": "hybrid_colbert",
        }
        if rank is not None:
            out["rank"] = int(rank)
        return out

    @lru_cache(maxsize=1)
    def _retrieve_colbert_raw(self, query: str, top_k: int) -> List[Tuple[float, str]]:
        results = self.searcher.search(query, k=top_k)

        if not isinstance(results, tuple):
            raise RuntimeError(f"Unexpected ColBERT result type: {type(results)}")

        if len(results) == 3:
            pids, ranks, scores = results
        elif len(results) == 2:
            pids, scores = results
            ranks = range(1, len(pids) + 1)
        else:
            raise RuntimeError(f"Unexpected ColBERT result tuple length: {len(results)}")

        out: List[Tuple[float, str]] = []
        for pid, rank, score in zip(pids, ranks, scores):
            pid_int = int(pid)
            doc = self.docs_by_pid.get(pid_int)
            if doc is None:
                continue
            src = str(doc.get("source", "")).strip()
            if not src:
                continue
            out.append((float(score), src))
        return out

    def _retrieve_hybrid_raw(self, query: str, top_k: int) -> List[Tuple[float, str]]:
        # Existing hybrid retriever returns (score, path)
        return hybrid_retrieve(query, k=top_k)

    def _lookup_doc(self, path: str) -> Optional[dict]:
        return self.docs_by_source.get(_normpath(path))

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        pool_k = max(self.candidate_pool, top_k * 10)

        hybrid_raw = self._retrieve_hybrid_raw(query, pool_k)
        colbert_raw = self._retrieve_colbert_raw(query, pool_k)

        hybrid_scores = self._normalize_scores(hybrid_raw)
        colbert_scores = self._normalize_scores(colbert_raw)

        all_paths = set(hybrid_scores.keys()) | set(colbert_scores.keys())
        fused: List[dict] = []

        for path in all_paths:
            doc = self._lookup_doc(path)
            if doc is None:
                # fallback so we still return something useful
                doc = {
                    "source": path,
                    "module": Path(path).stem,
                    "collection": "",
                    "short_description": "",
                }

            h = float(hybrid_scores.get(path, 0.0))
            c = float(colbert_scores.get(path, 0.0))

            final_score = (self.hybrid_weight * h) + (self.colbert_weight * c)

            fused.append(
                self._candidate_from_doc(
                    doc,
                    score=final_score,
                    hybrid_score=h,
                    colbert_score=c,
                )
            )

        fused.sort(
            key=lambda x: (
                x["score"],
                x["colbert_score"],
                x["hybrid_score"],
            ),
            reverse=True,
        )

        for idx, cand in enumerate(fused[:top_k], start=1):
            cand["rank"] = idx

        return fused[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k=top_k)