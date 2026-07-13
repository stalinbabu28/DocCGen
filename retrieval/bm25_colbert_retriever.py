from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from colbert import Searcher
from colbert.infra import Run, RunConfig

from retrieval.bm25_retriever import BM25Retriever


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


class BM25ColBERTRetriever:
    """
    DocCGen-style retrieval:
    - BM25 sparse retrieval
    - ColBERT dense retrieval
    - Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        docs_dir: str | Path | None = None,
        experiment: str = "phase3_colbert",
        index_name: str = "phase3_colbert_zero",
        doc_map_path: str | Path | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 100,
    ):
        self.project_root = Path(project_root).resolve()
        self.index_root = self.project_root / "colbert_data"
        self.experiment = experiment
        self.index_name = index_name
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

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
        self.docs_by_pid = {int(doc["pid"]): doc for doc in self.doc_map}

        self.bm25 = BM25Retriever(
            docs_dir=self.docs_dir,
            doc_map_path=self.doc_map_path,
        )

        with Run().context(
            RunConfig(
                root=str(self.index_root),
                experiment=self.experiment,
                nranks=1,
            )
        ):
            self.searcher = Searcher(index=self.index_name)

    def _retrieve_colbert(self, query: str, top_k: int) -> List[Dict[str, Any]]:
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

        out: List[Dict[str, Any]] = []
        for pid, rank, score in zip(pids, ranks, scores):
            pid_int = int(pid)
            doc = self.docs_by_pid.get(pid_int)
            if doc is None:
                continue

            module = str(doc.get("module", "")).strip()
            collection = str(doc.get("collection", "")).strip()
            module_fqn = f"{collection}.{module}" if collection else module

            out.append(
                {
                    "pid": pid_int,
                    "colbert_rank": int(rank),
                    "colbert_score": float(score),
                    "path": doc.get("source", ""),
                    "source": doc.get("source", ""),
                    "module": module,
                    "module_fqn": module_fqn,
                    "collection": collection,
                    "short_description": doc.get("short_description", ""),
                    "text": _doc_text(doc),
                }
            )

        return out

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _merge_candidate(
        self,
        fused: Dict[int, Dict[str, Any]],
        doc_key: int,
        base: Dict[str, Any],
        branch: str,
        rank: int,
        score: float,
    ) -> None:
        cand = fused.get(doc_key)
        if cand is None:
            cand = dict(base)
            cand.setdefault("pid", doc_key)
            cand.setdefault("score", 0.0)
            cand.setdefault("bm25_score", 0.0)
            cand.setdefault("colbert_score", 0.0)
            cand.setdefault("bm25_rank", None)
            cand.setdefault("colbert_rank", None)
            cand.setdefault("fusion_score", 0.0)
            cand.setdefault("retriever", "bm25_colbert")
            fused[doc_key] = cand

        cand["fusion_score"] += self._rrf(rank)

        if branch == "bm25":
            cand["bm25_score"] = float(score)
            cand["bm25_rank"] = int(rank)
        else:
            cand["colbert_score"] = float(score)
            cand["colbert_rank"] = int(rank)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        pool_k = max(top_k * 10, self.candidate_pool)

        bm25_results = self.bm25.retrieve(query, top_k=pool_k)
        colbert_results = self._retrieve_colbert(query, top_k=pool_k)

        fused: Dict[int, Dict[str, Any]] = {}

        for item in bm25_results:
            pid = item.get("pid")
            if pid is None:
                continue
            pid_int = int(pid)
            self._merge_candidate(
                fused=fused,
                doc_key=pid_int,
                base=item,
                branch="bm25",
                rank=int(item.get("rank", 0)),
                score=float(item.get("score", 0.0)),
            )

        for item in colbert_results:
            pid = item.get("pid")
            if pid is None:
                continue
            pid_int = int(pid)
            self._merge_candidate(
                fused=fused,
                doc_key=pid_int,
                base=item,
                branch="colbert",
                rank=int(item.get("colbert_rank", 0)),
                score=float(item.get("colbert_score", 0.0)),
            )

        ranked = sorted(
            fused.values(),
            key=lambda c: (
                float(c.get("fusion_score", 0.0)),
                float(c.get("colbert_score", 0.0)),
                float(c.get("bm25_score", 0.0)),
            ),
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        for rank, cand in enumerate(ranked[:top_k], start=1):
            out.append(
                {
                    "pid": int(cand.get("pid", -1)),
                    "rank": rank,
                    "score": float(cand.get("fusion_score", 0.0)),
                    "bm25_score": float(cand.get("bm25_score", 0.0)),
                    "colbert_score": float(cand.get("colbert_score", 0.0)),
                    "bm25_rank": cand.get("bm25_rank"),
                    "colbert_rank": cand.get("colbert_rank"),
                    "path": cand.get("path", ""),
                    "source": cand.get("source", ""),
                    "module": cand.get("module", ""),
                    "module_fqn": cand.get("module_fqn", cand.get("module", "")),
                    "collection": cand.get("collection", ""),
                    "short_description": cand.get("short_description", ""),
                    "text": cand.get("text", ""),
                    "retriever": cand.get("retriever", "bm25_colbert"),
                }
            )

        return out

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k=top_k)