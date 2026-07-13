from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COLBERT_INDEX_ROOT = Path(
    os.getenv("COLBERT_INDEX_ROOT", str(PROJECT_ROOT / "colbert_data"))
).resolve()
COLBERT_EXPERIMENT = os.getenv("COLBERT_EXPERIMENT", "phase3_colbert")
COLBERT_ZERO_INDEX_NAME = os.getenv("COLBERT_ZERO_INDEX_NAME", "phase3_colbert_zero")
COLBERT_FT_INDEX_NAME = os.getenv("COLBERT_FT_INDEX_NAME", "phase3_colbert_ft")
BM25_DOC_MAP = Path(
    os.getenv("BM25_DOC_MAP", str(PROJECT_ROOT / "colbert_data" / "data" / "doc_map.json"))
).resolve()

RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "colbert_zero").strip().lower()

INFO_CUES = (
    " show ",
    " details ",
    " information ",
    " info ",
    " get ",
    " list ",
    " display ",
    " describe ",
)

RESOURCE_HINTS: List[Tuple[str, str]] = [
    ("virtual network gateway", "virtualnetworkgateway"),
    ("virtual network", "virtualnetwork"),
    ("network interface", "networkinterface"),
    ("dns zone", "dnszone"),
    ("public ip address", "publicipaddress"),
    ("route table", "routetable"),
    ("storage account", "storageaccount"),
    ("managed disk", "manageddisk"),
    ("load balancer", "loadbalancer"),
    ("application gateway", "appgateway"),
    ("key vault", "keyvault"),
    ("container registry", "containerregistry"),
    ("web app", "webapp"),
    ("aks cluster", "aks"),
    ("sql database", "sqldatabase"),
    ("sql server", "sqlserver"),
    ("subnet", "subnet"),
]


def _normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


@lru_cache(maxsize=1)
def _load_doc_map() -> List[dict]:
    path = PROJECT_ROOT / "colbert_data" / "data" / "doc_map.json"
    with path.open("r", encoding="utf-8") as f:
        doc_map = json.load(f)

    if not isinstance(doc_map, list):
        raise ValueError(f"Expected doc_map to be a list, got {type(doc_map)}")

    return doc_map


@lru_cache(maxsize=1)
def _docs_by_pid() -> List[dict]:
    return sorted(_load_doc_map(), key=lambda d: int(d["pid"]))


@lru_cache(maxsize=1)
def _docs_by_source() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for doc in _load_doc_map():
        src = doc.get("source", "")
        if src:
            out[_normpath(src)] = doc
    return out


@lru_cache(maxsize=1)
def _docs_by_module() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for doc in _load_doc_map():
        module = str(doc.get("module", "")).strip()
        if module:
            out[module] = doc
    return out


def _lookup_doc(ref: str) -> Optional[dict]:
    if not ref:
        return None
    return _docs_by_source().get(_normpath(ref)) or _docs_by_module().get(ref)


def _module_slug(candidate: dict) -> str:
    module_fqn = str(candidate.get("module_fqn") or candidate.get("module") or "").strip()
    slug = module_fqn.split(".")[-1].lower()

    for prefix in ("azure_rm_", "azure_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]

    if slug.endswith("_info"):
        slug = slug[:-5]

    return slug


def _query_has_info_cue(query: str) -> bool:
    q = f" {query.lower()} "
    return any(cue in q for cue in INFO_CUES)


def _resource_hint_boost(query: str, candidate: dict) -> float:
    q = f" {query.lower()} "
    slug = _module_slug(candidate)

    score = float(candidate.get("score", 0.0))

    if _query_has_info_cue(query):
        if str(candidate.get("module_fqn", "")).endswith("_info"):
            score += 4.0
        else:
            score -= 1.5
    else:
        if str(candidate.get("module_fqn", "")).endswith("_info"):
            score -= 1.0

    for phrase, keyword in RESOURCE_HINTS:
        if phrase not in q:
            continue

        if keyword == "virtualnetwork":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0
            elif "networkinterface" in slug:
                score -= 6.0

        elif keyword == "dnszone":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0
            elif "dnszonegroup" in slug:
                score -= 4.0

        elif keyword == "keyvault":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0

        else:
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0

    return score


def _candidate_from_doc(doc: dict, score: float = 0.0, rank: Optional[int] = None) -> dict:
    module = str(doc.get("module", "")).strip()
    source = str(doc.get("source", "")).strip()
    collection = ".".join(module.split(".")[:-1]) if "." in module else ""

    candidate = {
        "score": float(score),
        "path": source,
        "source": source,
        "module": module,
        "module_fqn": module,
        "collection": collection,
        "short_description": doc.get("short_description", ""),
    }

    if rank is not None:
        candidate["rank"] = int(rank)

    return candidate


def _normalize_raw_candidate(raw: Any) -> Optional[dict]:
    if raw is None:
        return None

    if isinstance(raw, dict):
        candidate = dict(raw)

        if not candidate.get("source") and candidate.get("path"):
            candidate["source"] = candidate["path"]
        if not candidate.get("path") and candidate.get("source"):
            candidate["path"] = candidate["source"]

        if not candidate.get("module") and candidate.get("module_fqn"):
            candidate["module"] = candidate["module_fqn"].split(".")[-1]
        if not candidate.get("module_fqn") and candidate.get("module"):
            candidate["module_fqn"] = candidate["module"]

        if not candidate.get("collection") and candidate.get("module_fqn"):
            candidate["collection"] = ".".join(candidate["module_fqn"].split(".")[:-1])

        if not candidate.get("short_description"):
            ref = candidate.get("source") or candidate.get("path") or candidate.get("module_fqn") or candidate.get("module")
            doc = _lookup_doc(str(ref))
            if doc:
                candidate["short_description"] = doc.get("short_description", "")
                candidate.setdefault("module", doc.get("module", ""))
                candidate.setdefault("module_fqn", doc.get("module", ""))
                candidate.setdefault("collection", ".".join(str(doc.get("module", "")).split(".")[:-1]))
                candidate.setdefault("source", doc.get("source", ""))
                candidate.setdefault("path", doc.get("source", ""))

        if "score" not in candidate:
            candidate["score"] = 0.0

        return candidate

    if isinstance(raw, str):
        doc = _lookup_doc(raw)
        if doc:
            return _candidate_from_doc(doc, score=0.0)
        return {
            "score": 0.0,
            "path": raw,
            "source": raw,
            "module": Path(raw).stem,
            "module_fqn": Path(raw).stem,
            "collection": "",
            "short_description": "",
        }

    if isinstance(raw, (tuple, list)):
        if len(raw) >= 2 and isinstance(raw[1], str):
            score = float(raw[0]) if len(raw) >= 1 else 0.0
            ref = raw[1]
            doc = _lookup_doc(ref)
            if doc:
                return _candidate_from_doc(doc, score=score)
            return {
                "score": score,
                "path": ref,
                "source": ref,
                "module": Path(ref).stem,
                "module_fqn": Path(ref).stem,
                "collection": "",
                "short_description": "",
            }

    return None


@lru_cache(maxsize=1)
def _load_hybrid_backend() -> Callable[[str, int], List[Any]]:
    module = import_module("retrieval.hybrid_retriever")

    candidate_names = [
        "get_ranked_candidates",
        "retrieve_topk",
        "rank_documents",
        "search",
        "retrieve",
    ]

    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            def _call(query: str, k: int, _fn=fn):
                try:
                    return _fn(query, k=k)
                except TypeError:
                    try:
                        return _fn(query, k)
                    except TypeError:
                        return _fn(query)
            return _call

    cls_names = [
        "HybridRetriever",
        "HybridRetrieval",
        "Retriever",
    ]
    for cls_name in cls_names:
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        try:
            instance = cls()
        except Exception:
            continue

        for meth_name in ("get_ranked_candidates", "retrieve", "search", "rank"):
            meth = getattr(instance, meth_name, None)
            if callable(meth):
                def _call(query: str, k: int, _meth=meth):
                    try:
                        return _meth(query, k=k)
                    except TypeError:
                        try:
                            return _meth(query, k)
                        except TypeError:
                            return _meth(query)
                return _call

    raise ImportError(
        "Could not find a usable hybrid retriever entrypoint in retrieval.hybrid_retriever"
    )


@lru_cache(maxsize=1)
def _load_bm25_backend():
    from retrieval.bm25_retriever import BM25Retriever

    return BM25Retriever(doc_map_path=BM25_DOC_MAP)


@lru_cache(maxsize=2)
def _load_colbert_searcher(index_name: str):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig

    with Run().context(
        RunConfig(
            root=str(COLBERT_INDEX_ROOT),
            experiment=COLBERT_EXPERIMENT,
            nranks=1,
        )
    ):
        searcher = Searcher(index=index_name)

    return searcher


@lru_cache(maxsize=2)
def _load_bm25_colbert_backend(index_name: str):
    from retrieval.bm25_colbert_retriever import BM25ColBERTRetriever

    return BM25ColBERTRetriever(
        project_root=PROJECT_ROOT,
        experiment=COLBERT_EXPERIMENT,
        index_name=index_name,
        doc_map_path=BM25_DOC_MAP,
    )

def _load_hybrid_colbert_backend(index_name: str):
    from retrieval.hybrid_colbert_retriever import HybridColBERTRetriever

    return HybridColBERTRetriever(
        project_root=PROJECT_ROOT,
        experiment=COLBERT_EXPERIMENT,
        index_name=index_name,
    )


def _retrieve_hybrid(query: str, k: int) -> List[dict]:
    backend = _load_hybrid_backend()
    raw = backend(query, k)

    if isinstance(raw, tuple):
        if len(raw) == 2 and hasattr(raw[1], "__len__"):
            out: List[dict] = []
            a, b = raw
            if len(a) == len(b):
                if len(a) > 0 and isinstance(a[0], (str, dict, tuple, list)):
                    items, scores = a, b
                else:
                    scores, items = a, b
                for item, score in zip(items, scores):
                    cand = _normalize_raw_candidate(item)
                    if cand is None:
                        continue
                    cand["score"] = float(score)
                    out.append(cand)
                return out

    if isinstance(raw, list):
        out: List[dict] = []
        for item in raw:
            cand = _normalize_raw_candidate(item)
            if cand is not None:
                out.append(cand)
        return out

    cand = _normalize_raw_candidate(raw)
    return [cand] if cand is not None else []


def _retrieve_bm25(query: str, k: int) -> List[dict]:
    backend = _load_bm25_backend()
    raw = backend.retrieve(query, top_k=k)
    return [cand for cand in (_normalize_raw_candidate(x) for x in raw) if cand is not None]


def _retrieve_colbert(query: str, k: int, index_name: str) -> List[dict]:
    searcher = _load_colbert_searcher(index_name)
    results = searcher.search(query, k=k)

    if not isinstance(results, tuple):
        raise RuntimeError(f"Unexpected ColBERT result type: {type(results)}")

    if len(results) == 3:
        pids, ranks, scores = results
    elif len(results) == 2:
        pids, scores = results
        ranks = range(1, len(pids) + 1)
    else:
        raise RuntimeError(f"Unexpected ColBERT result tuple length: {len(results)}")

    docs = _docs_by_pid()
    out: List[dict] = []

    for pid, rank, score in zip(pids, ranks, scores):
        pid_int = int(pid)
        if pid_int < 0 or pid_int >= len(docs):
            continue
        doc = docs[pid_int]
        out.append(_candidate_from_doc(doc, score=float(score), rank=int(rank)))

    return out


def _retrieve_bm25_colbert(query: str, k: int, index_name: str) -> List[dict]:
    backend = _load_bm25_colbert_backend(index_name)
    raw = backend.retrieve(query, top_k=k)
    return [cand for cand in (_normalize_raw_candidate(x) for x in raw) if cand is not None]

def _retrieve_hybrid_colbert(query: str, k: int, index_name: str) -> List[dict]:
    backend = _load_hybrid_colbert_backend(index_name)
    raw = backend.retrieve(query, top_k=k)

    return [
        cand
        for cand in (_normalize_raw_candidate(x) for x in raw)
        if cand is not None
    ]

def _selected_mode() -> str:
    mode = RETRIEVER_MODE
    if mode in {"colbert", "zero", "zero_shot", "zero-shot"}:
        return "colbert_zero"
    if mode in {"ft", "fine_tuned", "fine-tuned"}:
        return "colbert_ft"
    if mode in {"bm25+colbert", "bm25-colbert", "bm25_colbert_fusion"}:
        return "bm25_colbert"
    if mode in {"bm25-only", "bm25_sparse", "bm25s"}:
        return "bm25"
    if mode in {
        "hybrid+colbert",
        "hybrid-colbert",
        "hybrid_colbert",
    }:
        return "hybrid_colbert"
    return mode


def _raw_candidates(query: str, k: int) -> List[dict]:
    mode = _selected_mode()

    if mode == "hybrid":
        raw_candidates = _retrieve_hybrid(query, k)
    elif mode == "bm25":
        raw_candidates = _retrieve_bm25(query, k)
    elif mode == "colbert_zero":
        raw_candidates = _retrieve_colbert(query, k, COLBERT_ZERO_INDEX_NAME)
    elif mode == "colbert_ft":
        raw_candidates = _retrieve_colbert(query, k, COLBERT_FT_INDEX_NAME)
    elif mode == "bm25_colbert":
        raw_candidates = _retrieve_bm25_colbert(query, k, COLBERT_ZERO_INDEX_NAME)
    elif mode == "hybrid_colbert":
        raw_candidates = _retrieve_hybrid_colbert(
            query,
            k,
            COLBERT_FT_INDEX_NAME,
        )
    else:
        raise ValueError(
            f"Unknown RETRIEVER_MODE={RETRIEVER_MODE!r}. "
            f"Use hybrid, bm25, bm25_colbert, hybrid_colbert, colbert_zero, colbert, or colbert_ft."
        )

    normalized: List[dict] = []
    for raw in raw_candidates:
        cand = _normalize_raw_candidate(raw)
        if cand is not None:
            normalized.append(cand)

    return normalized


def get_ranked_candidates(
    query: str,
    k: int = 10,
    **kwargs,
):
    """
    Returns ranked candidates using one of:
      - hybrid
      - bm25
      - bm25_colbert
      - colbert_zero / colbert
      - colbert_ft

    Controlled by:
      RETRIEVER_MODE=hybrid | bm25 | bm25_colbert | colbert_zero | colbert | colbert_ft
    """
    candidates = _raw_candidates(query, k)

    for cand in candidates:
        cand["heuristic_score"] = _resource_hint_boost(query, cand)
        cand["final_score"] = float(cand.get("score", 0.0)) + float(cand["heuristic_score"])

    candidates.sort(
        key=lambda c: (
            c.get("final_score", 0.0),
            c.get("score", 0.0),
        ),
        reverse=True,
    )

    return candidates


def select_best_candidate(
    query: str,
    k: int = 10,
) -> Optional[dict]:
    candidates = get_ranked_candidates(query, k=k)
    return candidates[0] if candidates else None