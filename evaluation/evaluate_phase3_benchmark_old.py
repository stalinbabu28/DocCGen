from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from pipeline.parser_state_decoder import generate_parser_state_yaml
from pipeline.schema_extractor import extract_schema
from pipeline.schema_text_utils import STOPWORDS
from retrieval.rerank_results import rerank

# Cap how much a single repeated token can contribute to the sparse score.
# Without this, a long, verbose module description that happens to repeat a
# common query word many times (e.g. "from", "to", "template") can outscore a
# short, exactly-correct module purely on word count, unrelated to relevance.
MAX_TOKEN_MATCH_CONTRIBUTION = 2


# ==============================================================================
# PLUGGABLE SELECTOR INTERFACE & IMPLEMENTATION
# ==============================================================================
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

RESOURCE_HINTS = [
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

# The RESOURCE_HINTS list above only knows Azure vocabulary, so it silently
# contributes nothing for every other collection in a universal corpus (AWS,
# GCP, Cisco, NetApp, Windows, Dell, F5, Zabbix, VMware, ...). Without any
# equivalent signal, retrieval falls back purely to dense/sparse text
# similarity, which regularly picks a wordy vendor-specific module over the
# short generic ansible.builtin one for plain, vendor-neutral queries (e.g.
# "Copy a file to X" -> ansible.windows.win_copy instead of
# ansible.builtin.copy). This table gives a *generic* version of the same
# idea: if a candidate belongs to a specific vendor/OS family and the query
# never names that vendor/OS, penalize it. Collections not listed here (and
# ansible.builtin / community.general) are left untouched.
VENDOR_FAMILY_KEYWORDS = {
    "cisco": ("cisco", "ise", "dnac", "catalyst", "ios", "nxos", "meraki"),
    "netapp": ("netapp", "ontap"),
    "dellemc": ("dell", "emc", "unity", "os9", "os10", "os6"),
    "f5networks": ("f5", "bigip", "big-ip"),
    "oracle": ("oracle", "oci"),
    "google": ("google", "gcp"),
    "vmware": ("vmware", "esxi", "vcenter"),
    "amazon": ("aws", "amazon", "ec2", "s3"),
    "azure": ("azure",),
    "inspur": ("inspur",),
    "community.zabbix": ("zabbix",),
    "community.postgresql": ("postgres", "postgresql"),
    "community.okd": ("okd", "openshift"),
    "community.routeros": ("routeros", "mikrotik"),
    "community.docker": ("docker", "container"),
    "ansible.windows": ("windows", "win", "iis", "powershell"),
    "community.windows": ("windows", "win", "iis", "powershell"),
}

VENDOR_MISMATCH_PENALTY = -0.15


def _vendor_family(module_fqn: str) -> Optional[str]:
    fqn_lower = module_fqn.lower()
    if "windows" in fqn_lower or re.search(r"(^|\.)win_", fqn_lower):
        return "ansible.windows"
    for family in VENDOR_FAMILY_KEYWORDS:
        if family in {"ansible.windows", "community.windows"}:
            continue
        if fqn_lower.startswith(family):
            return family
    return None


def _load_doc(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DocumentSelector:
    def retrieve(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError


class PipelineHybridSelector(DocumentSelector):
    """
    Deterministic hybrid retriever for benchmark evaluation.

    Important:
    - No LLM module selection.
    - Returns top-ranked candidate directly.
    - Still provides candidate list for Top-1/5/10 evaluation.
    """

    def __init__(self, docs_dir: str, embeddings_file: str):
        self.docs_dir = docs_dir
        self.embeddings_file = embeddings_file
        self.model = None
        self.database = None

    def _lazy_init(self):
        if self.model is not None:
            return

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if self.embeddings_file and os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "rb") as f:
                db = pickle.load(f)

            self.database = []
            for item in db:
                filename = os.path.basename(item["path"])
                normalized_path = os.path.join(self.docs_dir, filename)
                self.database.append(
                    {
                        "embedding": item["embedding"],
                        "path": normalized_path,
                    }
                )
        else:
            self.database = []

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _dense_retrieve(self, query: str, k: int = 50) -> List[Tuple[float, str]]:
        self._lazy_init()
        if not self.database:
            return []
        query_emb = self.model.encode(query)
        scores = []
        for item in self.database:
            score = self._cosine_similarity(query_emb, item["embedding"])
            scores.append((score, item["path"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9_./-]+", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def _sparse_retrieve(self, query: str, k: int = 50) -> List[Tuple[float, str]]:
        if not self.docs_dir or not os.path.isdir(self.docs_dir):
            return []

        query_tokens = self._tokenize(query)
        compact_query = re.sub(r"[^a-z0-9]+", "", query.lower())

        scores = []
        for file in os.listdir(self.docs_dir):
            if not file.endswith(".json"):
                continue

            path = os.path.join(self.docs_dir, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            module = data.get("module", "")
            short_desc = data.get("short_description", "")
            description = " ".join(data.get("description", []))

            text = f"{module} {short_desc} {description}"
            text_lower = text.lower()
            module_lower = module.lower()
            compact_module = re.sub(r"[^a-z0-9]+", "", module_lower)

            score = 0.0
            for token in query_tokens:
                score += min(text_lower.count(token), MAX_TOKEN_MATCH_CONTRIBUTION)
                if token in module_lower:
                    score += 3.0

            if compact_query and compact_query in compact_module:
                score += 12.0

            score -= 0.05 * len(module_lower.split("_"))
            scores.append((float(score), path))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

    def _normalize_scores(self, results: List[Tuple[float, str]]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [score for score, _ in results]
        mn = min(scores)
        mx = max(scores)
        if mx == mn:
            return {path: 1.0 for _, path in results}
        return {path: (score - mn) / (mx - mn) for score, path in results}

    def _module_slug(self, candidate: dict) -> str:
        slug = candidate["module_fqn"].split(".")[-1].lower()
        for prefix in ("azure_rm_", "azure_"):
            if slug.startswith(prefix):
                slug = slug[len(prefix):]
        if slug.endswith("_info"):
            slug = slug[:-5]
        return slug

    def _resource_hint_boost(self, query: str, candidate: dict) -> float:
        q = f" {query.lower()} "
        slug = self._module_slug(candidate)
        score = float(candidate.get("score", 0.0))

        has_info = any(cue in q for cue in INFO_CUES)
        if has_info:
            if candidate["module_fqn"].endswith("_info"):
                score += 4.0
            else:
                score -= 1.5
        else:
            if candidate["module_fqn"].endswith("_info"):
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

    def _vendor_mismatch_penalty(self, query: str, candidate: dict) -> float:
        family = _vendor_family(candidate["module_fqn"])
        if family is None:
            return 0.0
        keywords = VENDOR_FAMILY_KEYWORDS[family]
        q = f" {query.lower()} "
        if any(f" {kw} " in q or kw in q for kw in keywords):
            return 0.0
        return VENDOR_MISMATCH_PENALTY

    def retrieve(self, query: str) -> Dict[str, Any]:
        dense_res = self._dense_retrieve(query, k=50)
        sparse_res = self._sparse_retrieve(query, k=50)

        dense_map = self._normalize_scores(dense_res)
        sparse_map = self._normalize_scores(sparse_res)

        all_paths = set(dense_map.keys()) | set(sparse_map.keys())
        combined = []
        for path in all_paths:
            dense_score = dense_map.get(path, 0.0)
            sparse_score = sparse_map.get(path, 0.0)
            final_score = 0.7 * dense_score + 0.3 * sparse_score
            combined.append((final_score, path))

        combined = rerank(query, combined)
        combined.sort(key=lambda x: x[0], reverse=True)

        candidates = []
        for score, path in combined[:50]:
            doc = _load_doc(path)
            module = doc.get("module", "")
            collection = doc.get("collection", "")
            module_fqn = f"{collection}.{module}" if collection else module

            candidate = {
                "score": score,
                "path": path,
                "module": module,
                "module_fqn": module_fqn,
                "collection": collection,
                "short_description": doc.get("short_description", ""),
            }
            candidate["heuristic_score"] = (
                self._resource_hint_boost(query, candidate)
                + self._vendor_mismatch_penalty(query, candidate)
            )
            candidate["final_score"] = candidate["score"] + candidate["heuristic_score"]
            candidates.append(candidate)

        candidates.sort(key=lambda c: (c["final_score"], c["score"]), reverse=True)

        if not candidates:
            raise ValueError(f"No candidates found for query: {query}")

        # Deterministic: take top candidate directly, no LLM call.
        top = candidates[0]
        return {
            "path": top["path"],
            "module_fqn": top["module_fqn"],
            "candidates": [cand["path"] for cand in candidates],
        }

# ==============================================================================
# PIPELINE-LEVEL METRIC HELPERS (REUSED/LOCALIZED)
# ==============================================================================

def _as_set(keys) -> set:
    return set(keys)


def _normalize_str(v) -> str:
    return str(v).strip()


def _is_type_valid(value, type_name: str) -> bool:
    t = (type_name or "str").lower()
    if t in {"str", "string", "path"}:
        return isinstance(value, str)
    if t in {"bool", "boolean"}:
        return isinstance(value, bool)
    if t in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if t in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t in {"dict", "mapping", "object"}:
        return isinstance(value, dict)
    if t in {"list", "array"}:
        return isinstance(value, list)
    return isinstance(value, str)


def key_f1(expected_fields: dict, generated: Optional[dict]) -> float:
    expected_keys = _as_set(expected_fields.keys())
    generated_keys = _as_set((generated or {}).keys())

    if not expected_keys and not generated_keys:
        return 1.0
    if not expected_keys:
        return 0.0 if generated_keys else 1.0
    if not generated_keys:
        return 0.0

    inter = len(expected_keys & generated_keys)
    precision = inter / len(generated_keys) if generated_keys else 0.0
    recall = inter / len(expected_keys) if expected_keys else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def value_accuracy(expected_fields: dict, generated: Optional[dict]) -> float:
    if not expected_fields:
        return 1.0
    if not generated:
        return 0.0

    correct = 0
    for key, expected_value in expected_fields.items():
        if key in generated and _normalize_str(generated[key]) == _normalize_str(expected_value):
            correct += 1

    return correct / len(expected_fields)


def module_accuracy(expected_module: str, generated_module: str) -> float:
    return 1.0 if expected_module == generated_module else 0.0


def _alias_to_canonical(schema: dict) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    aliases = schema.get("aliases", {}) or {}

    for canonical_key, alias_list in aliases.items():
        if isinstance(alias_list, (list, tuple)):
            for alias in alias_list:
                if isinstance(alias, str):
                    alias_map[alias] = canonical_key

    return alias_map


def schema_correctness(schema: dict, generated: Optional[dict]) -> float:
    if generated is None:
        return 0.0

    required = set(schema.get("required", []))
    optional = set(schema.get("optional", []))
    types = schema.get("types", {}) or {}
    choices = schema.get("choices", {}) or {}
    alias_map = _alias_to_canonical(schema)

    allowed = required | optional | set(alias_map.keys())

    if not generated and not allowed:
        return 1.0

    invalid_keys = []
    normalized_generated = {}

    for k, v in generated.items():
        canonical = alias_map.get(k, k)
        normalized_generated[canonical] = v
        if k not in allowed and canonical not in allowed:
            invalid_keys.append(k)

    if invalid_keys:
        return 0.0

    if not normalized_generated:
        return 1.0

    valid_count = 0
    total_count = len(normalized_generated)

    for key, value in normalized_generated.items():
        if key in choices and value not in choices[key]:
            continue
        if not _is_type_valid(value, types.get(key, "str")):
            continue
        valid_count += 1

    return valid_count / total_count if total_count else 1.0


def ansible_aware_metric(
    expected_module: str,
    generated_module: str,
    expected_fields: dict,
    generated: Optional[dict],
) -> float:
    m = module_accuracy(expected_module, generated_module)
    k = key_f1(expected_fields, generated)
    v = value_accuracy(expected_fields, generated)
    return (m + k + v) / 3.0


def average_correctness(aam: float, scm: float) -> float:
    return (aam + scm) / 2.0


def exact_end_to_end_match(
    expected_module: str,
    generated_module: str,
    expected_fields: dict,
    generated: Optional[dict],
) -> bool:
    if generated is None:
        return False
    if expected_module != generated_module:
        return False
    if set(generated.keys()) != set(expected_fields.keys()):
        return False

    for key, expected_value in expected_fields.items():
        if _normalize_str(generated[key]) != _normalize_str(expected_value):
            return False

    return True


# ==============================================================================
# RETRIEVAL UTILS & FORMAT EXTRACTION
# ==============================================================================

def canonical_id(value: object) -> str:
    s = str(value).strip()
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    for suffix in (".json", ".yml", ".yaml"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = s.replace("azure.azcollection.", "")
    return s


def _extract_module_args_from_yaml(parsed_yaml: Any) -> Tuple[Optional[str], Optional[Dict]]:
    if parsed_yaml is None:
        return None, None

    def check_task(task: Any) -> Tuple[Optional[str], Optional[Dict]]:
        if not isinstance(task, dict):
            return None, None
        ignored_keys = {
            "name", "register", "when", "failed_when", "ignore_errors", 
            "become", "become_user", "become_method", "loop", "with_items"
        }
        for k, v in task.items():
            if k not in ignored_keys and isinstance(v, dict):
                return k, v
        return None, None

    if isinstance(parsed_yaml, list) and parsed_yaml:
        for item in parsed_yaml:
            mod, args = check_task(item)
            if mod:
                return mod, args

    if isinstance(parsed_yaml, dict):
        mod, args = check_task(parsed_yaml)
        if mod:
            return mod, args

    return None, None


# ==============================================================================
# CORPUS STATISTICS
# ==============================================================================

def print_corpus_statistics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    collection_counts = collections.Counter()
    task_counts = collections.Counter()
    difficulty_counts = collections.Counter()

    for case in cases:
        collection_counts[case.get("collection", "unknown")] += 1
        task_counts[case.get("task_type", "unknown")] += 1
        difficulty_counts[case.get("difficulty", "unknown")] += 1

    print("\n" + "=" * 60)
    print("                 BENCHMARK CORPUS STATISTICS")
    print("=" * 60)
    print("\nCollections:")
    for col, count in sorted(collection_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col:<40} {count:>5}")

    print("\nTask Types:")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task:<40} {count:>5}")

    print("\nDifficulty:")
    for diff, count in sorted(difficulty_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {diff:<40} {count:>5}")
    print("=" * 60 + "\n")

    return {
        "collections": dict(collection_counts),
        "task_types": dict(task_counts),
        "difficulties": dict(difficulty_counts)
    }


# ==============================================================================
# MAIN HARNESS
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 on Multi-Collection Benchmark.")
    parser.add_argument("--benchmark", default="../Structured-generation/nl2structure/guidance_pipeline/benchmark/benchmark_all.jsonl", help="Path to the JSONL benchmark file.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on count of examples to evaluate.")
    parser.add_argument("--output-dir", default="evaluation", help="Directory where evaluation outputs are written.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug prints.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle benchmark cases before running.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffling seed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_run_time = time.time()

    # Load benchmark cases
    cases = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # Print corpus statistics
    corpus_stats = print_corpus_statistics(cases)

    # Limit and shuffle cases
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(cases)
    if args.limit > 0:
        cases = cases[:args.limit]

    total_samples = len(cases)
    print(f"Loaded {total_samples} samples for evaluation.")

    # Auto-resolve fallbacks for documents and embeddings
    default_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Structured-generation", "nl2structure", "guidance_pipeline", "docs"))
    docs_dir = os.getenv("ANSIBLE_DOCS_DIR")
    if not docs_dir:
        if os.path.isdir(default_docs_dir):
            docs_dir = default_docs_dir
        else:
            docs_dir = "PLACEHOLDER_ANSIBLE_DOCS_DIR"

    default_embeddings_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ansible_embeddings.pkl"))
    embeddings_file = os.getenv("ANSIBLE_EMBEDDINGS_FILE")
    if not embeddings_file:
        if os.path.exists(default_embeddings_file):
            embeddings_file = default_embeddings_file
        else:
            embeddings_file = "PLACEHOLDER_EMBEDDINGS_FILE"

    print(f"ANSIBLE_DOCS_DIR: {docs_dir}")
    print(f"ANSIBLE_EMBEDDINGS_FILE: {embeddings_file}")

    # Instantiate Pluggable Selector
    selector = PipelineHybridSelector(docs_dir=docs_dir, embeddings_file=embeddings_file)

    # Initialize variables for metric tracking
    results = []
    failures = []
    passed_count = 0

    metric_sums = collections.defaultdict(float)
    count_sums = collections.defaultdict(float)
    runtime_sums = collections.defaultdict(float)
    failure_counts = collections.defaultdict(int)

    for idx, sample in enumerate(cases, 1):
        query = sample["query"]
        expected_module = sample["expected_module"]
        expected_fields = sample["expected_fields"]
        doc_file = sample["doc_file"]

        print(f"[{idx}/{total_samples}] Evaluating: {query}")

        # Metrics for this case
        retrieved_doc = None
        candidates = []
        schema = {}
        generated_yaml = ""
        meta = {}
        parsed_yaml = None
        yaml_invalid = True
        generated_module = None
        generated_fields = None

        retrieval_success = False
        top1_success = False
        top5_success = False
        top10_success = False
        m_acc = 0.0
        yaml_val = 0.0
        f_prec = 0.0
        f_rec = 0.0
        f_f1 = 0.0
        v_acc = 0.0
        aam = 0.0
        scm = 0.0
        ac = 0.0
        exact_match = False

        parser_events = 0
        decoder_events = 0
        trigger_events = 0
        switch_count = 0
        parser_transitions = 0
        decoder_transitions = 0
        trigger_transitions = 0
        max_template_depth = 1
        max_indentation = 0
        list_transitions = 0
        dict_transitions = 0
        scalar_transitions = 0
        invalid_indentation_events = 0
        template_switches = 0

        module_mismatch = False
        missing_fields = False
        wrong_value = False
        wrong_type = False
        extra_field = False
        retrieval_wrong = False
        failure_reasons = []

        # 1. Retrieval Phase
        t0 = time.perf_counter()
        try:
            ret_res = selector.retrieve(query)
            retrieved_doc = ret_res["path"]
            candidates = ret_res["candidates"]
        except Exception as exc:
            failure_reasons.append(f"Retrieval crashed: {str(exc)}")
            retrieved_doc = None
            candidates = []
        t1 = time.perf_counter()
        retrieval_ms = (t1 - t0) * 1000

        # Check retrieval rankings
        expected_doc_canonical = canonical_id(doc_file)
        if retrieved_doc:
            retrieved_doc_canonical = canonical_id(retrieved_doc)
            retrieval_success = (retrieved_doc_canonical == expected_doc_canonical)
            top1_success = retrieval_success
            top5_success = any(canonical_id(c) == expected_doc_canonical for c in candidates[:5])
            top10_success = any(canonical_id(c) == expected_doc_canonical for c in candidates[:10])
        else:
            retrieval_wrong = True

        if not retrieval_success:
            retrieval_wrong = True
            failure_reasons.append("Retrieval wrong")

        # 2. Schema Extraction Phase
        t2 = time.perf_counter()
        if retrieved_doc and os.path.exists(retrieved_doc):
            try:
                schema = extract_schema(retrieved_doc)
            except Exception as exc:
                failure_reasons.append(f"Schema extraction crashed: {str(exc)}")
        t3 = time.perf_counter()
        schema_ms = (t3 - t2) * 1000

        # 3. Generation / Decoding Phase
        t4 = time.perf_counter()
        if retrieved_doc and schema:
            try:
                with open(retrieved_doc, "r", encoding="utf-8") as f:
                    rdoc = json.load(f)
                module_fqn = f"{rdoc.get('collection', '')}.{rdoc.get('module', '')}".strip(".")

                generated_yaml, meta = generate_parser_state_yaml(
                    query=query,
                    schema=schema,
                    module_fqn=module_fqn,
                    max_tokens=256,
                    debug=args.debug,
                    return_metadata=True
                )
            except Exception as exc:
                failure_reasons.append(f"Decoder crashed: {str(exc)}")
        t5 = time.perf_counter()
        generation_ms = (t5 - t4) * 1000

        # 4. YAML Parse Phase
        t6 = time.perf_counter()
        if generated_yaml:
            try:
                parsed_yaml = yaml.safe_load(generated_yaml)
                yaml_invalid = (parsed_yaml is None)
            except Exception as exc:
                yaml_invalid = True
                failure_reasons.append(f"YAML parse error: {str(exc)}")
        t7 = time.perf_counter()
        yaml_parse_ms = (t7 - t6) * 1000

        if yaml_invalid:
            if "YAML parse error" not in "".join(failure_reasons):
                failure_reasons.append("YAML invalid")

        # 5. Evaluate Metrics Phase
        t8 = time.perf_counter()
        if parsed_yaml is not None:
            yaml_val = 1.0
            generated_module, generated_fields = _extract_module_args_from_yaml(parsed_yaml)

            m_acc = module_accuracy(expected_module, generated_module)
            if m_acc == 0.0:
                module_mismatch = True
                failure_reasons.append(f"Module mismatch: expected {expected_module}, got {generated_module}")

            # Fields evaluation
            alias_map = _alias_to_canonical(schema)
            normalized_generated = {alias_map.get(k, k): v for k, v in (generated_fields or {}).items()}
            expected_keys = set(expected_fields.keys())
            generated_keys = set(normalized_generated.keys())

            missing_keys = expected_keys - generated_keys
            extra_keys = generated_keys - expected_keys

            if missing_keys:
                missing_fields = True
                failure_reasons.append(f"Missing fields: {sorted(missing_keys)}")
            if extra_keys:
                extra_field = True
                failure_reasons.append(f"Extra fields: {sorted(extra_keys)}")

            for k in expected_keys & generated_keys:
                if _normalize_str(normalized_generated[k]) != _normalize_str(expected_fields[k]):
                    wrong_value = True
                    failure_reasons.append(f"Wrong value for {k}: expected {expected_fields[k]}, got {normalized_generated[k]}")

            # Check type validity
            types = schema.get("types", {}) or {}
            for k, v in normalized_generated.items():
                if k in types:
                    if not _is_type_valid(v, types[k]):
                        wrong_type = True
                        failure_reasons.append(f"Wrong type for {k}: expected {types[k]}, got {type(v).__name__}")

            # Baseline metrics
            f_prec = len(expected_keys & generated_keys) / len(generated_keys) if generated_keys else 0.0
            f_rec = len(expected_keys & generated_keys) / len(expected_keys) if expected_keys else 0.0
            if f_prec > 0.0 or f_rec > 0.0:
                f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec)
            else:
                f_f1 = 1.0 if (not expected_keys and not generated_keys) else 0.0

            v_acc = value_accuracy(expected_fields, generated_fields)
            aam = ansible_aware_metric(expected_module, generated_module or "", expected_fields, generated_fields)
            scm = schema_correctness(schema, generated_fields)
            ac = average_correctness(aam, scm)
            exact_match = exact_end_to_end_match(expected_module, generated_module or "", expected_fields, generated_fields)

        # 6. Extract Parser Log Metrics
        parser_log = meta.get("parser_log", []) or []
        decoder_log = meta.get("decoder_log", []) or []
        trigger_log = meta.get("trigger_log", []) or []
        transition_log = meta.get("transition_log", []) or []
        template_stack = meta.get("template_stack", []) or []
        current_indent = meta.get("current_indent", 0) or 0
        switch_count = meta.get("switch_count", 0) or 0

        parser_events = len(parser_log)
        decoder_events = len(decoder_log)
        trigger_events = len(trigger_log)

        parser_transitions = len([x for x in parser_log if any(x.startswith(p) for p in ("push:", "pop:", "switch:"))])
        decoder_transitions = decoder_events
        trigger_transitions = len([x for x in trigger_log if x.startswith("fire:")])

        # Compute max stack depth
        depth = 1
        for log in parser_log:
            if log.startswith("push:"):
                depth += 1
                max_template_depth = max(max_template_depth, depth)
            elif log.startswith("pop:"):
                depth = max(1, depth - 1)

        # Compute max indentation
        if generated_yaml:
            for line in generated_yaml.splitlines():
                if line.strip():
                    indent = len(line) - len(line.lstrip(" "))
                    max_indentation = max(max_indentation, indent)

        list_transitions = len([x for x in parser_log if ":list" in x])
        dict_transitions = len([x for x in parser_log if ":dict" in x])
        scalar_transitions = len([x for x in parser_log if ":scalar" in x])
        invalid_indentation_events = len([x for x in trigger_log if "invalid_indentation" in x])
        template_switches = len([x for x in parser_log if x.startswith("switch:")])

        t9 = time.perf_counter()
        evaluation_ms = (t9 - t8) * 1000

        # Update metrics sums
        metric_sums["retrieval_success"] += float(retrieval_success)
        metric_sums["top1_success"] += float(top1_success)
        metric_sums["top5_success"] += float(top5_success)
        metric_sums["top10_success"] += float(top10_success)
        metric_sums["module_accuracy"] += m_acc
        metric_sums["yaml_validity"] += yaml_val
        metric_sums["field_precision"] += f_prec
        metric_sums["field_recall"] += f_rec
        metric_sums["field_f1"] += f_f1
        metric_sums["value_accuracy"] += v_acc
        metric_sums["aam"] += aam
        metric_sums["scm"] += scm
        metric_sums["ac"] += ac
        metric_sums["exact_match"] += float(exact_match)

        count_sums["parser_events"] += parser_events
        count_sums["decoder_events"] += decoder_events
        count_sums["trigger_events"] += trigger_events
        count_sums["switch_count"] += switch_count
        count_sums["parser_transitions"] += parser_transitions
        count_sums["decoder_transitions"] += decoder_transitions
        count_sums["trigger_transitions"] += trigger_transitions
        count_sums["max_template_depth"] += max_template_depth
        count_sums["max_indentation"] += max_indentation
        count_sums["list_transitions"] += list_transitions
        count_sums["dict_transitions"] += dict_transitions
        count_sums["scalar_transitions"] += scalar_transitions
        count_sums["invalid_indentation_events"] += invalid_indentation_events
        count_sums["template_switches"] += template_switches

        runtime_sums["retrieval"] += retrieval_ms
        runtime_sums["schema"] += schema_ms
        runtime_sums["generation"] += generation_ms
        runtime_sums["yaml_parse"] += yaml_parse_ms
        runtime_sums["evaluation"] += evaluation_ms

        if exact_match:
            passed_count += 1
        else:
            # failure classifications
            if module_mismatch:
                failure_counts["Module mismatch"] += 1
            if missing_fields:
                failure_counts["Missing fields"] += 1
            if wrong_value:
                failure_counts["Wrong value"] += 1
            if wrong_type:
                failure_counts["Wrong type"] += 1
            if extra_field:
                failure_counts["Extra field"] += 1
            if yaml_invalid:
                failure_counts["YAML invalid"] += 1
            if retrieval_wrong:
                failure_counts["Retrieval wrong"] += 1

        # Record detail
        record = {
            "query": query,
            "expected_module": expected_module,
            "generated_module": generated_module,
            "expected_fields": expected_fields,
            "generated_fields": generated_fields,
            "generated_yaml": generated_yaml,
            "retrieval_success": retrieval_success,
            "top1_success": top1_success,
            "top5_success": top5_success,
            "top10_success": top10_success,
            "module_accuracy": m_acc,
            "yaml_validity": yaml_val,
            "field_precision": f_prec,
            "field_recall": f_rec,
            "field_f1": f_f1,
            "value_accuracy": v_acc,
            "aam": aam,
            "scm": scm,
            "ac": ac,
            "exact_match": exact_match,
            "parser_events": parser_events,
            "decoder_events": decoder_events,
            "trigger_events": trigger_events,
            "switch_count": switch_count,
            "parser_transitions": parser_transitions,
            "decoder_transitions": decoder_transitions,
            "trigger_transitions": trigger_transitions,
            "max_template_depth": max_template_depth,
            "max_indentation": max_indentation,
            "list_transitions": list_transitions,
            "dict_transitions": dict_transitions,
            "scalar_transitions": scalar_transitions,
            "invalid_indentation_events": invalid_indentation_events,
            "template_switches": template_switches,
            "retrieval_time_ms": retrieval_ms,
            "schema_time_ms": schema_ms,
            "generation_time_ms": generation_ms,
            "yaml_parse_time_ms": yaml_parse_ms,
            "evaluation_time_ms": evaluation_ms,
            "module_mismatch": module_mismatch,
            "missing_fields": missing_fields,
            "wrong_value": wrong_value,
            "wrong_type": wrong_type,
            "extra_field": extra_field,
            "yaml_invalid": yaml_invalid,
            "retrieval_wrong": retrieval_wrong,
            "failure_reasons": failure_reasons,
            "meta": meta,
        }
        results.append(record)

        # Write to failures list if failed
        if not exact_match:
            failures.append({
                "query": query,
                "document": retrieved_doc or "",
                "expected_module": expected_module,
                "generated_module": generated_module or "",
                "expected_fields": expected_fields,
                "generated_fields": generated_fields or {},
                "generated_yaml": generated_yaml,
                "parser_log": parser_log,
                "decoder_log": decoder_log,
                "trigger_log": trigger_log,
                "transition_log": transition_log,
                "failure_reason": "; ".join(failure_reasons)
            })

    # Total script runtime
    total_run_time = time.time() - start_run_time

    # ==============================================================================
    # WRITE OUTPUT FILES
    # ==============================================================================

    # 1. results.jsonl
    results_jsonl_path = os.path.join(args.output_dir, "results.jsonl")
    with open(results_jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. failures.jsonl
    failures_jsonl_path = os.path.join(args.output_dir, "failures.jsonl")
    with open(failures_jsonl_path, "w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")

    # 3. results.csv
    results_csv_path = os.path.join(args.output_dir, "results.csv")
    csv_headers = [
        "query", "expected_module", "generated_module", "retrieval_success",
        "top1_success", "top5_success", "top10_success", "module_accuracy",
        "yaml_validity", "field_precision", "field_recall", "field_f1",
        "value_accuracy", "aam", "scm", "ac", "exact_match", "parser_events",
        "decoder_events", "trigger_events", "switch_count", "parser_transitions",
        "decoder_transitions", "trigger_transitions", "max_template_depth",
        "max_indentation", "list_transitions", "dict_transitions", "scalar_transitions",
        "invalid_indentation_events", "template_switches", "retrieval_time_ms",
        "schema_time_ms", "generation_time_ms", "yaml_parse_time_ms", "evaluation_time_ms",
        "module_mismatch", "missing_fields", "wrong_value", "wrong_type", "extra_field",
        "yaml_invalid", "retrieval_wrong", "failure_reasons"
    ]
    with open(results_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for r in results:
            row = {h: r[h] for h in csv_headers[:-1]}
            row["failure_reasons"] = "; ".join(r["failure_reasons"])
            writer.writerow(row)

    # 4. summary.json
    summary_path = os.path.join(args.output_dir, "summary.json")
    divisor = total_samples if total_samples > 0 else 1

    failure_percents = {}
    for cat in ["Module mismatch", "Missing fields", "Wrong value", "Wrong type", "Extra field", "YAML invalid", "Retrieval wrong"]:
        count = failure_counts[cat]
        percent = (count / divisor) if total_samples > 0 else 0.0
        failure_percents[cat] = {
            "count": count,
            "percentage": percent
        }

    summary = {
        "Total samples": total_samples,
        "Passed": passed_count,
        "Failed": total_samples - passed_count,
        "Retrieval accuracy": metric_sums["retrieval_success"] / divisor,
        "Top-1 accuracy": metric_sums["top1_success"] / divisor,
        "Top-5 accuracy": metric_sums["top5_success"] / divisor,
        "Top-10 accuracy": metric_sums["top10_success"] / divisor,
        "Module accuracy": metric_sums["module_accuracy"] / divisor,
        "YAML validity": metric_sums["yaml_validity"] / divisor,
        "Field Precision": metric_sums["field_precision"] / divisor,
        "Field Recall": metric_sums["field_recall"] / divisor,
        "Field F1": metric_sums["field_f1"] / divisor,
        "Value Accuracy": metric_sums["value_accuracy"] / divisor,
        "AAM": metric_sums["aam"] / divisor,
        "SCM": metric_sums["scm"] / divisor,
        "AC": metric_sums["ac"] / divisor,
        "Exact Match": metric_sums["exact_match"] / divisor,
        "Average parser events": count_sums["parser_events"] / divisor,
        "Average decoder events": count_sums["decoder_events"] / divisor,
        "Average trigger events": count_sums["trigger_events"] / divisor,
        "Average switch count": count_sums["switch_count"] / divisor,
        "Average parser transitions": count_sums["parser_transitions"] / divisor,
        "Average decoder transitions": count_sums["decoder_transitions"] / divisor,
        "Average trigger transitions": count_sums["trigger_transitions"] / divisor,
        "Average max template depth": count_sums["max_template_depth"] / divisor,
        "Average max indentation": count_sums["max_indentation"] / divisor,
        "Average list transitions": count_sums["list_transitions"] / divisor,
        "Average dict transitions": count_sums["dict_transitions"] / divisor,
        "Average scalar transitions": count_sums["scalar_transitions"] / divisor,
        "Average invalid indentation events": count_sums["invalid_indentation_events"] / divisor,
        "Average template switches": count_sums["template_switches"] / divisor,
        "Average retrieval ms": runtime_sums["retrieval"] / divisor,
        "Average schema extraction ms": runtime_sums["schema"] / divisor,
        "Average generation ms": runtime_sums["generation"] / divisor,
        "Average YAML parse ms": runtime_sums["yaml_parse"] / divisor,
        "Average evaluation ms": runtime_sums["evaluation"] / divisor,
        "Total runtime sec": total_run_time,
        "Failure Category Counts": failure_percents,
        "Corpus Statistics": corpus_stats
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("                      EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Total Samples: {total_samples}")
    print(f"Passed:        {passed_count}")
    print(f"Failed:        {total_samples - passed_count}")
    print(f"Exact Match:   {summary['Exact Match']:.4f}")
    print(f"YAML Validity: {summary['YAML validity']:.4f}")
    print(f"AAM:           {summary['AAM']:.4f}")
    print(f"SCM:           {summary['SCM']:.4f}")
    print(f"AC:            {summary['AC']:.4f}")
    print(f"Runtime:       {total_run_time:.2f} seconds")
    print(f"Outputs written to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()