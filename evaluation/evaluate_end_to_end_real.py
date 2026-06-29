from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Set

from pipeline.full_doccgen import run_pipeline
from evaluation.test_cases_50 import TESTS


def _as_set(keys: Iterable[str]) -> Set[str]:
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


def key_f1(expected_fields: Dict, generated: Optional[Dict]) -> float:
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


def value_accuracy(expected_fields: Dict, generated: Optional[Dict]) -> float:
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


def _alias_to_canonical(schema: Dict) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    aliases = schema.get("aliases", {}) or {}

    for canonical_key, alias_list in aliases.items():
        if isinstance(alias_list, (list, tuple)):
            for alias in alias_list:
                if isinstance(alias, str):
                    alias_map[alias] = canonical_key

    return alias_map


def schema_correctness(schema: Dict, generated: Optional[Dict]) -> float:
    """
    Schema validity:
    - generated keys must be allowed
    - generated values must satisfy types and choices where applicable
    - aliases are accepted as equivalent to their canonical field
    """
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
    normalized_generated: Dict[str, Any] = {}

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
    expected_fields: Dict,
    generated: Optional[Dict],
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
    expected_fields: Dict,
    generated: Optional[Dict],
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


def main():
    total = len(TESTS)

    os.makedirs("evaluation/generated_yaml", exist_ok=True)
    os.makedirs("evaluation/generated_meta", exist_ok=True)

    module_scores = []
    aam_scores = []
    scm_scores = []
    ac_scores = []
    exact_matches = 0
    yaml_valid = 0

    for idx, sample in enumerate(TESTS):
        query = sample["query"]
        expected_module = sample["expected_module"]
        expected_fields = sample["expected_fields"]

        print("\n" + "=" * 80)
        print(query)

        result = run_pipeline(query)

        print("\nSELECTED DOC:")
        print(result["document"])
        print("MODULE:", result["module_name"])

        raw = result["raw"]
        generated = result["params"]
        schema = result["schema"]

        if result["yaml_parsed"] is not None:
            yaml_valid += 1

        m = module_accuracy(expected_module, result["module_name"])
        aam = ansible_aware_metric(
            expected_module,
            result["module_name"],
            expected_fields,
            generated,
        )
        scm = schema_correctness(schema, generated)
        ac = average_correctness(aam, scm)
        exact = exact_end_to_end_match(
            expected_module,
            result["module_name"],
            expected_fields,
            generated,
        )

        module_scores.append(m)
        aam_scores.append(aam)
        scm_scores.append(scm)
        ac_scores.append(ac)

        if exact:
            exact_matches += 1

        artifact = {
            "query": query,
            "expected_module": expected_module,
            "expected_fields": expected_fields,
            "generated_module": result["module_name"],
            "document": result["document"],
            "raw": raw,
            "params": generated,
            "yaml": result["yaml"],
            "yaml_parsed": result["yaml_parsed"],
            "module_accuracy": m,
            "aam": aam,
            "scm": scm,
            "ac": ac,
            "exact_match": exact,
        }

        with open(f"evaluation/generated_meta/test_{idx + 1}.json", "w", encoding="utf-8") as f:
            import json
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        with open(f"evaluation/generated_yaml/test_{idx + 1}.yml", "w", encoding="utf-8") as f:
            f.write(result["yaml"])

        print("\nRAW:")
        print(raw)

        print("\nPARSED:")
        print(generated)

        print("\nYAML:")
        print(result["yaml"])

        print("\nSCORES:")
        print(f"Module Accuracy: {m:.4f}")
        print(f"AAM: {aam:.4f}")
        print(f"SCM: {scm:.4f}")
        print(f"AC: {ac:.4f}")
        print(f"Exact Match: {1 if exact else 0}")

    module_accuracy_avg = sum(module_scores) / total if total else 0.0
    aam_avg = sum(aam_scores) / total if aam_scores else 0.0
    scm_avg = sum(scm_scores) / total if scm_scores else 0.0
    ac_avg = sum(ac_scores) / total if ac_scores else 0.0
    yaml_validity = yaml_valid / total if total else 0.0
    exact_match_rate = exact_matches / total if total else 0.0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Module Accuracy: {module_accuracy_avg:.4f}")
    print(f"YAML Validity: {yaml_validity:.4f}")
    print(f"AAM: {aam_avg:.4f}")
    print(f"SCM: {scm_avg:.4f}")
    print(f"AC: {ac_avg:.4f}")
    print(f"Exact End-to-End Match Rate: {exact_match_rate:.4f}")
    print(f"Exact End-to-End Matches: {exact_matches}/{total}")


if __name__ == "__main__":
    main()