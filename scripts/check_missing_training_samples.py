from pathlib import Path
import json
import os

BENCHMARK = Path("/home/dao-lab/stalin/phase3_trigger_engine/benchmark_all_repaired.jsonl")
DOCMAP = Path("/home/dao-lab/stalin/phase3withcolbert/colbert_data/data/doc_map.json")


def norm(p):
    return os.path.normpath(os.path.abspath(p))


with open(DOCMAP) as f:
    docmap = json.load(f)

by_module = {d["module"]: d for d in docmap}
by_source = {norm(d["source"]): d for d in docmap}

total = 0
missing = []

with open(BENCHMARK) as f:
    for line in f:
        total += 1
        sample = json.loads(line)

        expected = sample["expected_module"]
        document = sample.get("document", "")

        found = False
        reason = ""

        if document:
            if norm(document) in by_source:
                found = True
            else:
                reason = "document path not found"

        if not found:
            if expected in by_module:
                found = True
                reason = "found by module only"

        if not found:
            reason = "module not found"

        if not found:
            missing.append({
                "query": sample["query"],
                "expected_module": expected,
                "document": document,
                "reason": reason,
            })

print(f"Benchmark samples : {total}")
print(f"Missing samples   : {len(missing)}")

for m in missing:
    print("=" * 80)
    print("Query:", m["query"])
    print("Module:", m["expected_module"])
    print("Reason:", m["reason"])
    print("Document:", m["document"])