from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict


METRIC_PATTERNS = {
    "Module Accuracy": r"Module Accuracy:\s*([0-9.]+)",
    "YAML Validity": r"YAML Validity:\s*([0-9.]+)",
    "AAM": r"AAM:\s*([0-9.]+)",
    "SCM": r"SCM:\s*([0-9.]+)",
    "AC": r"AC:\s*([0-9.]+)",
    "Exact Match Rate": r"Exact End-to-End Match Rate:\s*([0-9.]+)",
    "Exact Matches": r"Exact End-to-End Matches:\s*([0-9]+)/50",
}


def _run_benchmark(use_dynamic: str) -> tuple[Dict[str, float], str]:
    env = os.environ.copy()
    env["USE_DYNAMIC_DECODER"] = use_dynamic

    proc = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate_end_to_end_real"],
        env=env,
        capture_output=True,
        text=True,
    )

    out = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        print("=" * 100)
        print(f"Benchmark failed for USE_DYNAMIC_DECODER={use_dynamic}")
        print(out[-4000:])
        raise SystemExit(proc.returncode)

    metrics: Dict[str, float] = {}
    for name, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, out)
        if not m:
            raise RuntimeError(f"Could not find metric '{name}' in benchmark output")
        metrics[name] = float(m.group(1))

    return metrics, out


def main() -> None:
    static_metrics, _ = _run_benchmark("0")
    dynamic_metrics, _ = _run_benchmark("1")

    print("\n" + "=" * 100)
    print("STATIC VS DYNAMIC")
    print("=" * 100)

    print(f"{'Metric':<28} {'Static':>10} {'Dynamic':>10} {'Delta':>10}")
    print("-" * 60)

    for key in [
        "Module Accuracy",
        "YAML Validity",
        "AAM",
        "SCM",
        "AC",
        "Exact Match Rate",
    ]:
        s = static_metrics[key]
        d = dynamic_metrics[key]
        print(f"{key:<28} {s:>10.4f} {d:>10.4f} {d - s:>10.4f}")

    print("\nStatic Exact Matches:", int(static_metrics["Exact Matches"]))
    print("Dynamic Exact Matches:", int(dynamic_metrics["Exact Matches"]))


if __name__ == "__main__":
    main()