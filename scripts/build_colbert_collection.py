#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def collect_text(obj, parts, prefix=""):
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"description", "short_description", "long_description", "options", "suboptions", "notes", "examples", "return", "returns", "requirements", "seealso", "author", "version_added"}:
                collect_text(v, parts, f"{prefix}{k}")
            else:
                collect_text(v, parts, f"{prefix}{k}")
    elif isinstance(obj, list):
        for x in obj:
            collect_text(x, parts, prefix)
    elif isinstance(obj, (str, int, float, bool)):
        s = str(obj).strip()
        if s:
            if prefix:
                parts.append(f"{prefix}: {s}")
            else:
                parts.append(s)

def flatten_doc(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = [f"module: {path.stem}"]
    collect_text(data, parts)
    seen = set()
    deduped = []
    for p in parts:
        p = " ".join(str(p).split())
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    text = "\n".join(deduped)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-txt", required=True)
    ap.add_argument("--docs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-out", required=True)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    files_txt = Path(args.files_txt)
    docs_root = Path(args.docs_root)
    out = Path(args.out)
    map_out = Path(args.map_out)

    out.parent.mkdir(parents=True, exist_ok=True)
    map_out.parent.mkdir(parents=True, exist_ok=True)

    passage_count = 0
    doc_map = []

    with out.open("w", encoding="utf-8") as f:
        for line in files_txt.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name or not name.endswith(".json"):
                continue

            path = docs_root / name
            if not path.exists():
                continue

            try:
                text = flatten_doc(path)
            except Exception:
                continue

            if not text:
                continue

            module = path.stem
            chunks = [text[i:i + args.max_chars] for i in range(0, len(text), args.max_chars)] or [text]
            for i, chunk in enumerate(chunks):
                pid = module if len(chunks) == 1 else f"{module}::chunk{i}"
                chunk = " ".join(chunk.split())
                if not chunk:
                    continue
                f.write(f"{pid}\t{chunk}\n")
                doc_map.append({"pid": pid, "module": module, "source": str(path)})
                passage_count += 1

    map_out.write_text(json.dumps(doc_map, indent=2), encoding="utf-8")
    print(f"Wrote {passage_count} passages to {out}")

if __name__ == "__main__":
    main()
