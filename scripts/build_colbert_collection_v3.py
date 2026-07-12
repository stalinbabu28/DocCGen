#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def norm_ws(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def as_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = norm_ws(item)
            if s:
                out.append(s)
        return out
    s = norm_ws(value)
    return [s] if s else []


def first_text(value: Any, limit: int = 220) -> str:
    lines = as_lines(value)
    if not lines:
        return ""
    text = " ".join(lines)
    text = norm_ws(text)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0].strip() + "..."
    return text


def join_list(value: Any, limit: int = 20) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        items = [norm_ws(x) for x in value if norm_ws(x)]
    else:
        items = [norm_ws(value)] if norm_ws(value) else []
    if not items:
        return ""
    return ", ".join(items[:limit])


def safe_bool(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return norm_ws(v)


def option_summary(name: str, spec: Dict[str, Any]) -> str:
    parts: List[str] = [f"option {name}"]
    t = spec.get("type")
    if t:
        parts.append(f"type={norm_ws(t)}")
    if "required" in spec:
        parts.append(f"required={safe_bool(spec.get('required'))}")
    if "default" in spec:
        parts.append(f"default={safe_bool(spec.get('default'))}")
    aliases = join_list(spec.get("aliases"))
    if aliases:
        parts.append(f"aliases={aliases}")
    choices = join_list(spec.get("choices"))
    if choices:
        parts.append(f"choices={choices}")
    elements = spec.get("elements")
    if elements:
        parts.append(f"elements={norm_ws(elements)}")
    desc = first_text(spec.get("description"), limit=260)
    if desc:
        parts.append(f"desc={desc}")
    return " | ".join(parts)


def render_option_tree(name: str, spec: Dict[str, Any]) -> List[str]:
    lines = [option_summary(name, spec)]

    suboptions = spec.get("suboptions")
    if isinstance(suboptions, dict):
        for child_name in sorted(suboptions.keys()):
            child_spec = suboptions[child_name]
            if isinstance(child_spec, dict):
                lines.extend(render_option_tree(f"{name}.{child_name}", child_spec))

    return lines


def render_doc(path: Path) -> Tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))

    collection = norm_ws(data.get("collection"))
    module = norm_ws(data.get("module")) or path.stem
    module_fqn = f"{collection}.{module}" if collection and not module.startswith(collection + ".") else (collection or module)

    parts: List[str] = []
    parts.append(f"module: {module_fqn}")

    short_desc = first_text(data.get("short_description"), limit=250)
    if short_desc:
        parts.append(f"short_description: {short_desc}")

    for desc in as_lines(data.get("description"))[:4]:
        parts.append(f"description: {desc}")

    reqs = as_lines(data.get("requirements"))
    if reqs:
        parts.append(f"requirements: {', '.join(reqs)}")

    notes = as_lines(data.get("notes"))
    for note in notes[:2]:
        parts.append(f"note: {note}")

    options = data.get("options")
    if isinstance(options, dict):
        for opt_name in sorted(options.keys()):
            opt_spec = options[opt_name]
            if isinstance(opt_spec, dict):
                parts.extend(render_option_tree(opt_name, opt_spec))
            else:
                txt = norm_ws(opt_spec)
                if txt:
                    parts.append(f"option {opt_name} | desc={txt}")

    examples = data.get("examples")
    if examples:
        ex_lines = as_lines(examples)
        for ex in ex_lines[:2]:
            parts.append(f"example: {ex}")

    returns = data.get("return") or data.get("returns")
    if isinstance(returns, dict):
        for ret_name in sorted(returns.keys())[:8]:
            ret_spec = returns[ret_name]
            if isinstance(ret_spec, dict):
                ret_desc = first_text(ret_spec.get("description"), limit=220)
                ret_type = norm_ws(ret_spec.get("type"))
                line = f"return {ret_name}"
                if ret_type:
                    line += f" | type={ret_type}"
                if ret_desc:
                    line += f" | desc={ret_desc}"
                parts.append(line)

    text = "\n".join(parts).strip()
    return module_fqn, text


def chunk_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines()
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        add_len = len(line) + 1
        if cur and cur_len + add_len > max_chars:
            chunk = "\n".join(cur).strip()
            if chunk:
                chunks.append(chunk)
            cur = [line]
            cur_len = len(line) + 1
        else:
            cur.append(line)
            cur_len += add_len

    if cur:
        chunk = "\n".join(cur).strip()
        if chunk:
            chunks.append(chunk)

    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = min(start + max_chars, len(chunk))
                if end < len(chunk):
                    cut = chunk.rfind("\n", start, end)
                    if cut > start + 2000:
                        end = cut
                piece = chunk[start:end].strip()
                if piece:
                    final_chunks.append(piece)
                start = end
    return final_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-txt", required=True)
    ap.add_argument("--docs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-out", required=True)
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    files_txt = Path(args.files_txt)
    docs_root = Path(args.docs_root)
    out_path = Path(args.out)
    map_path = Path(args.map_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    row_id = 1  # IMPORTANT: ColBERT expects 1-based line numbers for data rows.

    lines = files_txt.read_text(encoding="utf-8").splitlines()
    for line in lines:
        name = line.strip()
        if not name or not name.endswith(".json"):
            continue

        doc_path = docs_root / name
        if not doc_path.exists():
            continue

        try:
            module_fqn, passage = render_doc(doc_path)
        except Exception:
            continue

        if not passage:
            continue

        passages = chunk_text(passage, args.max_chars)
        for chunk_idx, chunk in enumerate(passages):
            chunk = chunk.strip()
            if not chunk:
                continue

            rows.append(
                {
                    "pid": row_id,
                    "module": module_fqn,
                    "source": str(doc_path),
                    "chunk": chunk_idx,
                    "text": chunk,
                }
            )
            row_id += 1

    with out_path.open("w", encoding="utf-8") as f:
        f.write("id\ttext\n")
        for r in rows:
            text = r["text"].replace("\t", " ").replace("\r", " ").replace("\n", " ")
            f.write(f"{r['pid']}\t{text}\n")

    map_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} passages to {out_path}")
    print(f"Wrote map to {map_path}")


if __name__ == "__main__":
    main()