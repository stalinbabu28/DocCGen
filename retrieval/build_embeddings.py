import os
import json
import pickle

from sentence_transformers import SentenceTransformer

DOC_DIR = "/home/dao-lab/stalin/azure_docs"

model = SentenceTransformer("all-MiniLM-L6-v2")

database = []

files = [
    f for f in os.listdir(DOC_DIR)
    if f.endswith(".json")
]

print(f"Processing {len(files)} docs...")

for file in files:

    path = os.path.join(DOC_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    module = data.get("module", "")

    short_desc = data.get(
        "short_description",
        ""
    )

    description = " ".join(
        data.get("description", [])
    )

    required_params = []

    for name, meta in data.get(
        "options",
        {}
    ).items():

        if (
            isinstance(meta, dict)
            and meta.get("required", False)
        ):
            required_params.append(name)

    module_keywords = (
        module
        .replace("azure_rm_", "")
        .replace("_", " ")
    )

    text = f"""
Module:
{module}

Keywords:
{module_keywords}
{module_keywords}
{module_keywords}

Short Description:
{short_desc}

Description:
{description}

Required Parameters:
{' '.join(required_params)}
"""

    embedding = model.encode(text)

    database.append({
        "path": path,
        "module": module,
        "text": text,
        "embedding": embedding
    })

with open(
    "azure_embeddings.pkl",
    "wb"
) as f:

    pickle.dump(database, f)

print("Saved embeddings.")