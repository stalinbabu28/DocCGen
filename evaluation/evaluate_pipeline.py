import json

from llama_cpp import Llama

from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from evaluation.extract_json import extract_json


MODEL_PATH = (
    "/home/dao-lab/.lmstudio/models/"
    "unsloth/gpt-oss-20b-GGUF/"
    "gpt-oss-20b-F16.gguf"
)


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=20,
    verbose=False
)


TESTS = [

    {
        "query":
        "create subnet backend-subnet in virtual network prod-vnet resource group prod-rg",

        "expected":
        {
            "name": "backend-subnet",
            "virtual_network_name": "prod-vnet",
            "resource_group": "prod-rg"
        }
    },

    {
        "query":
        "create storage account mystorage in resource group prod-rg",

        "expected":
        {
            "name": "mystorage",
            "resource_group": "prod-rg"
        }
    },

    {
        "query":
        "list storage accounts in resource group prod-rg",

        "expected":
        {
            "resource_group": "prod-rg"
        }
    },

    {
        "query":
        "show details of virtual network prod-vnet",

        "expected":
        {
            "name": "prod-vnet"
        }
    },

    {
        "query":
        "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",

        "expected":
        {
            "name": "backend-subnet",
            "virtual_network_name": "prod-vnet",
            "resource_group": "prod-rg"
        }
    }

]


correct = 0
total = 0

for test in TESTS:

    query = test["query"]

    print()
    print("=" * 80)
    print(query)

    doc_path = select_document(query)

    schema = extract_schema(doc_path)

    print()
    print("DOC:")
    print(doc_path)

    print()
    print("MODULE:")
    print(schema["module"])

    print()
    print("REQUIRED:")
    print(schema["required"])

    prompt = f"""
You extract Azure module parameters.

Return ONLY a JSON object.

Rules:

1. Extract values only from the user request.
2. Do not explain.
3. Do not describe parameters.
4. Do not return schema information.
5. Do not return commands.
6. If a value is not present, omit the field.
7. Return JSON only.

User request:
{query}

Available parameters:
{schema["required"] + schema["optional"]}
"""

    response = llm(
        prompt,
        max_tokens=150,
        temperature=0,
        stop=[
            "<|end|>",
            "<|start|>"
        ]
    )

    output = (
        response["choices"][0]["text"]
        .strip()
    )

    print()
    print("RAW:")
    print(output)

    try:

        parsed = extract_json(output)

        if parsed is None:
            print("INVALID JSON")
            continue

    except Exception:

        print("INVALID JSON")
        total += 1
        continue

    expected = test["expected"]

    match = True

    for k, v in expected.items():

        if parsed.get(k) != v:
            match = False

    if match:

        print("PASS")
        correct += 1

    else:

        print("FAIL")
        print("EXPECTED:", expected)
        print("GOT:", parsed)

    total += 1

print()
print("=" * 80)
print("Accuracy:", round(correct / total * 100, 2), "%")