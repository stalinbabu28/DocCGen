from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.dynamic_decoder import generate_dynamic_yaml

queries = [
    "create a virtual network called prod-vnet in resource group prod-rg",
    "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
    "create a dns zone named example.com in resource group prod-rg",
    "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8",
]

for q in queries:
    print("=" * 80)
    print(q)

    doc_path = select_document(q)
    schema = extract_schema(doc_path)
    module_fqn = f"{schema['module']}"

    out = generate_dynamic_yaml(
        query=q,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=256,
    )

    print("\nDOCUMENT:")
    print(doc_path)
    print("\nOUTPUT:")
    print(out)