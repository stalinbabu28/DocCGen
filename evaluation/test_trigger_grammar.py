from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.field_policy import select_fields
from pipeline.yaml_grammar_builder import build_yaml_grammar
from pipeline.extract_params_yaml import generate_constrained_yaml

queries = [
    "create a virtual network called prod-vnet in resource group prod-rg",
    "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
    "create a dns zone named example.com in resource group prod-rg",
]

for q in queries:
    print("=" * 80)
    print(q)

    doc_path = select_document(q)
    schema = extract_schema(doc_path)
    fields = select_fields(q, schema)

    print("\nDOCUMENT:")
    print(doc_path)

    print("\nFIELDS:")
    print(fields)

    grammar = build_yaml_grammar(
        module_fqn=schema["module"],
        schema=schema,
        include_fields=fields,
    )

    print("\nGRAMMAR:")
    print(grammar)

    print("\nYAML OUTPUT:")
    yaml_text = generate_constrained_yaml(
        query=q,
        schema=schema,
        module_fqn=schema["module"],
        max_tokens=128,
    )
    print(yaml_text)