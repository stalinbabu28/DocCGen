from pipeline.schema_extractor import (extract_schema)
from pipeline.extract_params_grammar import (generate_constrained_json)

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/"
    "azure.azcollection.azure_rm_virtualnetwork.json"
)

query = (
    "create virtual network prod-vnet "
    "in resource group prod-rg"
)
from pipeline.grammar_builder import build_grammar

print("=" * 80)
print(build_grammar(schema))
print("=" * 80)
print(
    generate_constrained_json(
        query,
        schema
    )
)