from pipeline.schema_extractor import extract_schema
from pipeline.extract_params_grammar import (
    generate_constrained_json
)

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/"
    "azure.azcollection.azure_rm_dnszone.json"
)

query = (
    "create dns zone example.com "
    "in resource group prod-rg"
)

result = generate_constrained_json(
    query,
    schema
)

print(result)