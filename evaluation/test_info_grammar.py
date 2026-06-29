from pipeline.schema_extractor import (
    extract_schema
)

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/"
    "azure.azcollection.azure_rm_virtualnetwork_info.json"
)

print("=" * 80)
print("REQUIRED")
print(schema["required"])

print("=" * 80)
print("OPTIONAL")
print(schema["optional"])