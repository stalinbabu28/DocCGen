from pipeline.schema_extractor import extract_schema
from pipeline.grammar_builder import build_grammar

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/"
    "azure.azcollection.azure_rm_virtualnetwork.json"
)

print(build_grammar(schema))