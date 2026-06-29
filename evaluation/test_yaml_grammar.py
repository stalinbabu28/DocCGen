from __future__ import annotations

import yaml

from pipeline.schema_extractor import extract_schema
from pipeline.yaml_grammar_builder import build_yaml_grammar

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/azure.azcollection.azure_rm_virtualnetwork.json"
)

grammar = build_yaml_grammar(
    module_fqn="azure.azcollection.azure_rm_virtualnetwork",
    schema=schema,
    include_fields=["name", "resource_group"],
)

print("=" * 80)
print(grammar)
print("=" * 80)

print(
    yaml.safe_load(
        """- name: Generated Task
  azure.azcollection.azure_rm_virtualnetwork:
    name: prod-vnet
    resource_group: prod-rg
"""
    )
)