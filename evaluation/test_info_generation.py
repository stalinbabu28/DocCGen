from pipeline.schema_extractor import (
    extract_schema
)

from pipeline.grammar_builder import (
    build_grammar
)

from llama_cpp import (
    Llama,
    LlamaGrammar
)

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

schema = extract_schema(
    "/home/dao-lab/stalin/azure_docs/"
    "azure.azcollection.azure_rm_virtualnetwork_info.json"
)

fields = [
    "name"
]

grammar_text = build_grammar(
    schema,
    include_fields=fields
)

print("=" * 80)
print(grammar_text)
print("=" * 80)

grammar = LlamaGrammar.from_string(
    grammar_text
)

query = (
    "show details of virtual network "
    "prod-vnet"
)

response = llm(
    f"""
User request:

{query}

Output JSON:
""",
    grammar=grammar,
    temperature=0,
    max_tokens=128
)

print(
    response["choices"][0]["text"]
)