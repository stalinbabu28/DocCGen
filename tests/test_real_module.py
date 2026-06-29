from llama_cpp import Llama, LlamaGrammar

grammar = LlamaGrammar.from_string(r'''
root ::= "{\"name\":\"" word "\",\"resource_group\":\"" word "\",\"location\":\"" word "\",\"state\":\"present\"}" |
          "{\"name\":\"" word "\",\"resource_group\":\"" word "\",\"location\":\"" word "\",\"state\":\"absent\"}"

word ::= [a-zA-Z0-9_-]+
''')

llm = Llama(
    model_path="/home/dao-lab/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf",
    n_ctx=2048,
    n_gpu_layers=20,
    verbose=False
)

response = llm(
    """
    Create a virtual network named prod-vnet
    in resource group prod-rg
    located in eastus.
    """,
    grammar=grammar,
    max_tokens=100,
    temperature=0
)

print(response["choices"][0]["text"])