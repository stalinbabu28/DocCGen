from llama_cpp import Llama
from llama_cpp import LlamaGrammar

MODEL_PATH = "/home/dao-lab/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=20,
    verbose=False
)

grammar_text = r'''
root ::= "{\"name\":\"" word "\",\"resource_group\":\"" word "\"}"
word ::= [a-zA-Z0-9_.-]+
'''

grammar = LlamaGrammar.from_string(
    grammar_text
)

prompt = """
Could you generate ansible code for assigning a VM with 4gb ram
"""

response = llm(
    prompt,
    grammar=grammar,
    max_tokens=100,
    temperature=0
)

print(response["choices"][0]["text"])