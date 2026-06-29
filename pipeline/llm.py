from functools import lru_cache
import os

from llama_cpp import Llama

MODEL_PATH = (
    "/home/dao-lab/.lmstudio/models/"
    "unsloth/gpt-oss-20b-GGUF/"
    "gpt-oss-20b-F16.gguf"
)


@lru_cache(maxsize=1)
def get_llm():
    use_gpu = os.getenv("USE_GPU", "1") == "1"

    print(f"Loading model (GPU={use_gpu})")
    print(f"Loading model using {'GPU' if use_gpu else 'CPU'}")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=20 if use_gpu else 0,
        verbose=False,
    )