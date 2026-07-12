from functools import lru_cache
import os
from pathlib import Path

from llama_cpp import Llama


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "/home/dao-lab/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf",
    )
)


@lru_cache(maxsize=1)
def get_llm():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{MODEL_PATH}"
        )

    print("=" * 80)
    print("Loading model")
    print(f"Path : {MODEL_PATH}")
    print(f"Size : {MODEL_PATH.stat().st_size / (1024**3):.2f} GB")
    print("=" * 80)

    use_gpu = os.getenv("USE_GPU", "1") == "1"

    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_gpu_layers=20 if use_gpu else 0,
            verbose=True,
        )

        print("Model loaded successfully.")
        return llm

    except Exception as e:
        print("\nFAILED TO LOAD MODEL\n")
        print(e)
        raise