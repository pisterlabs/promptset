from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download


# MODEL_ID = "TheBloke/zephyr-7B-beta-GGUF"
MODEL_ID = "TheBloke/Merged-DPO-7B-GGUF"
MODEL_BASENAME = "merged-dpo-7b.Q2_K.gguf"

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = 1024

model_path = hf_hub_download(
  repo_id=MODEL_ID,
  filename=MODEL_BASENAME,
  resume_download=True,
  cache_dir="./models",
)

kwargs = {
  "model_path": model_path,
  "n_ctx": CONTEXT_WINDOW_SIZE,
  "max_tokens": MAX_NEW_TOKENS,
  "n_gpu_layers":4,
}

dpo_llm = LlamaCpp(
  model_path=model_path,
  temperature=0.1,
  n_ctx=4096,
  max_tokens=1024,
  n_batch=100,
  top_p=1,
  verbose=True,
  n_gpu_layers=100,
  )