import os

import yaml

from huggingface_hub import HfApi
from langchain.embeddings import HuggingFaceEmbeddings
from omegaconf import OmegaConf


CONFIG = OmegaConf.create(
    yaml.load(open("app/config/model.yaml"), Loader=yaml.FullLoader)
)
LLM = CONFIG.llm.download_args
VS = CONFIG.vectorstore.model

## Download LLAMA model
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
hf = HfApi()
hf.hf_hub_download(
    repo_id=LLM.size,
    filename=LLM.name,
    local_dir= ".",
    local_dir_use_symlinks=False
)
os.system(f"mv {LLM.name} llm.gguf")


## Download HuggingFace model
_ = HuggingFaceEmbeddings(model_name=VS.model_name)