from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import langchain


load_dotenv()
langchain.debug = os.getenv("LANGCHAIN_DEBUG") == "True"


class Config:
    model_name = os.getenv("OPENAI_MODEL")
    llm_cache = os.getenv("LLM_CACHE") == "True"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    assert openai.api_key is not None, "Open AI key not found"

    emb_func = OpenAIEmbeddings()

    desc_dir = Path(os.getenv("CODE_DIR"))
    if not desc_dir.exists():
        desc_dir.mkdir(exist_ok=True, parents=True)

    graph_input_path = Path(os.getenv("INPUT_PATH"))
    if not graph_input_path.exists():
        graph_input_path.mkdir(exist_ok=True, parents=True)

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=0,
        request_timeout=os.getenv("REQUEST_TIMEOUT"),
        cache=llm_cache,
        streaming=True,
        verbose=True,
    )
    ui_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    save_fig_path = Path(os.getenv("SAVE_FIG"))

    db_path = Path(os.getenv("DB_PATH"))
    chunk_size = int(os.getenv("CHUNK_SIZE"))

    prompts_toml_path = Path(os.getenv("PROMPT_KNOWLEDGE_GRAPH"))

cfg = Config()


if __name__ == "__main__":
    # print("key: ", cfg.openai_api_key)
    print("model: ", cfg.model_name)
    print("configlist: ", cfg.save_fig_path)
    print("langchain-debug: ", langchain.debug)
