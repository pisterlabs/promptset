from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
import openai
import langchain



load_dotenv()
langchain.debug = os.getenv("LANGCHAIN_DEBUG") == "True"


class Config:
    model_name = os.getenv("OPENAI_MODEL")
    llm_cache = os.getenv("LLM_CACHE") == "True"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    assert openai.api_key is not None, "Open AI key not found"
    terminate_token = os.getenv("TERMINATE_TOKEN")

    config_list = [
    {
        "model": model_name,
        "api_key": openai.api_key ,
    }
    ]
    llm_config = {
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT")),
    "seed": int(os.getenv("SEED")),
    "config_list": config_list,
    "temperature": int(os.getenv("TEMPERATURE")),
    }
    max_consecutive_auto_reply = int(os.getenv("MAX_AUTO_REPLY"))
    code_dir = os.getenv("CODE_DIR")
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=0,
        request_timeout=os.getenv("REQUEST_TIMEOUT"),
        cache=llm_cache,
        streaming=True,
        verbose=True
    )
    ui_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    save_html_path = Path(os.getenv("SAVE_HTML"))

    if not save_html_path.exists():
        save_html_path.mkdir(exist_ok=True, parents=True)
    
    project_root = Path(os.getenv("PROJECT_ROOT"))

cfg = Config()


if __name__ == "__main__":
    #print("key: ", cfg.openai_api_key)
    print("model: ", cfg.model_name)
    print("configlist: ", cfg.config_list)
    print("langchain-debug: ", langchain.debug)