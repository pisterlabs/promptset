from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI


load_dotenv()


class Config:
    model_name = os.getenv("OPENAI_MODEL")
    llm_cache = os.getenv("LLM_CACHE") == "True"
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=0,
        request_timeout=os.getenv("REQUEST_TIMEOUT"),
        cache=llm_cache,
        streaming=True,
    )
    verbose_llm = os.getenv("VERBOSE_LLM") == "True"

    python_executor = Path(os.getenv("PYTHON_SCRIPT"))
    if not python_executor.exists():
        python_executor.mkdir(exist_ok=True, parents=True)

    ui_timeout = os.getenv("REQUEST_TIMEOUT")

    image_path = Path(os.getenv("IMAGE_PATH_DISC"))
    if not image_path.exists():
        image_path.mkdir(exist_ok=True, parents=True)

    project_root = Path(os.getenv("PROJECT_ROOT"))


cfg = Config()


if __name__ == "__main__":
    print("llm: ", cfg.llm)
    print("project root: ", cfg.project_root / "toml_support")
