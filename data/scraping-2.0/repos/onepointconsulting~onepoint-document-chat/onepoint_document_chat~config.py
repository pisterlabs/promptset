import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()


def create_folder_if_not_exists(folder: Path):
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    assert folder.exists()


class Config:
    model = os.getenv("OPENAI_MODEL")
    lower_model = os.getenv("OPENAI_MODEL_LOWER")
    request_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    has_langchain_cache = os.getenv("LANGCHAIN_CACHE") == "true"
    streaming = os.getenv("CHATGPT_STREAMING") == "true"
    verbose_llm = os.getenv("VERBOSE_LLM") == "true"
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
        temperature=0,
        request_timeout=request_timeout,
        cache=has_langchain_cache,
        streaming=streaming,
        verbose=verbose_llm,
    )
    llm_optional = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=lower_model,
        temperature=0,
        request_timeout=request_timeout,
        cache=has_langchain_cache,
        streaming=streaming,
        verbose=verbose_llm,
    )
    langchain_verbose = os.getenv("LANGCHAIN_VERBOSE") == "true"
    chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE"))
    embeddings = OpenAIEmbeddings(chunk_size=chunk_size)

    project_root = Path(os.getenv("PROJECT_ROOT"))
    data_folder = Path(os.getenv("DATA_FOLDER"))
    assert data_folder.exists()
    ui_folder = Path(os.getenv("UI_FOLDER"))
    assert ui_folder.exists()

    tmp_folder = Path(os.getenv("TMP_FOLDER"))
    create_folder_if_not_exists(tmp_folder)
    enhanced_text_folder = tmp_folder / "enhanced"
    create_folder_if_not_exists(enhanced_text_folder)
    extraction_text_folder = tmp_folder / "extraction"
    create_folder_if_not_exists(extraction_text_folder)
    embeddings_folder_faiss = Path(os.getenv("EMBEDDINGS_FOLDER_FAISS"))
    create_folder_if_not_exists(embeddings_folder_faiss)

    # Minimum length of document
    doc_min_length = int(os.getenv("DOC_MIN_LENGTH"))
    # Maximum allowed number of tokens in single context.
    context_token_limit = int(os.getenv("CONTEXT_TOKEN_LIMIT"))
    # How many search results
    search_results_how_many = int(os.getenv("SEARCH_RESULTS_HOW_MANY"))
    search_results_extra_attempts = int(os.getenv("SEARCH_RESULTS_EXTRA_ATTEMPTS"))

    # Websocket
    websocket_cors_allowed_origins = os.getenv("WEBSOCKET_CORS_ALLOWED_ORIGINS")
    websocket_server = os.getenv("WEBSOCKET_SERVER")
    assert websocket_server is not None
    websocket_port = int(os.getenv("WEBSOCKET_PORT"))

    # Web server
    webserver_files = os.getenv("WEBSERVER_FILES")
    webserver_server = os.getenv("WEBSERVER_SERVER")
    webserver_port = int(os.getenv("WEBSERVER_PORT"))
    webserver_upload_folder = Path(os.getenv("WEBSERVER_UPLOAD_FOLDER"))
    webserver_upload_token = os.getenv("WEBSERVER_UPLOAD_TOKEN")
    assert webserver_upload_token is not None

    create_folder_if_not_exists(webserver_upload_folder)

    def __repr__(self) -> str:
        props = {
            "model": self.model,
            "llm": self.llm,
            "request_timeout": self.request_timeout,
            "streaming": self.streaming,
            "langchain_verbose": self.langchain_verbose,
            "data_folder": self.data_folder,
            "tmp_folder": self.tmp_folder,
            "project_root": self.project_root,
            "embeddings": self.embeddings,
            "embeddings_folder_faiss": self.embeddings_folder_faiss,
            "doc_min_length": self.doc_min_length,
        }
        s = ""
        for k, v in props.items():
            s += f"{k}: {v}\n"
        return s


cfg = Config()

if __name__ == "__main__":
    from onepoint_document_chat.log_init import logger

    assert cfg is not None
    logger.info(cfg)
    for f in cfg.data_folder.iterdir():
        print(f)
