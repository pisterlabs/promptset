"""By breaking down the code into smaller modules, you make it more modular and easier to maintain.
Each module focuses on a specific task, making it easier to understand and optimize individual components.
"""
from typing import Any

from langchain.chains import RetrievalQA

from core import app

from .modules.config import get_env_variable
from .modules.llms_setup import (
    setup_gpt4all,
    setup_llama_cpp,
)
from .modules.vector_store_setup import setup_chroma_vector_store

target_source_chunks = int(get_env_variable("TARGET_SOURCE_CHUNKS", 4))  # type: ignore


def setup_chain() -> Any:
    """_summary_
    Returns:
        Any: _description_
    """
    model_type = get_env_variable("MODEL_TYPE")

    db = setup_chroma_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    if model_type == "GPT4All":
        llm_params = {}
        llm = setup_gpt4all(llm_params)
    elif model_type == "LlamaCpp":
        llm_params = {}  # Add any specific parameters needed for LlamaCpp setup
        llm = setup_llama_cpp(llm_params)
    else:
        print(f"Model {model_type} not supported!")
        return None

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not app.config.get("HIDE_SOURCE", False),
    )
