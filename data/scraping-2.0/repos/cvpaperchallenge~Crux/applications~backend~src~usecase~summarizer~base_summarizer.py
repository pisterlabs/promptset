import pathlib
from abc import ABC

from jinja2 import Environment, FileSystemLoader
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores.base import VectorStore

from src.domain.paper_format_dto import SummaryFormat


class BaseSummarizer(ABC):
    """Base class for summarizer.

    Args:
        llm_model (BaseLanguageModel): language model to use
        vectorstore (dict[str, VectorStore]): vectorstore to use
        prompt_template_dir_path (pathlib.Path): path to prompt template directory

    """

    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: dict[str, VectorStore],
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        self.llm_model = llm_model
        self.vectorstore = vectorstore
        self.template_env = Environment(
            loader=FileSystemLoader(str(prompt_template_dir_path))
        )

    def summarize(self) -> SummaryFormat:
        raise NotImplementedError
