from pathlib import Path
from types import MappingProxyType

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

from ragga.core.config import Config, Configurable


class MarkdownDataset(Configurable):
    """Markdown dataset"""

    _config_key = "dataset"
    _default_config = MappingProxyType({
        "path": "data",
        "recursive": True,
        "show_progress": True,
        "use_multithreading": True,
    })

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        self._documents: list[Document] | None = None
        default_loader_kwargs = {
            "mode": "single",
            "unstructured_kwargs": {
                "include_metadata": True,
                "languages": ["en"],
                "chunking_strategy": "by_title",
            }
        }
        self._merge_default_kwargs(default_loader_kwargs, "loader_kwargs")

    @property
    def documents(self) -> list[Document]:
        if self._documents is None:
            self._load_documents()
        return self._documents  # type: ignore

    def _load_documents(self) -> None:
        path = Path(self.config[self._config_key]["path"])
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise ValueError(msg)
        if not path.is_dir():
            msg = f"Path is not a directory: {path}"
            raise ValueError(msg)

        loader = DirectoryLoader(
            self.config[self._config_key]["path"],
            glob="**/*.md",
            show_progress=self.config[self._config_key]["show_progress"],
            use_multithreading=self.config[self._config_key]["use_multithreading"],
            recursive=self.config[self._config_key]["recursive"],
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs=self.config[self._config_key]["loader_kwargs"],
        )
        self._documents = loader.load()
        if len(self._documents) == 0:
            msg = "No documents found in dataset path"
            raise ValueError(msg)


    def __repr__(self) -> str:
        return f"MarkdownDataset({self.config[self._config_key]['path']})"
