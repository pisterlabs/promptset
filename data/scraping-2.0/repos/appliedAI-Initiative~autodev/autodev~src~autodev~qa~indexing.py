import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from langchain.schema import Document
from langchain.text_splitter import TextSplitter

log = logging.getLogger(__name__)


@dataclass
class FileInfo:
    path: str
    content: Optional[str] = None
    extension: Optional[str] = None
    size: Optional[int] = None
    line_count: Optional[int] = None
    last_modified: Optional[datetime] = None

    def to_lc_document(self, additional_metadata: Optional[dict] = None):
        if self.content is None:
            raise RuntimeError(
                f"Content must be set to convert to LC Document. Have you modified this object?"
            )

        metadata = {
            "source": self.path,
            "extension": self.extension,
            "size": self.size,
            "line_count": self.line_count,
            "last_modified": self.last_modified.isoformat()
            if self.last_modified is not None
            else None,
        }
        additional_metadata = additional_metadata or {}
        metadata.update(additional_metadata)
        return Document(page_content=self.content, metadata=metadata)

    def __post_init__(self):
        if self.content is None:
            with open(self.path, "r") as f:
                self.content = f.read()

        if self.extension is None:
            self.extension = Path(self.path).suffix

        self.size = len(self.content.encode("utf-8"))
        self.line_count = len(self.content.splitlines())

        if self.last_modified is None:
            self.last_modified = datetime.fromtimestamp(os.path.getmtime(self.path))


def scan_directory(
    path: os.PathLike[str] | str,
    ignored_dir_regex: Optional[
        str | re.Pattern[str]
    ] = r"(_.*)|(data)|(venv)|(env)|(node_modules)",
    ignored_file_regex: Optional[str | re.Pattern] = ".*lock.*",
    last_modified_after: Optional[datetime] = None,
    skip_hidden_dirs=True,
) -> Iterable[FileInfo]:
    """
    Scans a directory and returns an iterable of FileInfo objects for each non-excluded file.
    """

    path = Path(path)

    ignored_dir_regex = (
        re.compile(ignored_dir_regex) if ignored_dir_regex is not None else None
    )

    ignored_file_regex = (
        re.compile(ignored_file_regex) if ignored_file_regex is not None else None
    )

    for file_path in path.iterdir():
        if file_path.is_dir():
            if skip_hidden_dirs and file_path.name.startswith("."):
                log.debug(f"Skipping hidden directory: {file_path}")
                continue
            if ignored_dir_regex and ignored_dir_regex.match(file_path.name):
                log.debug(f"Skipping {file_path} due to {ignored_dir_regex=}")
                continue

            yield from scan_directory(
                file_path, ignored_dir_regex, ignored_file_regex, last_modified_after
            )
        elif file_path.is_file():
            if ignored_file_regex and ignored_file_regex.match(file_path.name):
                log.debug(f"Skipping {file_path} due to {ignored_file_regex=}")
                continue

            last_modified = None
            if last_modified_after is not None:
                try:
                    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                except OSError as e:
                    log.warning(
                        f"Could not access file modification time for {file_path}. Skipping."
                    )
                    continue
                if last_modified < last_modified_after:
                    log.debug(
                        f"Skipping file: {file_path} since it was modified before {last_modified_after}"
                    )
                    continue

            try:
                yield FileInfo(str(file_path.resolve()), last_modified=last_modified)
            except UnicodeDecodeError as e:
                log.warning(f"Could not decode file {file_path}. Skipping.")


def load_lc_documents(
    path: os.PathLike[str] | str,
    ignored_dir_regex: Optional[
        str | re.Pattern[str]
    ] = r"(_.*)|(data)|(venv)|(env)|(node_modules)",
    ignored_file_regex: Optional[str | re.Pattern] = ".*lock.*",
    last_modified_after: Optional[datetime] = None,
    skip_hidden_dirs=True,
    splitter: Optional[TextSplitter] = None,
    splitter_factory: Optional[Callable[[FileInfo], TextSplitter]] = None,
) -> Iterable[Document]:
    if splitter_factory and splitter:
        raise ValueError("Only one of splitter and splitter_factory can be specified")

    if splitter:
        splitter_factory = lambda fi: splitter  # type: ignore

    file_infos = scan_directory(
        path,
        ignored_dir_regex=ignored_dir_regex,
        ignored_file_regex=ignored_file_regex,
        last_modified_after=last_modified_after,
        skip_hidden_dirs=skip_hidden_dirs,
    )
    if not splitter_factory:
        return map(lambda fi: fi.to_lc_document(), file_infos)

    for fi in file_infos:
        splitter = splitter_factory(fi)
        split_docs = splitter.split_documents([fi.to_lc_document()])
        num_parts = len(split_docs)
        for i, doc in enumerate(splitter.split_documents([fi.to_lc_document()])):
            doc.metadata.update(
                {
                    "part": i,
                    "num_parts": num_parts,
                }
            )
            yield doc
