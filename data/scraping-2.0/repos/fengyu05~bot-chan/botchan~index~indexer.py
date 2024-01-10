import os

import yaml
from langchain import LLMChain, OpenAI, PromptTemplate

from botchan.constants import GPT_3_MODEL_NAME
from botchan.index.doc_kind import DocKind
from botchan.index.knowledge_index import DEFAULT_KNOWLEDGE_INDEX
from botchan.index.utils import read_file_content

_INDEXER_SUMMARY_LENGTH_LIMIT = 4096

_INDEXER_TEMPLATE = """
{content}
---
Please summary the above paragraph.
"""


class Indexer:
    def __init__(
        self, knowledge_folder: str, file_extensions: tuple, dry_run: bool = True
    ) -> None:
        self.knowledge_folder = knowledge_folder
        self.file_extensions = file_extensions
        self.dry_run = dry_run
        self.records = []
        self.summary_chain = LLMChain(
            llm=OpenAI(
                model_name=GPT_3_MODEL_NAME,
                temperature=0,
            ),
            prompt=PromptTemplate(
                input_variables=["content"],
                template=_INDEXER_TEMPLATE,
            ),
            verbose=False,
        )

    def process(self, file_path: str) -> None:
        print(f"Processing {file_path}...")
        self.records.append(
            {
                "path": file_path,
                "summary": self.summary(file_path),
                "kind": self.extract_kind(file_path).name,
            }
        )

    def walk(self) -> None:
        for dirpath, _, filenames in os.walk(self.knowledge_folder):
            for filename in filenames:
                if filename.lower().endswith(self.file_extensions):
                    file_path = os.path.join(dirpath, filename)
                    self.process(file_path)

    def summary(self, file_path: str) -> str:
        content = read_file_content(file_path, _INDEXER_SUMMARY_LENGTH_LIMIT)
        summary = self.summary_chain.predict(
            content=content,
        )
        return summary

    def extract_kind(self, file_path: str) -> DocKind:
        _, extension = os.path.splitext(file_path)
        extension = extension.lstrip(".")
        if extension == "md":
            return DocKind.MARK_DOWN
        elif extension == "py":
            return DocKind.SOURCE
        elif extension == "doc":
            return DocKind.WORD
        else:
            return DocKind.UNRECOGNIZED

    def write2yaml(self) -> None:
        """Write records to yaml.

        Records is a list of dict have 3 fields: 'path', 'summary' and 'kind'
        """
        index_output_file = os.path.join(self.knowledge_folder, DEFAULT_KNOWLEDGE_INDEX)
        if self.dry_run:
            index_output_file += ".dryrun"
        with open(index_output_file, "w", encoding="utf8") as outfile:
            yaml.dump(self.records, outfile, default_flow_style=False)


def create_index(knowledge_folder: str, file_extensions: tuple, dryrun: bool) -> None:
    indexer = Indexer(
        knowledge_folder=knowledge_folder,
        file_extensions=file_extensions,
        dry_run=dryrun,
    )
    indexer.walk()
    indexer.write2yaml()
