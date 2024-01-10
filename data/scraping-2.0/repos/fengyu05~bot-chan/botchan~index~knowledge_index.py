import glob
import os
from typing import List, Optional

import structlog
import yaml
from langchain import LLMChain, OpenAI, PromptTemplate
from pydantic import BaseModel, Field

from botchan.constants import GPT_3_MODEL_NAME
from botchan.index.utils import read_file_content
from botchan.settings import KNOWLEDGE_ACCEPT_PATTERN

DEFAULT_KNOWLEDGE_INDEX = "_index.yaml"

logger = structlog.getLogger(__name__)

KNOWLEDGE_CONTEXT_LENGTH_LIMIT = 10000

KNOWLEDGE_INDEX_RECOVERY_PROMPT_INPUT_VARIABLES = ["question", "index_mapping"]

KNOWLEDGE_INDEX_RECOVERY_PROMPT_TEMPLATE = """
Find out which files describe by the below mapping has the best relevance with the question.
------------
Mapping:
{index_mapping}

Question: {question}
------------
Answer with the single `file_path`, please do not included any other text.
"""

META_CHAIN = LLMChain(
    llm=OpenAI(
        model_name=GPT_3_MODEL_NAME,
        temperature=0,
    ),
    prompt=PromptTemplate(
        input_variables=KNOWLEDGE_INDEX_RECOVERY_PROMPT_INPUT_VARIABLES,
        template=KNOWLEDGE_INDEX_RECOVERY_PROMPT_TEMPLATE,
    ),
    verbose=False,
)


class IndexInfo(BaseModel):
    path: str = Field(..., description="File path")
    summary: str = Field(..., description="Description of the file")
    kind: str = Field(..., description="Kind of the file")


class KnowledgeIndex(BaseModel):
    knowledge_folder: str
    items: List[IndexInfo]

    @classmethod
    def from_folder(cls, knowledge_folder: str) -> "KnowledgeIndex":
        logger.info("create index from folder", knowledge_folder=knowledge_folder)
        with open(
            os.path.join(knowledge_folder, DEFAULT_KNOWLEDGE_INDEX),
            "r",
            encoding="utf8",
        ) as stream:
            try:
                data = yaml.safe_load(stream)
                return cls(items=data, knowledge_folder=knowledge_folder)
            except yaml.YAMLError as exc:
                print(exc)

    def learn_all_knowledge_flat(
        self,
        patterns: list[str] = None,
        knowledge_files: Optional[list[str]] = None,
    ) -> str:
        """
        Concatenates text from all files in the specified folder and file types.

        Args:
            knowledge_folder (str): Path to the folder containing documents.
            patterns (list): List of file type patterns (e.g. ['*.txt', '*.md']).
            knowledge_files (list): List of specific files to read.

        Returns:
            str: Concatenated string of all text from the files.
        """
        all_text = ""
        if patterns is None:
            patterns = KNOWLEDGE_ACCEPT_PATTERN
        if knowledge_files:
            for file_path in knowledge_files:
                all_text += read_file_content(
                    os.path.join(self.knowledge_folder, file_path),
                    KNOWLEDGE_CONTEXT_LENGTH_LIMIT,
                )

        else:
            for pattern in patterns:
                for file_path in glob.glob(
                    os.path.join(self.knowledge_folder, "*" + pattern)
                ):
                    all_text += KnowledgeIndex._read_file_content(file_path)

        return all_text

    @property
    def index_mapping_str(self) -> str:
        """Generate the index mapping from the this data class in the below format.

        behavior/ml_ranker.md: "description of the entry";
        behavior/metrics_notebook.md: "description of the entry";
        """
        mapping_str = ""
        for item in self.items:
            mapping_str += f'{item.path}: "{item.summary}";\n'

        return mapping_str

    def locate_knowledge(self, question: str) -> str:
        knowledge_file_answers = META_CHAIN.predict(
            question=question, index_mapping=self.index_mapping_str
        )
        knowledge_files = knowledge_file_answers.split(",")
        return self.learn_all_knowledge_flat(knowledge_files=knowledge_files)
