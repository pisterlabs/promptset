# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Module for MSTICPy natural language doc search."""
import contextlib
import logging
from pathlib import Path
from IPython.display import Markdown

from langchain.document_loaders import ReadTheDocsLoader

from retrieval_base import RetrievalBase
from vector_stores import create_vector_store, read_vector_store
from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question
(enclosed in <<< and >>>).
If you don't know the answer, just say that you don't know, don't try to make
up an answer.

{context}

Question: <<<{question}>>>

Additional instructions:
1. When answering give preference to functions that are exposed
via a pandas extension method (e.g. `df.mp_plot.<func_name>()`) or
a pivot function (e.g. `IpAddress.<func_name>()`) over a standard function.

2. Supply any relevant code examples and include these with your answer
(ensure you delimit the code with triple backticks and add the language name).

3. Examine your answer and ensure that you are not using functions imported
from a deprecated location. Your examples must not include any functions or
modules imported from the paths 'msticpy.sectools' or 'msticpy.nbtools'.

Return the answer in Markdown format.
"""

MP_VS_PATH = "E:/src/chat-langchain/mp-rtd-vs.pkl"


class RTDocSearch(RetrievalBase):
    """Ask questions about the MSTICPy Documentation."""

    _PROMPT_TEMPLATE = PROMPT_TEMPLATE

    def __init__(
        self,
        vs_path: str = MP_VS_PATH,
        max_tokens: int = 1000,
        memory: bool = False,
        verbose: bool = False,
        model_name: str = "gpt-3.5-turbo",
    ):
        """Initialize the code search."""
        super().__init__(
            vs_path=vs_path or MP_VS_PATH,
            max_tokens=max_tokens,
            memory=memory,
            model_name=model_name,
            verbose=verbose,
        )

    @classmethod
    def create_vectorstore(cls, input_path: str, vs_path: str):
        """
        Read in RTD HTML docs to vector store and save.

        Parameters
        ----------
        doc_path : str
            Path to read HTML documents from.
        vs_path : str
            Path to save pickled vectorstore to.

        """
        logger.info("Loading documents from %s", input_path)
        if Path(vs_path).exists():
            with contextlib.suppress(ValueError):
                return read_vector_store(vs_path, f"{cls.__name__}.create_vector_store")
        print("Loading document files from", input_path, "...")
        loader = ReadTheDocsLoader(
            input_path, errors="ignore", encoding="utf-8", features="lxml"
        )
        return create_vector_store(loader, vs_path)

    def ask(self, question: str):
        """Ask a question about MSTICPy documents and return the answer as Markdown."""
        return Markdown(self.retriever(question)["result"])


# mp_doc_search = RTDocSearch()


# def search_doc(question: str):
#     """Search the code for the answer to the question."""
#     response = mp_doc_search.ask(question)
#     return Markdown(response)
