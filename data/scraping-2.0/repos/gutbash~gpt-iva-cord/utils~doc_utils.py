from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain

class AnalyzeDocumentChain(Chain):
    """Chain that splits documents, then analyzes it in pieces."""

    input_key: str = "input_document"  #: :meta private:
    output_key: str = "output_text"  #: :meta private:
    text_splitter: TextSplitter = Field(default_factory=RecursiveCharacterTextSplitter)
    combine_docs_chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(other_keys, return_only_outputs=True)

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs

        # Assuming combine_docs_chain has an async __call__ method
        combined_docs = await self.combine_docs_chain(other_keys, return_only_outputs=True)
        print("COMBINED DOCS")
        return combined_docs
