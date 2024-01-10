from langchain.retrievers.document_compressors import LLMChainFilter
from typing import Optional, Sequence, Tuple, Any
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks
from langchain import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

class VerboseFilter(LLMChainFilter):
    """
    Filter that uses an LLM to drop documents that aren't relevant to the query.
    Has the following additional properties:
        - Returns both the list of relevant documents, and the irrelevant documents.
        - Adds metadata with an explanation for why the LLM thinks they are 
          relevant/irrelevant, if provided.
        - Supports verbose mode for the LLMChain
    """
    
    reason_metadata_key: str = 'keep_reason'
    
    llm_chain: LLMChain
    """LLM wrapper to use for filtering documents. 
    The chain prompt is expected to parse output to a tuple of Boolean, String
    Where the Boolean value indicates if the document should be returned,
    and the String value contains any justification for the decision."""
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Tuple[Sequence[Document],Sequence[Document]]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []
        removed_docs = []
        
        inputs = [self.get_input(query, doc) for doc in documents]
        results = self.llm_chain.apply_and_parse(inputs)
        
        for result,doc in zip(results,documents):
            include_doc = result[0]
            reason = result[1]
            doc.metadata[self.reason_metadata_key] = reason
            if include_doc:
                filtered_docs.append(doc)
            else:
                removed_docs.append(doc)
        return filtered_docs, removed_docs
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> "VerboseFilter":
        """Create a VerboseFilter from a language model.

        Args:
            llm: The language model to use for filtering.
            prompt: The prompt to use for the filter.
            verbose: Sets the chain to verbose if true
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A VerboseFilter that uses the given language model.
        """
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
        return cls(llm_chain=llm_chain, **kwargs)