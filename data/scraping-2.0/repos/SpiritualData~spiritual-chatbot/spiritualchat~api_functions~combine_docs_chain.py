from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from langchain.schema import BaseMessage, BaseRetriever, Document

class NamespaceStuffDocumentsChain(StuffDocumentsChain):
    def _create_context_from_docs(self, namespace_docs: Dict[str, List[Document]]) -> str:
        """Create context from docs."""
        # This is just a simple example. You might want to adjust the format of the context string.
        context_parts = []
        for namespace, docs in namespace_docs.items():
            if docs is not None:  # add this line
                doc_texts = [doc.page_content for doc in docs]
                context_parts.append(f"{namespace}:\n" + "\n\n".join(doc_texts))
        return "\n\n".join(context_parts)

    def _get_inputs(self, input_documents: Dict[str, List[Document]], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and namespace_docs.

        Format and the join all the documents together into one input with name
        `self.document_variable_name` using _create_context_from_docs. The pluck any additional variables
        from **kwargs.

        Args:
            input_documents: Dict of namespace to list of documents to create the context.
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Add context to the inputs using _create_context_from_docs
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self._create_context_from_docs(input_documents)
        return inputs