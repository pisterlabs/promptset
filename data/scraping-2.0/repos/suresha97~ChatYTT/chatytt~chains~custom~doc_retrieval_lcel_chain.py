from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema import format_document
from langchain_core.callbacks import CallbackManagerForChainRun

from chatytt.vector_store.pinecone_db import PineconeDB


class DocRetrievalLCELChain(Chain):
    query_key: str = "query"
    top_k_docs_key: str = "top_k"
    vector_store_key: str = "vector_store"
    output_key: str = "context"

    @property
    def input_keys(self) -> List[str]:
        return [self.query_key, self.top_k_docs_key, self.vector_store_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        retrieval_chain = (
            inputs["vector_store"].as_retriever(search_kwargs={"k": inputs["top_k"]})
            | _combine_documents
        )

        docs = retrieval_chain.invoke(inputs["query"])

        return {self.output_key: docs}


def context_container_prompt():
    return PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_separator="\n\n"):
    doc_strings = [format_document(doc, context_container_prompt()) for doc in docs]
    return document_separator.join(doc_strings)


if __name__ == "__main__":
    load_dotenv()

    pinecone_db = PineconeDB(
        index_name="youtube-transcripts", embedding_source="open-ai"
    )

    query = (
        "Is buying a house a good financial decision to make in your 20s ? Give details on the "
        "reasoning behind your answer. Also provide the name of the speaker in the provided context from"
        "which you have constructed your answer."
    )
    doc_retrieval_chain = DocRetrievalLCELChain()

    res = doc_retrieval_chain.run(
        query=query, vector_store=pinecone_db.vector_store, top_k=5
    )
    print(res)
