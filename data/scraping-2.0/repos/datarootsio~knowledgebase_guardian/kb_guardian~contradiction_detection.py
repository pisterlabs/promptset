from typing import List

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStoreRetriever

from kb_guardian.logger import INFO_LOGGER, log_contradiction_result
from kb_guardian.utils.deployment import get_deployment_llm
from kb_guardian.utils.paths import get_config

CONFIG = get_config()


def get_detection_chain(retriever: VectorStoreRetriever) -> RetrievalQAWithSourcesChain:
    """
    Construct and return an LLM chain to detect contradictions at ingestion time.

    Args:
        retriever (VectorStoreRetriever): A retriever for the vector store to which you want to add new documents

    Returns:
        RetrievalQAWithSourcesChain: A retrieval chain that allows you to detect contradictions
    """  # noqa: E501
    llm = get_deployment_llm()

    messages = [
        SystemMessagePromptTemplate.from_template(CONFIG["system_message"]),
        HumanMessagePromptTemplate.from_template(CONFIG["user_message"]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
        },
    )

    chain.reduce_k_below_max_tokens = True
    chain.return_source_documents = True
    return chain


def contradiction_detection(
    vectorstore: VectorStore,
    chunks: List[Document],
) -> VectorStore:
    """
    Use an LLM to detect inconsistencies / contradictions between the documents in a vectorstore and the documents that should be added to that vectorstore.

    New documents that do not contradict the existing documents are added to the vectorstore.
    New documents that are contradicting the existing documents are not added. Instead, a log with information about the detected contradictions will be stored.

    Args:
        vectorstore (VectorStore): An existing vectorstore to which you want to add new information
        docs_chunks (List[Document]): Chunks of the new documents you want to add

    Returns:
        VectorStore: An updated version of the vectorstore, including the new documents that do not contradict the existing documents.
    """  # noqa: E501
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": CONFIG["nb_retrieval_docs"]}
    )
    chain = get_detection_chain(retriever)

    nb_valid, nb_contradiction = 0, 0
    for chunk in chunks:
        result = chain({"question": chunk.page_content})
        if result["answer"].startswith("CONSISTENT"):
            vectorstore.add_documents([chunk])
            log_contradiction_result(chunk, result, contradiction=False)
            nb_valid += 1
        else:
            log_contradiction_result(chunk, result, contradiction=True)
            nb_contradiction += 1

    INFO_LOGGER.info(
        f"""
        {nb_valid} new chunks have been added to the vectorstore.
        {nb_contradiction} chunks were found to contain contradictions and have been rejected."""  # noqa: E501
    )

    return vectorstore
