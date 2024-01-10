import logging
from logging import Logger
from typing import Any, Dict, List

from langchain.schema import Document


def setup_logger(logger_name: str, file_name: str) -> Logger:
    """
    Initialize a logger that writes to a certain file.

    Args:
        logger_name (str): The name of the logger
        file_name (str): The file to which the logger should write

    Returns:
        Logger: A logger object
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


INFO_LOGGER = setup_logger("execution_info", "execution.log")
CONTRADICTION_LOGGER = setup_logger("contradictions", "contradictions.log")


def format_retrieved_documents(source_docs: List[Document]) -> str:
    """
    Format the content of the retrieved documents in a format suitable for logging.

    Args:
        source_docs (List[Document]): a list of retrieved Langchain Documents

    Returns:
        str: A single string containing a formatted version of the content of the retrieved documents.
    """  # noqa: E501
    retrieved_docs = ""
    for i, doc in enumerate(source_docs):
        content = doc.page_content.replace("\n", "\n\t\t")
        retrieved_docs += f"""
        {i+1}. Retrieved chunk from document {doc.metadata['source']} with the following content:
        {content}
        """  # noqa: E501
    return retrieved_docs


def log_contradiction_result(
    doc: Document, llm_result: Dict[str, Any], contradiction=True
):
    """
    Log the document for which a contradiction was detected, together with the explanation of the LLM and the documents from the vectorstore with which the new document conflicts.

    Args:
        doc (Document): The new document (chunk) that conflicts with the documents in the existing vectorstore
        llm_result (Dict[str, Any]): The output of the LLM, describing the contradiction(s).
        logger (Logger): The logger object to use for logging the contradictions.
    """  # noqa: E501
    result = "Contradiction" if contradiction else "No contradiction"
    logger = CONTRADICTION_LOGGER if contradiction else INFO_LOGGER

    retrieved_docs = format_retrieved_documents(llm_result["source_documents"])
    answer = llm_result["answer"].replace("\n", "\n\t\t")
    doc_content = doc.page_content.replace("\n", "\n\t\t")

    logger.info(
        f"""
        {result} detected for document {doc.metadata["source"]}.

        Conclusion from the LLM:
        {answer}

        Content of new document:
        {doc_content}

        Retrieved documents:
        {retrieved_docs}
    """
    )
