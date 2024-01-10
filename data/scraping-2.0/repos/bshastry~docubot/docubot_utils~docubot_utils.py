#!/usr/bin/env python3

"""
This module provides functions for answering questions using retrieval-based question answering systems.

It includes functions for answering a single question, answering a question within a conversational context,
and formatting source document citations.

Note: This module requires the 'langchain' library to be installed.
"""
from typing import List, TypeVar

T = TypeVar("T")


def answer_question(question: str, vector_store, num_neighbors=5):
    """
    This function takes a question as input and returns an answer using a retrieval-based question answering system.

    Args:
    - question (str): The question to be answered.
    - vector_store: A vector store object used for retrieving similar documents.
    - num_neighbors (int): The number of similar documents to retrieve.

    Returns:
    - The answer to the input question.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": num_neighbors}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    return chain.run(question)


def answer_question_session(
    question: str, vector_store, num_neighbors=5, chat_history=[]
):
    """
    This function takes a question, a vector store, and an optional number of neighbors and chat history.
    It uses the LangChain library to create a conversational retrieval chain, which is used to generate an answer to the question.
    The answer is then appended to the chat history.
    The function returns a dictionary containing the answer and the source documents used to generate the answer, as well as the updated chat history.

    Args:
        question (str): The question to be answered.
        vector_store: The vector store used to retrieve similar documents.
        num_neighbors (int, optional): The number of similar documents to retrieve. Defaults to 5.
        chat_history (list, optional): A list of tuples containing the question and answer pairs from previous conversations. Defaults to an empty list.

    Returns:
        tuple: A tuple containing a dictionary with the answer and source documents, and the updated chat history.
    """
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": num_neighbors}
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    result = conversational_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result, chat_history


def format_citations(source_docs: List[T]) -> str:
    """
    Formats a list of source documents into a string of citations.

    Args:
        source_docs (List[T]): A list of source documents.

    Returns:
        str: A string of formatted citations.
    """
    # Iterate through source_docs and reference a field in each index called metadata
    citation_list = []
    for index, doc in enumerate(source_docs):
        source_document_name = doc.metadata["source"]
        page = doc.metadata.get("page")
        page_str = str(int(page)) if page else "not available"
        citation_list.append(
            "{}: {}, page {}".format(index, source_document_name, page_str)
        )
    return "\n".join(citation_list)
