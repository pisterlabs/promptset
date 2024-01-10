from operator import itemgetter

import streamlit as st
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from src.chat_session_manager import (
    display_chat_history,
    restore_history_from_memory,
    save_history,
    save_history_to_file,
    send_message,
)
from src.embedder import Embedder


def format_docs(docs):
    """
    Takes a list of documents and returns a string with the page content of
    each document separated by two new lines.

    Args:
        docs: A list of documents. Each document has a property called "page_content" which contains the content of the document.

    Returns:
        str: A string containing the page content of each document in the input list.
    """
    return "\n\n".join(document.page_content for document in docs)


def manage_chat_session(file, prompt, llm, history_file_path, **kwargs):
    """
     Manages a chat session.

     It is responsible for the following (probably too many things):
        - Embedding a file
        - Displaying chat history
        - Receiving user input
        - Processing user input by chaining functions
        - Saving chat history

    Args:
        file: The path to the file containing the data you want to use for
        the chat session. It could be a text file, a PDF file, or a DOCX file.
        prompt: The prompt template for the chat model.
        llm: The language model that will be used for generating responses in the chat session.
        history_file_path: The file path where the chat history will be saved.
    """
    retriever = Embedder.embed_file(file, **kwargs)
    send_message("I'm ready! Ask away.", "ai", save=False)
    restore_history_from_memory()
    display_chat_history()

    message = st.chat_input("Ask me any question about your file")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(
                chat_history=RunnableLambda(
                    st.session_state["memory"].load_memory_variables
                )
                | itemgetter("chat_history")
            )
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            result = chain.invoke(message)
            save_history(message, result.content)

        if len(st.session_state["memory"].chat_memory.messages) != 0:
            save_history_to_file(history_file_path)
