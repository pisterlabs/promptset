"""
Prompt chaining and vectorstore example using LangChain
- load all code from a code base and answer questions about it as a chatbot.
Based on chat-langchain (https://github.com/hwchase17/chat-langchain/)
"""

import pickle
import os
import logging

from langchain.callbacks import CallbackManager, StdOutCallbackHandler
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from src.utils.mapping_directory_loader import MappingDirectoryLoader

logger = logging.getLogger(__name__)

def ingest_docs(path: str):
    """Read, split and store code and other files from within the repository/folder."""
    loader = MappingDirectoryLoader(path, recursive=True, silent_errors=True)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def get_chain(
    vectorstore: VectorStore,
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""

    # callback manager for logging
    manager = CallbackManager([StdOutCallbackHandler()])

    # LLM interface (needs os.environ["OPENAI_API_KEY"] set)
    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
    )

    # Creates standalone question from chat history context
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager, verbose=True
    )

    # Asks final question
    doc_chain = load_qa_chain(
        question_gen_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager, verbose=True
    )

    # Chains together QuestionGenerator->MemoryLookup->QuestionAnswering
    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
    )
    return qa


if __name__ == "__main__":
    # vector-store docs (only necessary on first run) (delete vectorstore.pkl if you change the path)
    if not os.path.exists("vectorstore.pkl"):
        logger.info("No pickle file found, ingesting docs...")
        ingest_docs(r"C:\Users\colli\PycharmProjects\langchain-master")
    else:
        logger.info("Using existing pickle file.")

    # Load Up vectorstore
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)

    # Chat loop
    chat_history = []
    while True:
        question = input("Question: ")
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(result['answer'])