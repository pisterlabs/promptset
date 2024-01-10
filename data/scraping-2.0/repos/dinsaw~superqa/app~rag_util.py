import logging

from fastapi import HTTPException
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.collection_util import flatten, delete_none_values

logger = logging.getLogger(__name__)

MODEL_TEMPERATURE = 0.1

MODEL_NAME = "gpt-3.5-turbo"


def get_rag(content_type, source_file_location):
    if content_type == "application/pdf":
        return prepare_pdf_rag(source_file_location)
    elif content_type == "application/json":
        return prepare_json_rag(source_file_location)
    else:
        raise HTTPException(
            status_code=500, detail="RAG chain is not defined for this type"
        )


def static_metadata_func(record: dict, metadata: dict) -> dict:
    """
    Static metadata function is better specially for converting keys in nested structure.
    But not using it for generalization purpose.
    :param record:
    :param metadata:
    :return:
    """
    metadata["content"] = record.get("content")
    metadata["answer"] = record.get("answer")
    metadata["comment"] = record.get("comment")
    metadata["id"] = record.get("id")
    metadata["createdAt"] = record.get("createdAt")
    metadata["modifiedAt"] = record.get("modifiedAt")
    metadata["pageNumber"] = record.get("pageNumber")
    metadata["questionNumber"] = record.get("questionNumber")
    metadata["product"] = record.get("product")
    if record.get("products"):
        metadata["products"] = ""
        for product in record.get("products"):
            metadata["products"] += product.get("id")

    metadata["isFavorite"] = record.get("isFavorite")
    metadata["accessLevel"] = record.get("accessLevel")
    metadata["source"] = record.get("source")
    if record.get("subtype"):
        metadata["raw_subtypes"] = ",".join(record.get("subtype").get("raw", []))
        metadata["pretty_subtypes"] = ",".join(record.get("subtype").get("pretty", []))

    delete_none_values(metadata)

    logger.debug(f"metadata={metadata}")

    return metadata


def dynamic_metadata_func(record: dict, metadata: dict) -> dict:
    metadata = flatten(record)

    delete_none_values(metadata)

    logger.debug(f"metadata={metadata}")

    return metadata


def prepare_json_rag(source_file_location):
    loader = JSONLoader(
        file_path=source_file_location,
        jq_schema=".[]",
        content_key="content",
        metadata_func=dynamic_metadata_func,
    )

    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=MODEL_TEMPERATURE)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, vectorstore


def prepare_pdf_rag(source_file_location):
    loader = PyPDFLoader(source_file_location)
    pages = loader.load_and_split()
    logger.info(f"pages={pages}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=MODEL_TEMPERATURE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, vectorstore
