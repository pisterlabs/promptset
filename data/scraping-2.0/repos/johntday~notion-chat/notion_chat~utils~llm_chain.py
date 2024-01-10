import os
from typing import Any

import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from notion_load.qdrant_util import get_vector_db
from notion_load.qdrant_util import get_qdrant_client
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

prompt_template = """SYSTEM: You are an AI chatbot with knowledge of SAP Commerce Cloud, also known as Hybris and can answer all questions.
---
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def qa(query,
       model_name,
       temperature,
       k,
       search_type,
       history,
       verbose
       ) -> dict[str, Any]:
    """Get Qdrant client"""
    q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

    """Qdrant Vector DB"""
    embeddings = OpenAIEmbeddings()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    vectors = get_vector_db(q_client, collection_name, embeddings)
    retriever = vectors.as_retriever(search_type=search_type, search_kwargs={'k': k})

    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    streaming_llm = ChatOpenAI(streaming=True, model_name=model_name, callbacks=[StreamingStdOutCallbackHandler()], temperature=temperature)

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=verbose
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
        verbose=verbose,
    )

    qa = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        verbose=verbose,
        return_source_documents=True,
    )

    result = qa({"question": query, "chat_history": history})

    st.session_state["chat_sources"] = result['source_documents']
    return result["answer"]
