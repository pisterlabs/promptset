import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain.retrievers import AzureCognitiveSearchRetriever
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()

chat_model_deployment = os.getenv("OPENAI_CHAT_MODEL")

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)
@cl.on_chat_start
async def on_chat_start():
    message_history = ChatMessageHistory()

    prompt_template = """You are a helpful and friendly professor at Pennsylvania State University. You are an expert at understanding the course details for the Artificial Intelligence Program at Pennsylvania State University.
    Your job is to assist students in answering any questions they may have about the courses within the program.

    {context}

    Question: {question}
    Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=10)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # Ensure message.content is a string
    if not isinstance(message.content, str):
        # Handle non-string content appropriately
        # For example, convert to string or log an error
        message_content = str(message.content)
    else:
        message_content = message.content

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    # Use message_content instead of message
    res = await chain.acall(message_content, callbacks=[cb])
    answer = res["answer"]

    text_elements = []


    await cl.Message(content=answer, elements=text_elements).send()
