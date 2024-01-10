# chainlit run pinecone-RAG.py -w

import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document

import pinecone
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('../.env')) # read local .env file

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

# initialize pinecone client and embeddings
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)

welcome_message = """
Welcome to the Lab Protocol QA demo! 
I can answer questions about any lab protocols that have been uploaded
"""

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            # source_name = f"source_{source_idx}"
            source_name = source_doc.metadata['source']
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [f"{text_el.name}\n" for text_el in text_elements]

        if source_names:
            answer += f"\nSources:\n * {'* '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()