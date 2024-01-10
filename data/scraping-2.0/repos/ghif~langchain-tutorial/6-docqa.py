from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

import chainlit as cl

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())

pdf_filepath = "Peraturan Menteri Pendidikan dan Kebudayaan Nomor 14 Tahun 2020.pdf"

# PDF parser
loader = PyPDFLoader(pdf_filepath)
pages = loader.load_and_split()

# Create vector store DB
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

# Initialize chat
@cl.on_chat_start
def init():
    """
    Model
    """
    chat_llm = ChatOpenAI(
        temperature=0.3,
        streaming=True
    )

    """
    Chain
    """
    # prompt = ChatPromptTemplate.from_template(
    # """ 
    # You are a friendly helpdesk agent for Ministry of Eduction, Research, and Technology. 
    # Always answer in Indonesian.
    # """
    # )

    # chain = LLMChain(
    #     llm=chat_llm,
    #     prompt=prompt,
    #     verbose=True
    # )
    
    chain = RetrievalQA.from_llm(
        llm=chat_llm,
        retriever=db.as_retriever()
    )
        
    cl.user_session.set("chain", chain)
    

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # Infer from the chain
    outputs = await chain.acall(
        message,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # Post-processing (if any)
    res = outputs["result"]

    # Send the response 
    await cl.Message(
        content=res
    ).send()