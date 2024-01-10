# To execute this file, you will need to install the following packages:
# pip install openai pandas cassandra-driver cassio langchain PyMuPDF chainlit tiktoken
#
# This also assumes that you have already created an Astra DB database and a keyspace.
# Also, we assume that you have already ingested the Astra DB security white paper PDF file into the database as embeddings.
# See the comments in the code below for more details.
#
# After that, you will need to set the following environment variables:
# - ASTRA_DB_SECURE_BUNDLE_PATH
# - ASTRA_DB_APPLICATION_TOKEN
# - ASTRA_DB_KEYSPACE
# You can get the values for these environment variables from the Astra DB console.
#
# - OPENAI_API_KEY
# This is the API key for OpenAI. You can get it from the OpenAI console.
#
# Once you have set the environment variables, you can run this file using the following command:
#
# chainlit run PDFBasedChat.py -w
#


from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
from cassandra.query import SimpleStatement

from collections import namedtuple

import chainlit as cl
import os

def setup_env():
    exitflag = 0
    notset = ""
    Env = namedtuple("Env", ["scb", "token", "keyspace", "openaikey"])
    scb = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    openaikey = os.getenv("OPENAI_API_KEY")

    if scb == None:
        exitflag = exitflag + 1
        notset = "ASTRA_DB_SECURE_BUNDLE_PATH\n"
    if token == None:
        exitflag = exitflag + 1
        notset = notset + "ASTRA_DB_APPLICATION_TOKEN\n"
    if keyspace == None:
        exitflag = exitflag + 1
        notset = notset + "ASTRA_DB_KEYSPACE\n"
    if openaikey == None:
        exitflag = exitflag + 1
        notset = notset + "OPENAI_API_KEY\n"
    
    if exitflag == 0:
        print("All environment variables set.")
    else:
        print(f"One or more environment variables not set:\n{notset}")
    
    return Env(scb, token, keyspace, openaikey)
 
def get_session(scb: str, token: str) -> Session:
    cluster = Cluster(
        cloud={
            "secure_connect_bundle": scb,
        },
        auth_provider=PlainTextAuthProvider("token", token),
    )
    return cluster.connect()

env = setup_env()
session = get_session(env.scb, env.token)

embeddings = OpenAIEmbeddings()
table_name = "pdftexttable"

documents =""

"""
# IF YOU WANT TO INGEST A PDF FILE, UNCOMMENT THE FOLLOWING LINES AND REPLACE THE FILE PATH
# THE PDF FILE IS AVAILABLE FROM HERE: https://www.datastax.com/resources/whitepaper/astra-security

pdffilepath = "FILE PATH"
loader = PyMuPDFLoader(pdffilepath)
documents = loader.load()
"""

pdftextsearchdb = Cassandra.from_documents(
    documents=documents,
    embedding=embeddings,
    table_name=table_name,
    keyspace=env.keyspace,
    session=session,
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
 
@cl.on_message
async def main(message: cl.Message):
            
            docs = pdftextsearchdb.similarity_search(message.content, k=3)
            
            supporting_text = ""
            for doc in docs:
                supporting_text = supporting_text + "\n" + doc.page_content

            chat_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful sales person. You are helping a customer with a question. The customer asks you a question. You answer the question."),
                ("system", "An assistant will provide you with some supporting text. You will have to answer the question based on the supporting text."),
                ("system", "If the assistant does not provide you with relevant supporting text, you can ask the customer to rephrase the question."),
                ("assistant", "he following are some supporting text: {assistant_supporting_text}"),
                ("human", "Hi, I have a question. {customer_question}"),
            ])

            messages = chat_template.format_messages(
                assistant_supporting_text = supporting_text,
                customer_question = message.content,
            )

            response = llm(messages)
            elements = [
                 cl.Text(name="Answer", content=response.content, display="inline")
            ]

            await cl.Message(author="Assistant", content="Here is the answer to your question:", elements=elements).send()


@cl.on_chat_start
async def start():
     await cl.Message(author="Assistant", content="Hello ! Ask any question about Astra DB security.\n\nMake sure that you have already ingested the Astra DB security white paper available from here:\nhttps://www.datastax.com/resources/whitepaper/astra-security \n\nIf not, uncomment the PDF ingestion code in the source of this app and start from there.").send()