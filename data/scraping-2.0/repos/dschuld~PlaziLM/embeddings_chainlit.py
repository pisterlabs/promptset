
import os
import openai
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chat_models import ChatOpenAI as OpenAI
from langchain.chains.question_answering import load_qa_chain
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import OpenAI
import chainlit as cl

openai.api_key = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment='gcp-starter',
)

@cl.on_chat_start
def main():
    llm = OpenAI(temperature=0.5, model_name="gpt-4-1106-preview")

    chain = load_qa_chain(llm, chain_type="stuff", verbose = True)
    cl.user_session.set("llm_chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    print("Received message: " + message.content)
    embeddings = retrieve_embeddings(message.content)
    chain = cl.user_session.get("llm_chain")
    
    res = await chain.arun(
        question=message.content + ". Beantworte die Frage auf Deutschm, nicht auf Englisch.", input_documents = embeddings, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=res).send()



def load_docx_file(file_path):
    loader = Docx2txtLoader(file_path)
    return loader.load()


def retrieve_embeddings(question):
    index_name = "plazi-lm-index"
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "Erstelle eine geographische Übersicht des Lebens des Vaters des Autors. Wo wurde er geboren? Welche Orte hat er Zeit seines Lebens besucht? Gehe chronologisch vor."

    docs = docsearch.similarity_search(question, k = 25)

    return docs


def init_data():
    data = load_docx_file("docs/Teil1-50Seiten.docx")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(data)

    openai.api_key = os.environ['OPENAI_API_KEY']
