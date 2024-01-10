# PATCH sqlite3 to use pysqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import chromadb
import dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.vectorstores.chroma import Chroma

import streamlit as st

dotenv.load_dotenv()


class TextExtractOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip()


def query_psylio_agent(query: str):
    chroma_client = chromadb.PersistentClient("chromadb_psylio_kb")

    embedding_function = OpenAIEmbeddings()
    chroma = Chroma(
        client=chroma_client, collection_name="psylio", embedding_function=embedding_function
    )
    retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.75})

    chat = ChatOpenAI(verbose=True, model_name="gpt-4-1106-preview", temperature=0)

    # system_template = "Tu es un assistant francophone de service-client pour la plateforme Psylio, un outil de tenue de dossier pour spécialistes en santé mentale. Tu réponds à des requêtes courriel (email) de client en utilisant uniquement le contexte suivant: {context}"
    system_template = """
        You are a help desk assistant for Psylio, a recordkeeping and client management platform for mental health specialists.
        You respond in the same language as the user request (french or english) in email format. 
        If the answer cannot be found in the context, write "I could not find the answer." 
        You provide email responses using only the following context: {context}
        """

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{question}"),
        ])

    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": chat_prompt})

    output = qa({"query": query})
    documents_used = output["source_documents"]

    return output["result"], documents_used


st.title("Psylio Service-Client Helper")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Salut! Je suis un assistant francophone de service-client pour la plateforme Psylio. Comment puis-je t'aider?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    response, documents_used = query_psylio_agent(query)

    st.session_state.messages.append({"role": "assistant", "content": response})
    message = st.chat_message("assistant")
    message.write(response)
    message.write("Documents utilisés:")
    for document in documents_used:
        message.write(f"- [{document.metadata['title']}]({document.metadata['url']})")
