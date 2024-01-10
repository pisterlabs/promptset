
#Install necessary libraries
import streamlit as st

#Get api_key from env
import os
api_key = st.secrets["API_KEY"]

from langchain.document_loaders import UnstructuredURLLoader

urls = [
    "https://web.archive.org/web/20230617044915/https://consmendoza.esteri.it/consolato_mendoza/it/la_comunicazione/dal_consolato/2023/06/prenot-mi-appuntamenti-anagrafe_14.html",
    "https://web.archive.org/web/20221208015326/https://consmendoza.esteri.it/consolato_mendoza/it/i_servizi/per_i_cittadini/cittadinanza/ciudania-matrimonio-req.html",
    "https://web.archive.org/web/20230619195057/https://consmendoza.esteri.it/consolato_mendoza/resource/doc/2021/06/guida_citt_x_mat.pdf",
    "https://web.archive.org/web/20230601000000*/https://consmendoza.esteri.it/consolato_mendoza/es/i_servizi/per_i_cittadini/cittadinanza/ciudadania-matrimonio-doc.html",
    "https://web.archive.org/web/20230308222154/https://consmendoza.esteri.it/consolato_mendoza/it",
    "https://docs.google.com/document/d/e/2PACX-1vT8qHba7oGpVWg8FJvToQMgJBGpEib3xyKLHvei_7S2i3Gi5PyFtU6SS7z47AgJQEaNqS0EubNpVYkx/pub",
    ]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

# Import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split the documents
documents = splitter.split_documents(data)

# Import tiktoken
import tiktoken

# Create an encoder
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

# Count tokens in each document
doc_tokens = [len(encoder.encode(doc.page_content)) for doc in documents]

# Calculate the sum of all token counts
total_tokens = sum(doc_tokens)

# Calculate a cost estimate
cost = (total_tokens/1000) * 0.0004
print(f"Total tokens: {total_tokens} - cost: ${cost:.2f}")

# Import chroma
from langchain.vectorstores import Chroma

# Import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

# Create the mebedding function
embedding_function = OpenAIEmbeddings(openai_api_key= api_key)

# Create a database from the documents and embedding function
db = Chroma.from_documents(documents=documents, embedding=embedding_function, persist_directory="my-embeddings")

# Persist the data to disk
db.persist()

# Import
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI

# Set the question variable
question = "¿Que pasos debo realizar para pedir la ciudadania por matrimonio?"
#question = "¿Cual es el handle de twitter del consulado de italia en mendoza?"
#question = "¿Quando sono i proximi appuntamenti per cittadinanza del consolato d' italia en mendoza?"
#question = "Que pasos debo realizar  para pedir la ciudadania por matrimonio? ¿Puedes ponerlo en una lista en viñetas?"
#question = "Que requisitos debo cumplir  para pedir la ciudadania por matrimonio? ¿Puedes ponerlo en una lista en viñetas?"
#question = "¿cuanto demora el certificado de antecedentes penales?"


# Query the database as store the results as `context_docs`
context_docs = db.similarity_search(question)

# Create a prompt with 2 variables: `context` and `question`

prompt = PromptTemplate(
    template=""""Usa los siguientes elementos de contexto para responder a la pregunta al final. Si no conoces la respuesta, di que no lo sabes,no inventes una respuesta.

<context>
{context}
</context>

Domanda: {question}
Respuesta:""",
    input_variables=["context", "question"]
)

# Create an LLM with ChatOpenAI
llm = ChatOpenAI(openai_api_key=api_key,temperature=0)

# Create the chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Call the chain
result = qa_chain({
    "question": question,
    "context": "\n".join([doc.page_content for doc in context_docs])
})

# Print the result
print(result["text"])

header = st.container()
features = st.container()


with header:
    st.title('Welcome to the Q&A bot for Procedures at the Italian Consulate in Mendoza')
    st.title('Bienvenido al Bot para preguntas sobre trámites en el consulado de Mendoza')


with features:
    st.header('Pregunta sobre un trámite (ask a question):')
    
    question = st.text_input("Pregunta sobre un tramite (ask a question):", "¿Que pasos debo realizar  para pedir la ciudadania por matrimonio?")
    context_docs = db.similarity_search(question)
    llm = ChatOpenAI(openai_api_key=api_key,temperature=0)
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    result = qa_chain({
    "question": question,
    "context": "\n".join([doc.page_content for doc in context_docs])
    })
    
    st.write(result["text"])

