import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pprint import pprint
import streamlit as st

def create_dataset(list_of_documents: list) -> pd.DataFrame:
    """
    create a Pandas DataFrame of trade register documents.
    """
    data = []
    for name in tqdm(list_of_documents, desc="Documents"):
        content = Path(f'data/{name}').read_text()
        d = {
            "content": content,
            "title": name
        }
        data.append(d)
    df = pd.DataFrame(data)

    return df

list_of_documents = [
    'chronological-excerpt.txt',
    'registration.txt',
    'shareholder-list.txt'
]
df = create_dataset(list_of_documents)
df.to_csv("./data/dataset.csv", index=False)

def load_dataset(dataset_name:str="dataset.csv") -> pd.DataFrame:
    """
    Load dataset from file_path

    Args:
        dataset_name (str, optional): Dataset name. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: Dataset
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df

def create_chunks(dataset:pd.DataFrame, chunk_size:int, chunk_overlap:int) -> list:
    """
    Create chunks from the dataset

    Args:
        dataset (pd.DataFrame): Dataset
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap

    Returns:
        list: List of chunks
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="content"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )
    # aggiungiamo i metadati ai chunk stessi per facilitare il lavoro di recupero
    for doc in text_chunks:
        title = doc.metadata["title"]
        content = doc.page_content
        final_content = f"TITLE: {title}\DESCRIPTION: {title}\BODY: {content}"
        doc.page_content = final_content

    return text_chunks

def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Create or get vector store

    Args:
        chunks (list): List of chunks

    Returns:
        FAISS: Vector store
    """

    embeddings = HuggingFaceInstructEmbeddings()

    if not os.path.exists("./db"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(
            chunks, embeddings
        )
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)
    print(vectorstore)
    return vectorstore

def get_conversation_chain(vector_store:FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Get the chatbot conversation chain

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta") # if you want to use open source LLMs
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def handle_style_and_responses(user_question: str) -> None:
    """
    Handle user input to create the chatbot conversation in Streamlit

    Args:
        user_question (str): User question
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    human_style = "background-color: #e6f7ff; border-radius: 10px; padding: 10px;"
    chatbot_style = "background-color: #f9f9f9; border-radius: 10px; padding: 10px;"

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(
                f"<p style='text-align: right;'><b>User</b></p> <p style='text-align: right;{human_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align: left;'><b>Chatbot</b></p> <p style='text-align: left;{chatbot_style}'> <i>{message.content}</i> </p>",
                unsafe_allow_html=True,
            )
   
load_dotenv() # need to create a file called .env in the root of the working folder and insert our Hugging Face API key
df = load_dataset()
chunks = create_chunks(df, 1000, 0)
system_message_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are a chatbot tasked with responding to questions about the Ticos Systems company.

    You should never answer a question with a question, and you should always respond with the most relevant page from documents.

    Do not answer questions that are not about the Ticos Systems company.

    Given a question, you should respond with the most relevant documents page by following the relevant context below:\n
    {context}
    """
)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = create_or_get_vector_store(chunks)
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
st.set_page_config(
    page_title="Company Register Documents Chatbot",
    page_icon=":books:",
)

st.title("Company Register Documents Chatbot")
st.subheader("Chat with all information about Ticos Systems Company!")
st.markdown(
    """
    This chatbot was created to answer questions about the Ticos Systems Company.
    Ask a question and the chatbot will respond with the most relevant page of documents.
    """
)
# Image from Alexandra Koch on pixabay
st.image("https://cdn.pixabay.com/photo/2023/01/15/17/19/robot-7720755_1280.jpg") 

user_question = st.text_input("Ask your question")
with st.spinner("Processing..."):
    if user_question:
        handle_style_and_responses(user_question)

st.session_state.conversation = get_conversation_chain(
    st.session_state.vector_store, system_message_prompt, human_message_prompt
)