import os

import streamlit as st
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.vectorstores.chroma import Chroma

load_dotenv()

st.title("🧢 Hummingbot Helper")

# Set OpenAI API key from Streamlit secrets
openai_model = st.sidebar.selectbox("Select a model", ["gpt-3.5-turbo", "gpt-3.5", "gpt-4"], index=2)

# Replace with persist_directory location from 01_load_hummingbot_docs.ipnynb
persist_directory = os.environ.get("PERSIST_DIRECTORY", "/tmp")

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    collection_name="hummingbot_documentation",
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Build prompt
template = """Use the following pieces of context that are part of the Hummingbot Documentation to answer the user question. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as professional and friendly. 
Try to answer the question using as much as possible the context and then expand in bullet points suggestions to the user and possible questions that the context can answer. 
If you think that more info will be needed suggest at the end of the answer and also provide useful links to review the information.
{context}
Also you can use information from the previous messages in the conversation {conversation}.
Question: {question}
Helpful Answer:"""

prompt_template = ChatPromptTemplate.from_template(template=template)
model = ChatOpenAI(temperature=0, model_name=openai_model, streaming=True)
output_parser = StrOutputParser()

retriever = vectordb.as_retriever(search_type="mmr", k=5, fetch_k=10)


def get_documents_avoiding_pdfs(documents):
    return [doc for doc in documents if not doc.metadata['url'].endswith(".pdf")]


chain = RunnableMap({
    "context": lambda x: get_documents_avoiding_pdfs(retriever.get_relevant_documents(x["question"])),
    "question": lambda x: x["question"],
    "conversation": lambda x: x["conversation"]
}) | prompt_template | model | output_parser

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in chain.stream({"question": prompt,
                                      "conversation": " ".join([message["content"] for message in st.session_state.messages if message["role"] == "user"])}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})