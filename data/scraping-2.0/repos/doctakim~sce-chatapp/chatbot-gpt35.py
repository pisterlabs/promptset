"""
chatbot-gpt35.py is a gpt powered chatbot that uses the OpenAI API to answer questions.
If an answer to the user query is not found in the vectordb, the chatbot will use the OpenAI API to answer the question.
"""

# Import libraries
import pickle
import faiss
import openai
import streamlit as st

# Environment variables for OpenAI API
import os
os.environ['OPENAI_API_TYPE'] = 'TODO'
os.environ['OPENAI_API_VERSION'] = 'TODO' 
os.environ['OPENAI_API_BASE'] = 'TODO'
os.environ['OPENAI_API_KEY'] = 'TODO'

# Langchain imports
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Import scraped docs from docs.py
from docs import docs

def split_docs():
    """
    Splits the documents in `docs` into chunks of 1000 characters with 200 characters of overlap
    """
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap  = 200,
      # length_function = len,
      # add_start_index = True,
    )

    docs_split = text_splitter.split_documents(docs)
    return docs_split

def create_vectordb():
  """
  Embeds each document in `docs_split` as a vector and stores the resulting vectors in a vectordb
  """
  docs_split = split_docs()

  embeddings = OpenAIEmbeddings(deployment="TODO", model='TODO', chunk_size=1)
  # Iterates through all chunk stored in `docs_split` and embeds each one as a vector using `embeddings`
  vectordb = FAISS.from_documents(docs_split, embeddings)

  # Persist the vector store for later reuse
  with open("faiss_store_openai.pkl", "wb") as f:
      pickle.dump(vectordb, f)

def load_vectordb():
  """
  Loads the vectordb from the pickle file
  """
  create_vectordb()

  with open("faiss_store_openai.pkl", "rb") as f:
      vectordb = pickle.load(f)
  
  return vectordb

# Load the vector store
vectordb = load_vectordb()

# Define the LLM model
llm = AzureOpenAI(
    deployment_name='TODO',
    model_name='TODO',
    temperature=0
  )

def retrieval_answer(query):
    """
    Returns the answer and source to a question using the vectordb
    """
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectordb.as_retriever())
    question = query
    response = chain({"question": f"{question}"}, return_only_outputs=True)
    answer = response["answer"]
    source = response["sources"]
    return answer, source

# Set up the Streamlit app
st.title("Sparky ⚡️ ", anchor=False)
st.markdown("""### *SCE's :red[GPT-powered] chatbot*""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar= "👱" if message['role'] == "user" else "⚡️"):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Query vectordb before defaulting to GPT 3.5 model
        answer, source = retrieval_answer(prompt)
        # If the answer is "I don't know", use GPT 3.5 model
        if answer.strip() == "I don't know.":
          for response in openai.ChatCompletion.create(
              engine='chat',
              messages=[
                  {"role": m["role"], "content": m["content"]}
                  for m in st.session_state.messages
              ],
              stream=True,
          ):
              full_response += response.choices[0].delta.get("content", "")
              message_placeholder.markdown(full_response + "▌")
        # Otherwise, use the answer from the vectordb
        else:
          full_response = f"{answer} \n\nSource: {source}"
          
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})