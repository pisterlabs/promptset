from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os
import tempfile
import threading

app = Flask(__name__)

# Define global variables
history = []
generated = [""]

def initialize_session_state():
    global history
    global generated
    if 'history' not in history:
        history = []

    if 'generated' not in generated:
        generated = [""]

@app.route('/initialize', methods=['POST'])
def initialize():
    initialize_session_state()
    return "Session initialized."

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['input']
    global history
    global generated

    if user_input:
        output = conversation_chat(user_input, chain, history)
        history.append(user_input)
        generated.append(output)

    return jsonify({'user_input': user_input, 'generated_output': output})

def create_conversational_chain(vector_store):
    # Create llm
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
        callbacks=None,
        
        input={
            "temperature": 0.1,
            "max_length": 1000,
            "top_p": 1,
            "system_prompt": """<s>[INST] <<SYS>>
            You are a helpful, respectful and honest university assistant called Jane. Always answer as
            helpfully as possible, while being safe. Your answers should not include
            any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
            Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain
            why instead of answering something not correct. If you don't know the answer
            to a question, please don't share false information.

            Your goal is to provide answers relating to university or universities, admission and other campus life you can be creative to provide additional relevant answers only where applicable.
            The document/documents have information of various universities and not specific to one university, your goal is to pick the most relevant information that the user want to know, do not make up any information.
            <</SYS>>
            """
        })

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)

    return chain

if __name__ == "__main__":
    # Initialize session state
    initialize_session_state()

    # Create the chain object
    text = []  # Replace this with your text data
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    app.run(debug=True)
