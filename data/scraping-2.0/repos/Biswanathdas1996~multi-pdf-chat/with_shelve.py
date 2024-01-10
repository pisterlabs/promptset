from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.llms import AzureOpenAI
import pinecone

import shelve

# Constants for shelve database
SHELVE_DB = "app_data.db"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://soumenopenai.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "3a5a6eba4d2546558d3fa749ef9fb5ce"
os.environ["deployment_name"] = "gpt-35-turbo"

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Global variables to store state

embeddings = None
text_chunks = None

collection_name = 'embeddings'
index = pinecone.Index(collection_name)
dimension = 18432


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            try:
                # Try decrypting with an empty password
                pdf_reader.decrypt('')
            except:
                # If decryption fails, you can either skip the file or ask the user for a password
                continue

        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = AzureOpenAI(deployment_name='gpt-35-turbo',
                      model_name="gpt-35-turbo",)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@app.route('/upload', methods=['POST'])
def upload():
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    embeddings = OpenAIEmbeddings()

    # Save text_chunks and embeddings to shelve database
    with shelve.open(SHELVE_DB) as db:
        db['text_chunks'] = text_chunks
        db['embeddings'] = embeddings

    return jsonify({"message": "PDFs processed successfully"})


@app.route('/process', methods=['POST'])
def process():
    # Retrieve text_chunks and embeddings from shelve database
    with shelve.open(SHELVE_DB) as db:
        text_chunks = db.get('text_chunks', [])
        embeddings = db.get('embeddings', None)

    user_question = request.json.get('user_question')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    conversations = get_conversation_chain(vectorstore)
    response = conversations({'question': user_question})
    chat_history = response['chat_history']
    messages = []
    for i, message in enumerate(chat_history):
        messages.append(message.content)
    return jsonify({"messages": messages})


if __name__ == '__main__':
    app.run(debug=True)
