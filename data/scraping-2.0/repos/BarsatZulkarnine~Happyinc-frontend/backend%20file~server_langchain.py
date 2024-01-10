from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import os
os.environ["OPENAI_API_KEY"] = ''

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/process_data": {"origins": "http://localhost:3000"}})

pdf_directory = r'C:\Users\Barsat\Desktop\Project\backend\Chat with multiple pdf\assets'

def process_single_pdf(pdf_path):
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)
    return conversation

# Modify the get_pdf_text function to take a single file path
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks (text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

    # ... same as in your code ...

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        user_message = data.get('user_message')
        pdf_path = os.path.join(pdf_directory, "Community_Database_-_Sheet1_1.pdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF path not found: {pdf_path}")

        conversation = process_single_pdf(pdf_path)
        response = conversation({'question': user_message})
        chat_history = response.get('chat_history', [])

        messages = []
        for msg in chat_history:
            role = msg.role if hasattr(msg, 'role') else 'Unknown'
            content = msg.content if hasattr(msg, 'content') else 'No content'
            messages.append(f"{role}: {content}")

        return jsonify({"response": messages})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)



if __name__ == '__main__':
    app.run(debug=True)
