'''from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
import openai 
import PyPDF2
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import dotenv

openai.api_key = "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"

app = Flask(__name__)
app.secret_key = 'your secret key'  # Replace with your secret key

def get_pdf_text(filename):
    text = ""
    with open(filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()        
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

def get_vectorstore(chunks):
    os.environ["OPENAI_API_KEY"]="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"
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/api/chat', methods=['POST'])
def chat():
    message = request.json['message']
    chat_models = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=chat_models,
        messages=messages
    )
    return jsonify(message=response['choices'][0]['message']['content'])

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part in the request'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No file selected for uploading'), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(filename)

        text = get_pdf_text(filename)
        chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(chunks)
        session["conversation"] = get_conversation_chain(vectorstore)
        os.remove(filename)
        return jsonify(success=True)
    else:
        return jsonify(error='Allowed file type is pdf'), 400

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)'''



from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import openai 
import PyPDF2
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

openai.api_key = "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"

app = Flask(__name__)
app.secret_key = 'your secret key'  # Replace with your secret key

conversation_chains = {}

def get_pdf_text(filename):
    text = ""
    with open(filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()        
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

def get_vectorstore(chunks):
    os.environ["OPENAI_API_KEY"]="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"
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/api/chat_without_file', methods=['POST'])
def chat_without_file():
    if 'message' not in request.json:
        return jsonify(error='No message in the request'), 400
    message = request.json['message']  
    print(message)                                  # this is assumed to be included in the POST data
    chat_models = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=chat_models,
        messages=messages
    )
    return jsonify(message=response['choices'][0]['message']['content'])

@app.route('/api/upload_without_message', methods=['POST'])
def upload_without_message():
    if 'file' not in request.files:
        return jsonify(error='No file part in the request'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No file selected for uploading'), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(filename)

        text = get_pdf_text(filename)
        chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(chunks)
        conversation_chains = get_conversation_chain(vectorstore)
        query = "Write a summary of the  uploaded file"
        response = conversation_chains({"question": query}) # you'll have to adapt this line to how the method is actually called
        print(response)
        os.remove(filename)
        return jsonify(message = response['answer'])
    else:
        return jsonify(error='Allowed file type is pdf'), 400
    

@app.route('/api/chat_with_file', methods=['POST'])
def chat_with_file():
    if 'file' not in request.files or 'message' not in request.form:
        return jsonify(error='No file or message part in the request'), 400
    file = request.files['file']
    message = request.form['message']
    #filename = request.json['filename']  # this is assumed to be included in the POST data

    filename = secure_filename(file.filename)
    file.save(filename)

    text = get_pdf_text(filename)
    chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(chunks)
    conversation_chains = get_conversation_chain(vectorstore)
    query = message
    print(message)
    response = conversation_chains({"question": query}) # you'll have to adapt this line to how the method is actually called
    print(response)
    os.remove(filename)
    return jsonify(message = response['answer'])

    
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/style.css")
def styles():
    return send_from_directory("static", "style.css")    

if __name__ == '__main__':
    app.run(debug=True, port=3000)



