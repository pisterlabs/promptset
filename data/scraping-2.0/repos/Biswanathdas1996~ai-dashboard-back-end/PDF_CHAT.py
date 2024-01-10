from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.llms import AzureOpenAI
from flask import Flask, request, jsonify, Response, send_file

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://soumenopenai.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "3a5a6eba4d2546558d3fa749ef9fb5ce"
os.environ["deployment_name"] = "gpt-35-turbo"


# Global variables to store state
conversation = None
MAX_CHUNKS = 16


conversation = None
MAX_CHUNKS = 16
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


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
    avg_chunk_size = len(text) // MAX_CHUNKS
    chunk_size = max(500, avg_chunk_size)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Combine chunks if there are more than MAX_CHUNKS
    while len(chunks) > MAX_CHUNKS:
        smallest_chunk = min(chunks, key=len)
        chunks.remove(smallest_chunk)
        next_smallest_chunk = min(chunks, key=len)
        chunks.remove(next_smallest_chunk)

        combined_chunk = smallest_chunk + next_smallest_chunk
        chunks.append(combined_chunk)

    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        return jsonify({"error": "The text couldn't be split into chunks."}), 400
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
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


def upload(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        if pdf and allowed_file(pdf.filename):
            # Process the saved file
            chunk_text = get_pdf_text(pdf_docs)
            raw_text += chunk_text
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks)
    global conversation
    conversation = get_conversation_chain(vectorstore)

    return "PDFs processed successfully"


def uploadText(docs):
    raw_text = docs
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks)
    global conversation
    conversation = get_conversation_chain(vectorstore)

    return "PDFs processed successfully"


def process(user_question, config):
    global conversation
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = config['OPENAI_API_VERSION']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']
    os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    messages = ""
    for i, message in enumerate(chat_history):
        messages = message.content
    return messages
