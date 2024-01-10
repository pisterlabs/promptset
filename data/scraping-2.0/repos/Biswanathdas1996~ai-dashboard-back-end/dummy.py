from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.llms import AzureOpenAI


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://soumenopenai.openai.azure.com"
os.environ["OPENAI_API_KEY"] = "3a5a6eba4d2546558d3fa749ef9fb5ce"
os.environ["deployment_name"] = "gpt-35-turbo"

app = Flask(__name__)


# Global variables to store state
conversation = None
MAX_CHUNKS = 16


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
    print("==============>", chunks)
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


@app.route('/upload', methods=['POST'])
def upload():
    pdf_docs = request.files.getlist('pdf_docs')
    global conversation
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    return jsonify({"message": "PDFs processed successfully"})


@app.route('/process', methods=['POST'])
def process():
    user_question = request.json.get('user_question')
    global conversation

    response = conversation({'question': user_question})
    chat_history = response['chat_history']

    messages = []
    for i, message in enumerate(chat_history):
        messages.append(message.content)

    return jsonify({"messages": messages})
    # except openai.error.InvalidRequestError as e:
    #     return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
