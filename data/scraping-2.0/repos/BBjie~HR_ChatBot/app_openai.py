from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import time
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import config 
import glob


app = Flask(__name__)


openai.api_key = config.API_KEY


# Get list of all PDF files in the directory
pdf_files = glob.glob('data/**/*.pdf', recursive=True)
pdf_files.sort(key=os.path.getmtime, reverse=True)

all_data = [PyPDFLoader(file).load() for file in pdf_files]

# Store data from the most recent file separately
recent_data = all_data[0]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Process all data
all_docs = []
for data in all_data:
    for page in data:
        texts = text_splitter.split_text(page.page_content)
        docs = [Document(page_content=t) for t in texts]
        all_docs.extend(docs)

# Process recent data
recent_docs = []
for page in recent_data:
    texts = text_splitter.split_text(page.page_content)
    docs = [Document(page_content=t) for t in texts]
    recent_docs.extend(docs)

# When retrieving information, first check recent_docs, then check all_docs
def retrieve_information(query):
    # Check recent_docs
    for doc in recent_docs:
        if query in doc.page_content:
            return doc

    # If not found, check all_docs
    for doc in all_docs:
        if query in doc.page_content:
            return doc

    return None



# Create embeddings and vector store
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding)

# Set up the RetrievalQA Chain
system_template = """Context: {context}
Use the following context to answer the users question.
If you don't know the answer, just say that you don't know.

Begin!
----------------
Question: {question}
Helpful Answer:"""

# Update your messages to include the system template with context
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

llm = ChatOpenAI(
    **config.openai_configs
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)



# Initialize the QA chain outside of the route to avoid recreating it on each request
chain_type_kwargs = {"prompt": prompt}
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Ensure this is correctly configured
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    return get_Chat_response(msg)

def get_Chat_response(text):
    try:
        # Run the QA chain with the user's text
        response = qa_chain.run({"query": text})
        answer = response.get('result') if isinstance(response, dict) else response
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in get_Chat_response: {e}")
        return jsonify({"answer": "Sorry, I couldn't process that request."})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)