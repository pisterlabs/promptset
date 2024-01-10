import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma


load_dotenv()

app = Flask(__name__)

# Your existing code here
embeddings = OpenAIEmbeddings()

if not os.path.exists("./chromadb"):
    loader = PyPDFLoader(os.environ["DOCUMENT_PATH"])
    documents = loader.load_and_split()
    docsearch = Chroma.from_documents(documents, embeddings, persist_directory="./chromadb")
else:
    docsearch = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.0),
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50}),
                                    return_source_documents=True)

# Move the code block inside a route function
@app.route('/', methods=['GET', 'POST'])
def qa_endpoint():
    if request.method == 'POST':
        # Get the input question from the request
        question = request.form['question']

        # Use the QA model to get the answer
        answer = qa(question)

    	# answer['source_documents']

        # Return the answer as a JSON response
        return jsonify({'answer': answer['result']})
    else:
        # Render the HTML template with the input field and output
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
