from flask import Flask, render_template, request
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv

# load env vars
load_dotenv()

# OpenAI embeddings
embeddings = OpenAIEmbeddings(client=None)

# load FAISS db
db = FAISS.load_local("faiss_db", embeddings)

# memory for chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# llm
llm = OpenAI(temperature=0, client=None)

# init retrieval chain
qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    question = request.form.get('question')
    print(question)
    result = qa({"question": question})
    return result["answer"]

if __name__ == '__main__':
    app.run(debug=True)