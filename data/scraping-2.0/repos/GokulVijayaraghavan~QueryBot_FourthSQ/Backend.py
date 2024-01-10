import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS,cross_origin

from pymongo import MongoClient

from langchain import HuggingFaceHub, embeddings, LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ( ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
app.secret_key = "Key"
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=3000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def find_docs(db, query):
    num = 3
    matching_docs = db.similarity_search(query, num)
    print(matching_docs)
    return matching_docs

def save_query_answer(query, answer):
    query_answer_data = {"query": query, "answer": answer}
    collection.insert_one(query_answer_data)

def answer(query,matching_docs):
    template = """
        You are a helpful assistant that that can answer questions
        based on :  {docs}
        Only use the factual information from the transcript to answer the question.
        Your answers should be verbose and detailed.
        """
    print(template)
    docs_page_content = " ".join([d.page_content for d in matching_docs])

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


@app.route('/api/query', methods=['POST'])
def process_query():
    query = request.json['query']
    matching_docs = find_docs(db, query)
    response, docs = answer(query, matching_docs)
    print(response)
    save_query_answer(query, response)
    print("returning answer")
    return(jsonify({"answer": response}))
    
@app.route("/signup", methods=["POST"])
@cross_origin()
def signup():
    username = request.json.get("username")
    password = request.json.get("password")

    # Check if the username already exists in the database
    existing_user = users.find_one({"username": username})
    if existing_user:
        return jsonify({"message": "Username already taken. Please try a new username."}), 400

    user_data = {"username": username, "password": password}
    users.insert_one(user_data)
    return jsonify({"message": "Signup successful"})

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    user_data = users.find_one({"username": username})
    if user_data and user_data["password"] == password:
        session["username"] = username
        return jsonify({"message": f"Welcome back, {username}!"})
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route('/checkLoggedIn')
@cross_origin()
def check_logged_in():
    username = session.get('username')
    if username:
        return jsonify({'isLoggedIn': True, 'username': username})
    else:
        return jsonify({'isLoggedIn': False})


@app.route('/logout', methods=['POST'])
@cross_origin()
def logout():
    session.pop('username', None)
    return jsonify({'isLoggedIn': False, 'message': 'Logout successful'})

if __name__ == "__main__":
    directory = r'C:\Users\gokul\OneDrive\Desktop\Fourth Square\Interview 2\Docs'
    url = "mongodb+srv://gvijaya5:task2@cluster0.kkkewow.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(url)
    db = client['QA']
    collection = db['QA']
    users = db["users"]
    chat = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":1})
    embedding = embeddings.HuggingFaceHubEmbeddings()
    documents = load_docs(directory)
    docs = split_docs(documents)
    db = Chroma.from_documents(docs, embedding)
    app.run(debug=True)
