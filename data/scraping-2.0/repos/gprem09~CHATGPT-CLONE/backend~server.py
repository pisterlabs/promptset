from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import pymysql
from db_config import *

def get_db_connection():
    return pymysql.connect(
        host=MYSQL_DATABASE_HOST,
        user=MYSQL_DATABASE_USER,
        password=MYSQL_DATABASE_PASSWORD,
        db=MYSQL_DATABASE_DB,
        cursorclass=pymysql.cursors.DictCursor
    )


openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key is not set.")

os.environ["OPENAI_API_KEY"] = openai_api_key

app = Flask(__name__)
CORS(app)

PERSIST = False
PERSIST_DIRECTORY = "persist"
DATA_DIRECTORY = "data/"

def load_or_create_index():
    if PERSIST and os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader(DATA_DIRECTORY)
        creator = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": PERSIST_DIRECTORY}) if PERSIST else VectorstoreIndexCreator()
        return creator.from_loaders([loader])

index = load_or_create_index()

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history_global = []

last_session_id = 0

@app.route("/api/chat", methods=['POST'])
def chat():
    global last_session_id

    input_text = request.json.get('input')
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400

    if 'new_session' in request.json and request.json['new_session']:
        last_session_id += 1

    session_id = last_session_id if 'session_id' not in request.json else request.json['session_id']

    result = chain({"question": input_text, "chat_history": chat_history_global})
    chat_history_global.append((input_text, result['answer']))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO chats (user_input, bot_response, session_id) VALUES (%s, %s, %s)', 
                       (input_text, result['answer'], session_id))
        conn.commit()
    conn.close()

    return jsonify({'response': result['answer']})


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT session_id, user_input, bot_response FROM chats WHERE session_id IS NOT NULL ORDER BY session_id, timestamp')
        results = cursor.fetchall()
    conn.close()

    sessions = {}
    for row in results:
        session_id = row['session_id']
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({'user_input': row['user_input'], 'bot_response': row['bot_response']})

    formatted_sessions = [{"session_id": sid, "chats": chats} for sid, chats in sessions.items()]
    return jsonify(formatted_sessions)



@app.route('/api/session/<int:session_id>', methods=['GET'])
def get_session_chats(session_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM chats WHERE session_id = %s', (session_id,))
        chats = cursor.fetchall()
    conn.close()
    return jsonify(chats)



@app.route('/api/session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM chats WHERE session_id = %s', (session_id,))
        conn.commit()
    conn.close()
    return '', 204


if __name__ == "__main__":
    app.run(debug=True, port=8080)
