
from flask import request
from flask import Flask, jsonify
from flask_cors import CORS
import socket
from openai_utils import num_tokens_from_messages
from chatbot import ChatBot
from utils import (
    handle_server_errors, 
    get_hash,
    get_duplicate_embedding,
    query_all_embeddings,
    create_memorization_prompt,
    emit_socket_event
)
from response_database_manager import response_db_manager
from qdrant_vector_store import qdrant_vector_store
from embedding_service import embedding_service, Embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import openai
from chatroom_answer_service import ChatRoomAnswerService
from chat_room_resource_service import ChatRoomResouceService
import threading


app = Flask(__name__)
CORS(app, supports_credentials=True)
server_print = lambda content: app.logger.info(content)
host = socket.gethostname()

default_text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 500, chunk_overlap  = 50, length_function = len, add_start_index = True
)

@app.route("/")
def index():
    return f"Welcome to a server for answering question {host} and creating embeddings", 200

@app.route("/answer", methods=["POST"])
@handle_server_errors
def answer():
    request_json = request.get_json()
    prompt = request_json["prompt"]
    api_key = request_json["api_key"]
    room_id = request_json["room_id"]
    user_id = request_json["user_id"]
    user_name = request_json["user_name"]
    messages = request_json["messages"]
    room_rule = request_json["room_rule"]
    answer_service = ChatRoomAnswerService(
        room_rule= room_rule,
        prompt= prompt, 
        api_key= api_key, 
        room_id= room_id,
        user_id= user_id,
        user_name= user_name,
        messages= messages
    )
    source = request_json.get("source", "db")
    source_lookup = {
        "db": answer_service.ask_vector_store,
        "web": answer_service.ask_web
    }
    handler = source_lookup[source]
    resource_service = ChatRoomResouceService(
        room_id= room_id,
        user_id= user_id,
        user_name= user_name,
        prompt= prompt
    )
    threading.Thread(target= lambda: resource_service.run_external_service(handler)).start()
    return "ok"

@app.route("/count_tokens", methods=["POST"])
def count_tokens():
    request_json = request.get_json()
    messages = request_json["messages"]
    num_tokens = num_tokens_from_messages(messages)
    return jsonify({"num_tokens": num_tokens, "host": host}), 200

@app.route("/improve_prompt", methods=["POST"])
@handle_server_errors
def improve_prompt():
    request_json = request.get_json()

    openai.api_key = request_json["api_key"]
    prompt = request_json["prompt"]
    user_id = request_json["user_id"]
    lang = request_json["language"]

    socket_event = f"prompt/{user_id}"
    system_prompt = create_memorization_prompt(prompt, lang)
    bot = ChatBot(system_prompt, [])
    for current_message in bot.answer():
        emit_socket_event(socket_event, {"user_id": user_id,"content": current_message})
        
    emit_socket_event(socket_event, {"user_id": user_id,"content": current_message,"is_message_persist": True})

@app.route("/memo", methods=["POST"])
@handle_server_errors
def memo():
    request_json = request.get_json()
    openai.api_key = request_json["api_key"]
    room_id = request_json["room_id"]
    prompt = request_json["prompt"]
    if len(prompt) == 0:
        raise ValueError("Memo prompt cannot be empty")

    chunks = [
        chunk for chunk in default_text_spliter.split_text(prompt)
    ]
    embeddings = []
    document_id = str(uuid.uuid4())
    for chunk in chunks:
        chunk_hash = get_hash(chunk)
        embedding_dict = get_duplicate_embedding(chunk_hash)
        embedding = (
            embedding_service.get_embedding(chunk, document_id, chunk_hash) 
            if embedding_dict is None
            else Embedding(
            document_id= document_id,
            chunk_id= embedding_dict["chunk_id"],
            text= embedding_dict["text"],
            text_hash= embedding_dict["text_hash"],
            updated_at= embedding_dict["updated_at"],
            vector= embedding_dict["vector"]) 
             
            )
        embeddings.append(embedding)
    if len(embeddings) == 0:
        raise ValueError("All embeddings are duplicates")
    
    '''
    "is_ok": is_ok,
    "collection_name": collection_name,
    "embeddings": embeddings
    '''
    upsert_resp  = qdrant_vector_store.upsert_text(room_id, embeddings)
    
    if not upsert_resp["is_ok"]:
        raise ValueError("Failed to insert Text")
    response_db_manager.save_embeddings(
        collection_name= upsert_resp["collection_name"],
        embeddings= [embedding.to_dict() for embedding in upsert_resp["embeddings"]]
    )
    
    return upsert_resp

@app.route("/init_qdrant_store")
def init_qdrant_store():
    all_embeddings = query_all_embeddings()
    for embedding_dict in all_embeddings:
        collection_name = embedding_dict["collection_name"]
        embedding = Embedding(
            document_id= embedding_dict["document_id"],
            chunk_id= embedding_dict["chunk_id"],
            text= embedding_dict["text"],
            text_hash= embedding_dict["text_hash"],
            updated_at= embedding_dict["updated_at"],
            vector= embedding_dict["vector"]
        )
        print(f"Upserting: {embedding.chunk_id} into {collection_name}")
        qdrant_vector_store.upsert_text(collection_name, [embedding])
    return {"message": "ok", "upserted_rows": len(all_embeddings)}

        




if __name__ == "__main__":
    
    app.run(host="0.0.0.0", debug= True)
