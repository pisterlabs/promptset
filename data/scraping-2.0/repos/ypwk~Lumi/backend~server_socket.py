#! /usr/bin/python3
import time
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO, send

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from langchain.llms.base import LLM
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

from typing import Optional
from threading import Thread

import psycopg2 as pg2

import torch.cuda


embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# model_id = "garage-bAInd/Platypus2-7B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "Intel/neural-chat-7b-v3-1"
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)
llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

cache = {}

vdb_params = {
    "driver": "psycopg2",
    "host": "localhost",
    "port": 5432,
    "database": "vdb",
    "user": "lumi",
    "password": "lumipass",
}

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=vdb_params["driver"],
    host=vdb_params["host"],
    port=vdb_params["port"],
    database=vdb_params["database"],
    user=vdb_params["user"],
    password=vdb_params["password"],
)

COLLECTION_NAME = "chat_logs"
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model,
)


class CustomLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None
    history = []

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.history = []
        self.streamer = TextIteratorStreamer(
            tokenizer=llm_tokenizer, skip_prompt=True, timeout=5
        )
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        thread = Thread(target=llm_model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def stream_tokens(self):
        for token in self.streamer:
            time.sleep(0.05)
            yield token


class CustomSocketLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None
    history = []
    question = ""
    session_saved = False

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.history = []
        self.streamer = TextIteratorStreamer(
            tokenizer=llm_tokenizer, skip_prompt=True, timeout=5
        )
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        thread = Thread(target=llm_model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def set_question(self, question):
        self.question = question
        self.update_cache(question, "")

    def stream_tokens(self):
        if not self.streamer.text_queue.empty():
            for token in self.streamer:
                time.sleep(0.05)
                cache[self.question] += token
                send(token)
        else:
            time.sleep(0.1)
            self.stream_tokens()

    def query_cache(self, question):
        """Check the cache for a response to the given question."""
        return cache.get(question)

    def update_cache(self, question, response):
        """Update the cache with the new question-response pair."""
        cache[question] = response

    def clear_cache(self):
        global cache
        cache = {}

    def build_context(self):
        """Construct the context string from the cache."""
        context_parts = []
        for question, answer in cache.items():
            context_parts.append(f" - Question: {question}\n - Answer: {answer}")
        return "\n".join(context_parts)

    def build_list_context(self):
        """Construct the context string from the cache."""
        context_parts = []
        for question, answer in cache.items():
            context_parts.append(f" - Question: {question}\n - Answer: {answer}")
        return context_parts

    def clear_session_flag(self):
        """Clears the session saved flag."""
        self.session_saved = False


llm_tokenizer.pad_token_id = llm_model.config.eos_token_id

template = """You are a friendly virtual assistant named Lumi.
Here is some context that may or may not be helpful and relevant. 
{context}
Here is the user's question: {question}
Your answer: Let's be concise and think carefully."""
prompt = PromptTemplate.from_template(template)
llm = CustomLLM()
chain = prompt | llm

socket_llm = CustomSocketLLM()
socket_chain = prompt | socket_llm

app = Flask(__name__)
socketio = SocketIO(app)


def save_data():
    if not socket_llm.session_saved:
        context_string = socket_llm.build_list_context()
        if context_string:
            print("Saving {}".format(context_string))
            store.add_documents(
                [Document(page_content=context) for context in context_string]
            )
            socket_llm.session_saved = True  # Update the flag
            socket_llm.clear_cache()
        else:
            print("No context to save.")
    else:
        print("Session data already saved.")


@app.route("/")
def hello_world():
    return "This is the base address for Lumi!"


@app.route("/query/<question>", methods=["GET"])
def query(question):
    print(question)
    chain.invoke(input=dict({"question": question}))
    return Response(llm.stream_tokens(), mimetype="text/plain")


@app.route("/get_archive/", methods=["GET"])
def get_archive():
    # SQL query to retrieve all entries from a table
    query = "SELECT * FROM langchain_pg_embedding"
    try:
        conn = pg2.connect(
            dbname=vdb_params["database"],
            user=vdb_params["user"],
            password=vdb_params["password"],
            host=vdb_params["host"],
            port=vdb_params["port"],
        )
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the cursor and connection
        if conn:
            cursor.close()
            conn.close()
    return jsonify(result)


@app.route("/delete_archive/<archive_id>", methods=["DELETE"])
def delete_archive(archive_id):
    try:
        conn = pg2.connect(
            dbname=vdb_params["database"],
            user=vdb_params["user"],
            password=vdb_params["password"],
            host=vdb_params["host"],
            port=vdb_params["port"],
        )
        cursor = conn.cursor()

        # SQL query to delete the specified entry
        query = "DELETE FROM langchain_pg_embedding WHERE uuid = %s"
        cursor.execute(query, (archive_id,))

        # Commit the changes to the database
        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"message": "No record found to delete"}), 404

        return jsonify({"message": "Archive deleted successfully"}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


CONTEXT_QUERY_SIZE = 3


@socketio.on("message")
def handle_message(data):
    print("Request: {}".format(data))
    docs_with_score = store.similarity_search_with_score(data)
    print(docs_with_score)
    context_string = "Context:\n"
    for doc, score in docs_with_score:
        context_string += doc.page_content
    context_string += socket_llm.build_context()
    print("\n{}\n".format(context_string))
    socket_llm.set_question(data)
    socket_chain.invoke(input=dict({"context": context_string, "question": data}))
    socket_llm.stream_tokens()


@socketio.on("connect")
def handle_connect():
    # Reset the session saved flag when a new connection is established
    socket_llm.clear_session_flag()


@socketio.on("disconnect")
def handle_disconnect():
    print("Socket disconnected, checking save status...")
    save_data()
