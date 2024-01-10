# main.py
"""
Interactive chat with memory for earlier chats.
"""
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin  # Setting cors for http
from index_handler import load_index, store_index
from llama_index.memory import ChatMemoryBuffer
import logging
import asyncio
import threading
import re
from utils import check_nature_of_my_response, handle_user_prompt

# Create a Flask app
chat_app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for the app
cors = CORS(chat_app)  # Setting cors for http
chat_app.config["CORS_HEADERS"] = "Content-Type"
logging.basicConfig(level=logging.DEBUG)
text_completion = None
# Define a function to run in a thread
import re
import json


def handle_chat_thread(chat_engine, input_text):
    global text_completion
    if chat_app.reference_title is None:
        reference_title = " "
    else:
        reference_title = chat_app.reference_title
    logging.info(f"\n###########Context Title :{reference_title}###########")
    text_completion = chat_engine.chat(
        "Only answer from the context," + reference_title + input_text
    )
    chat_app.reference_title = access_context_title(text_completion)


"""
Method for handling user chat api calls # Define an API route for handling user text input
"""


@chat_app.route("/api/send_text", methods=["POST"])
@cross_origin()  # Setting cors for http
def send_text():
    try:
        message_response = "Server Error or Usage Limit"  # To avoid HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out
        data = request.get_json()
        input_text = (
            data.get("text", "hi") + "?"
        )  # if no text in json,add 'hi' and add '?' to make proper conversation answer.
        flag = False
        message_response, flag = handle_user_prompt(input_text)
        if not flag:
            # Load the chat engine
            if not hasattr(chat_app, "chat_engine") or chat_app.chat_engine is None:
                asyncio.run(get_chat_engine())
                logging.info("Chat engine was missing - Recreated.")
            # making a thread for handling openai chat calls
            chat_thread = threading.Thread(
                target=handle_chat_thread,
                kwargs={
                    "chat_engine": chat_app.chat_engine,
                    "input_text": input_text,
                },
            )
            chat_thread.start()
            # main thread waiting for chat_thread completion
            chat_thread.join(30)
            # assigning the value created by chat_thread
            if text_completion is not None:
                reference = check_context(text_completion)
                message_response = check_nature_of_my_response(f"{text_completion}")
                if len(message_response) == 0:
                    message_response = (
                        "Inconvenience Regretted!\nNo response could be generated..!"
                    )
                if reference is not None:
                    disclaimer = "\nI believe the answer is correct.I am still under experiment, please verify my response using the details below:\n"
                    message_response = message_response + disclaimer + reference
                else:
                    print("**************", "\n", reference)
    except Exception as e:
        logging.error("Exception occurred while generating the response - " + str(e))
        message_response = "Inconvenience Regretted!\nAn Exception occurred while generating the response."
    finally:
        return jsonify({"response": message_response})


"""
api call for resetting the chat # Define an API route for resetting the chat (if needed)
"""


@chat_app.route("/api/chat_reset", methods=["POST"])
@cross_origin()  # Setting cors for http
def chat_reset():
    if hasattr(chat_app, "chat_engine") and chat_app.chat_engine is not None:
        chat_app.chat_engine.reset()
        chat_app.reference_title = None
        response = "Chat Resetted"
    else:  # when chat_enine not exist or None
        # creating a new chat_engine
        asyncio.run(get_chat_engine())
        response = "Chatbot is Ready..!"
        logging.error("Chatbot is Ready..!")

    return make_response(response, 200)


"""
Method for loading the index file and initializing the chat_engine with memory
"""


async def get_chat_engine():
    logging.info(" Inside method get_chat_engine ")
    if not hasattr(chat_app, "chat_engine"):
        index = load_index()  # Load your index using your custom function
        memory = ChatMemoryBuffer.from_defaults()
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=1,
            memory=memory,
            system_prompt="You are a virtual office assistant of Botswana Government with a limited knowledge to "
            "given context, you should respond to user queries with a friendly language, "
            "you should engage in small talk, you should be empathetic, who can provide clear "
            "and helpful responses."
            "\nAsk user to clarify the question if you are confused or have multiple answers."
            "\nYou can only provide answers and information based on the PROVIDED CONTEXT. "
            "\nIf you did not find the relevant information in the context, please respond to "
            "the user that: 'I'm afraid that what you asked is something that I can't answer "
            "per my knowledge. Sorry for the inconvenience caused. Please let me know if "
            "there is something else that I can help you with.'"
            "\nWhen you respond to a user, never mention that you are answering from 'provided context'.",
        )
        chat_app.chat_engine = chat_engine  # Store chat_engine in the app context
        chat_app.reference_title = None


def check_context(response_text):
    if hasattr(response_text, "source_nodes"):
        document_info = str(response_text.source_nodes)

        # Extract URL, department, section, and title from the JSON data
        urls = re.findall(r'"url": "([^"]*)"', document_info)
        departments = re.findall(r'"department": "([^"]*)"', document_info)
        sections = re.findall(r'"section": "([^"]*)"', document_info)
        titles = re.findall(r'"title": "([^"]*)"', document_info)
        if urls:
            # Create a list of dictionaries containing all four attributes
            info_list = [
                {
                    "url": url,
                    "department": department,
                    "section": section,
                    "title": title,
                }
                for url, department, section, title in zip(
                    urls, departments, sections, titles
                )
            ]
            # Create a formatted string with newline-separated entries
            reference = "\n".join(
                [
                    f"\nTitle: {info['title']}\nDepartment: {info['department']}\nSection: {info['section']}URL: {info['url']}"
                    for info in info_list
                ]
            )
            return reference

    else:
        logging.error("no such attribute^^^^^")


def access_context_title(response_text):
    if hasattr(response_text, "source_nodes"):
        document_info = str(response_text.source_nodes)

        titles = re.findall(r'"title": "([^"]*)"', document_info)
        if titles:
            # Create a list of dictionaries containing all four attributes
            info_list = [
                {
                    "title": title,
                }
                for title in zip(titles)
            ]
            # Create a formatted string with newline-separated entries
            reference_titles = "\n".join([f"{info['title']}" for info in info_list])
            return reference_titles

    else:
        logging.error("no such attribute^^^^^")


# Run the Flask app if this script is the main entry point
if __name__ == "__main__":
    # asyncio.run(store_index())  # Called only once to create the index files
    asyncio.run(get_chat_engine())
    chat_app.run(host="0.0.0.0", port=5000, debug=True)



utils.py



import logging
import re


# process the response with custin operations
def check_nature_of_my_response(message_response: str):
    logging.info(message_response)
    # replacing the context word
    message_response = message_response.replace(
        "context", "botswana government services"
    )
    return message_response


# handling user prompt to manage locally or sending the openai
def handle_user_prompt(user_prompt: str):
    message_response_processed = check_match(user_prompt)
    if message_response_processed is not None:
        return message_response_processed, True
    return message_response_processed, False


# perform custom check on string to retuen dictionary value for a key
def check_match(string_to_check: str):
    greet_dict = {
        "hello": "Hello! This is your AI assistant. \nHow can I assist you today?",
        "hey": "Hello! This is your AI assistant. \nHow can I assist you today?",
        "hi": "Hello! This is your AI assistant. \nHow can I assist you today?",
        "good morning": "Good morning! This is your AI assistant. \nHow can I assist you today?",
        "good afternoon": "Good afternoon! This is your AI assistant. \nHow can I assist you today?",
        "good evening": "Good evening! This is your AI assistant. \nHow can I assist you today?",
        "thanks": "I am glad that I could serve you properly.\nPlease let me know if "
        "there is something else that I can help you with.?",
        "thank you": "I am glad that I could serve you properly.\nPlease let me know if "
        "there is something else that I can help you with.?",
        "welcome": "I am glad that I could serve you properly.\nPlease let me know if "
        "there is something else that I can help you with.?",
    }
    logging.info(string_to_check)
    string_to_check_lower = string_to_check.lower()
    for key in greet_dict:
        # Create a regex pattern to match whole words or phrases
        pattern = r"\b{}\b".format(re.escape(key.lower()))
        # Check if the pattern is found in the input string
        if re.search(pattern, string_to_check_lower):
            return greet_dict.get(key)



index_handler.py


# index_handler.py
import os, logging
from pathlib import Path
from dotenv import load_dotenv

import openai
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptHelper,
    LLMPredictor,
    ServiceContext,
)
from langchain.chat_models import ChatOpenAI


# Set base directory and load environment variables -- IT is must
BASE_DIR = Path(__file__).resolve().parent
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]


def process_initializers():  # only initialize variables
    # Configuration parameters
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    logging.info("Inside process_initializers")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    try:
        # Initialize prompt_helper and ChatOpenAI
        prompt_helper = PromptHelper(
            max_input_size,
            num_outputs,
            chunk_overlap_ratio=0.2,
            chunk_size_limit=chunk_size_limit,
        )
        # llm = ChatOpenAI(
        #     openai_api_key=openai_api_key, temperature=0.3, model_name="gpt-4"
        # )
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
            # model_name="gpt-3.5-turbo",
            model_name="gpt-3.5-turbo-0613",
            request_timeout=20,
            max_retries=2,
            max_tokens=500,  # max token to generate
            # model_kwargs={"stop": "\n"},
            # model_kwargs={"messages": last_chats},
        )
        # llm = OpenAI(api_key=openai_api_key,temperature=0,model="gpt-3.5-turbo-0613",\
        #              additional_kwargs={"request_timeout":12},max_retries=2)

        llm_predictor = LLMPredictor(llm=llm)
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper
        )
        return service_context
    except Exception as e:
        logging.error(str(e))
        raise


async def store_index():
    try:
        # get the current working directory
        cwd = os.getcwd()
        # define the relative path to the index_repo folder
        file_repo_path = os.path.join(cwd, "file_repo")
        index_repo_path = os.path.join(cwd, "index_repo")
        service_context = process_initializers()  # calling method

        documents = SimpleDirectoryReader(file_repo_path).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents=documents, service_context=service_context
        )
        # index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_repo_path)
    except Exception as e:
        logging.error(str(e))
        raise


def load_index():
    try:
        # get the current working directory
        cwd = os.getcwd()
        # define the relative path to the index_repo folder
        index_repo_path = os.path.join(cwd, "index_repo")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_repo_path)
        service_context = process_initializers()  # calling method

        # load index
        index = load_index_from_storage(
            storage_context=storage_context, service_context=service_context
        )
        return index
    except Exception as e:
        logging.error(str(e))
        raise
chat_engine.py

import logging
from index_handler import load_index
from llama_index.memory import ChatMemoryBuffer
import asyncio


"""
Method for loading the index file and initializing the chat_engine with memory
"""

# TODO remove later not using
async def get_chat_engine(chat_app):
    logging.info(" Inside method get_chat_engine ")
    global memory, system_prompt
    if not hasattr(chat_app, "chat_engine"):
        index = load_index()  # Load your index using your custom function
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        memory.reset()
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt="You are a virtual office assistant of Botswana Government who should respond to \
                user queries with a friendly language, who should engage in small talk, who should be empathetic,\
                      who can provide clear and helpful responses, who should offer compliments on positive feedback.\
                        Please provide answers and information \
                                based only on the content within those documents.\
                            \nWhen question outside context asked decline with a message:\
                                    \nYour query is outside my purview,Please contact ********",
        )
        chat_app.chat_engine = chat_engine  # Store chat_engine in the app context
