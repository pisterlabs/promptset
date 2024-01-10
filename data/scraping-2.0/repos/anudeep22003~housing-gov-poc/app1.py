from langchain.llms import OpenAI
from agents import Agent, classification_agent
from indexer import BuildRagIndex, index_to_product_mapping, product_descriptions

from fastapi import FastAPI
from pydantic import BaseModel, Field
import json

from flask import Flask, make_response, jsonify
from flask import request
import json
from flask_cors import CORS

from utils import documents_to_index

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("query.log", mode="a")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

###### Pydantic base classes for FastAPI ######


class Message(BaseModel):
    content: str


class Response(BaseModel):
    content: str
    product: str | None
    sources: str | None


####################################################


app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "Hello, World!"


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


# query = "What are the most important maintenance steps I need to do within one year?"
# query = "Something is wrong with the scanner. What should I do?"


def memory_refresher():
    f = open("memory.txt", "w")
    f.close()


def memory_getter() -> Message | None:
    f = open("memory.txt", "r")
    memory = f.read()
    f.close()
    if memory == "":
        # return means this is a new request
        return None
    else:
        memory = json.loads(memory)
        return Message(**memory)


def memory_writer(memory: Message):
    with open("memory.txt", "w") as f:
        f.write(json.dumps(memory.dict()))


def get_classification(message: Message) -> Response:
    product_that_query_is_about = classification_agent(message.content)
    product_that_query_is_about = product_that_query_is_about.strip()

    logger.debug(f"product_that_query_is_about: {product_that_query_is_about}")
    # appropriate rag index
    try:
        index_id = index_to_product_mapping[product_that_query_is_about]
        msg1 = f"You seem to be asking about {product_that_query_is_about}. Press enter if I got it right. \n\nIf not type `no`, and I will try asking the question again.\n\nI am fairly capable, so help me with a few contextual clues and I'll figure it out."
        return Response(content=msg1, product=product_that_query_is_about, sources=None)
    except KeyError:
        msg1 = f"Sorry, I cannot seem to find the product you are asking about in my database.\n\n"
        msg2 = f"As reference, I only have the following products in my database:\n{list(index_to_product_mapping.keys())}"
        msg3 = f"\n\nPlease try again. It may help to give any identifying infromation about the product for my benefit."
        return Response(content=f"{msg1}{msg2}{msg3}", product=None, sources=None)


def perform_rag_call(message: Message) -> Response:
    # response query initialize
    response_query = []
    # find the appropriate index for the product
    product_that_query_is_about = classification_agent(message.content)
    product_that_query_is_about = product_that_query_is_about.strip()

    print(f"product_that_query_is_about: {product_that_query_is_about}")
    # appropriate rag index
    try:
        index_id = index_to_product_mapping[product_that_query_is_about]
        msg1 = f"Product: {product_that_query_is_about}.\n\n"
    except KeyError:
        msg1 = f"Sorry, I cannot seem to find the product you are asking about in my database."
        msg2 = f"I only have the following products in my database: {list(index_to_product_mapping.keys())}"
        msg3 = f"Please try again. It may help to give any identifying infromation about the product for my lookup benefit."
        response_query.extend([msg1, msg2, msg3])
        response_obj = {
            "content": "\n\n".join(response_query),
            "product": None,
            "sources": None,
        }
        logger.info(response_obj)
        logger.info(f"\n {'-'*30}\n")
        return Response(**response_obj)

    b = BuildRagIndex(index_id)
    response_text, page_numbers = b.query(message.content)
    # sort page numbers for presentation
    page_numbers = sorted(page_numbers)
    response_query.append(msg1)
    response_query.append(response_text)
    response_obj = {
        "content": "\n\n".join(response_query),
        "product": product_that_query_is_about,
        "sources": ", ".join([str(page_num) for page_num in page_numbers]),
    }
    logger.info(response_obj)
    logger.info(f"\n {'-'*30}\n")
    return Response(**response_obj)


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


@app.route("/chat", methods=["POST", "OPTIONS"])
def get_response() -> dict | Response | None:
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_preflight_response()
    else:
        print("Hello")
        # postData = json.loads(request.data)
        bytes_data = request.data
        json_obj = request.get_json()
        passed_to_message = jsonify(request.get_json())

        message = Message(content=json_obj["content"])
        #       print(postData)
        #       message = Dict2Class(postData)
        memory = memory_getter()

        if memory is None:
            print("memory is None, hence doing classification")
            # means this is a fresh request
            # send a classification response
            memory_writer(message)
            response_msg = get_classification(message)
            if "sorry" in response_msg.content.lower():
                memory_refresher()
                return response_msg
            return response_msg
        elif message.content.strip().lower() in ["n", "no"]:
            memory_refresher()
            return Response(
                content="Sorry for getting it wrong, request you to try asking your question again.\n\n",
                product=None,
                sources=None,
            )
        elif message.content.strip().lower() in ["", "y", "yes"]:
            # switch the message so as to reset the memory for the next call
            memory_refresher()
            # perform the rag call
            return perform_rag_call(memory)
        else:
            memory_refresher()
            return Response(
                content="...\nApologoes for the hiccup. Needed to reset my memory there. I am ready now. Please ask me again.",
                product=None,
                sources=None,
            )


class ConversationHandler:
    def __init__(self, message: Message):
        self.memory: Message | None = None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
