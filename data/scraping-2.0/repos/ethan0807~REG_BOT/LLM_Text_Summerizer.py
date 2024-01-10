import Globals
import openai
import logging
from llama_index.chat_engine import SimpleChatEngine
from langchain.chat_models import ChatOpenAI
from llama_index import ServiceContext
import sys

# Generates a summary for a given node(s)

globals = Globals.Defaults()
openai.api_key = globals.open_api_key
temperature = 0  # globals.default_temperature
model = globals.default_model


def generate_summary_chat(nodes):
    responses = []
    service_context = ServiceContext.from_defaults(
        llm=ChatOpenAI(temperature=temperature, model=model))
    chat_engine = SimpleChatEngine.from_defaults(
        service_context=service_context)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    for node_with_score in nodes:
        text = node_with_score.node.text
        response = chat_engine.chat("Restate the following text more succintly and clearly: " + text)
        #response = chat_engine.chat(text + "\n\nTl;dr")
        responses.append(response)
        chat_engine.reset()

    return responses
