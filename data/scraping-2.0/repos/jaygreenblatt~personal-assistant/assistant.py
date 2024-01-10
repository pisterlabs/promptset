"""
This is the main file for the assistant.
"""

import os
import sys

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def create_model():
    """
    Creates the LLM model.
    """
    model = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
    return model


def generate_response(model, prompt: str):
    """
    Generates a response from the model.
    """
    response = model.predict_messages([HumanMessage(content=prompt)])
    return response


if __name__ == "__main__":
    print("os.environ", os.environ["OPENAI_API_KEY"])
    query = sys.argv[1]
    llm = create_model()
    print(generate_response(llm, query))
