# from dotenv import dotenv_values
import logging
from typing import Dict
from typing import List

import openai
import pinecone
import yaml
from fastapi import HTTPException
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from doc_utils import fetchTopK
from doc_utils import search_documents_by_file_name
from retrieve import get_embedding
from retrieve import query_pinecone

# config = dotenv_values(".env")
# from flask import jsonify, make_response
# from flask import request


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as config_file:
        return yaml.safe_load(config_file)


config = load_config("config.yaml")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pinecone.init(api_key=config["PINECONE_API_KEY"],
              environment=config["PINECONE_ENVIRONMENT"])
index = pinecone.Index(config["PINECONE_INDEX_NAME"])

tone = config["tone"]
persona = config["persona"]

# Initialize the QA chain
logger.info("Initializing QA chain......")
chain = load_qa_chain(
    ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"]),
    chain_type="stuff",
    memory=ConversationBufferMemory(memory_key="chat_history_redis",
                                    input_key="human_input"),
    prompt=PromptTemplate(
        input_variables=[
            "chat_history",
            "human_input",
            "context",
            "tone",
            "persona",
            "filenames",
            "text_list",
        ],
        template=
        """You are a chatbot who acts like {persona}, having a conversation with a student.

Given the following extracted parts of a long document answer the question in the tone {tone}.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "FILENAMES" part only at the end of your answer with the {filenames}.

Extracted parts: {text_list}.
APPLY NLP TECHNIQUES FOR A WELL FORMATTED ANSWER


{context}

{chat_history}
Human: {human_input}
Chatbot:""",
    ),
    verbose=False,
)


def chat_ask_question(
    user_input: str,
    chat_history_redis: List[Dict[str, str]],
    file_name=None,
    truncated_question=None,
    truncation_step=0,
):
    """
    Handles the chat request, retrieves relevant documents, and generates the chatbot's response.

    :param truncated_question: The original question with a reduced length, if any (default: None).
    :param truncation_step: The number of times the incput question has been truncated (default: 0).
    :return: A JSON serialized response containing the chatbot's response or an error message.
    """
    try:
        # Get the question from the request
        question = user_input
        file_name = file_name
        query_embeds = get_embedding(question)

        documents = search_documents_by_file_name(index,
                                                  tuple(query_embeds),
                                                  file_name,
                                                  include_metadata=True)
        """ print(chat_history_redis)
        print(type(chat_history_redis)) """
        # Convert chat history from list of dicts to string
        chat_history_str = "\n".join(f'{msg["user"]}: {msg["message"]}'
                                     for msg in chat_history_redis)

        # print(query_pinecone.cache_info())

        # Log number of matching documents
        logger.debug(
            f"Number of matching documents: {len(documents['matches'])}")

        # Extract the unique filenames from the matching documents
        filenames = get_unique_filenames(documents["matches"])
        logger.info(f"Unique source filenames: {filenames}")

        # Extract the relevant text from the matching documents (if truncated, remove truncation_step number of elements)
        text_list = [{
            "text": match["metadata"]["text"]
        } for match in documents["matches"]]

        if truncation_step == 0:
            truncated_question = user_input

        if truncation_step > 0:
            original_length = len(text_list)
            text_list = text_list[:-truncation_step]
            logger.info(
                f"Truncating text_list from {original_length} to {len(text_list)} elements."
            )

        # Get the bot's response
        response = chain(
            {
                "input_documents": documents["matches"],
                "human_input": question,
                "chat_history": chat_history_str,
                "tone": tone,
                "persona": persona,
                "filenames": filenames,
                "text_list": text_list,
            },
            return_only_outputs=True,
        )
        # Print chat history
        # chat_history = chain.memory.buffer
        chat_history = chat_history_redis
        print(f"Chat history redis {chat_history}")
        # print(f"Chat history: {chat_history}")
        # Extract the response text
        response_text = response["output_text"]
        logger.info(f"RESPONSE: {response_text} ")
        # Return the JSON serialized response
        return response_text
    except openai.InvalidRequestError as e:
        error_message = str(e)
        if "maximum context length" in error_message:
            if truncation_step < 4:
                return chat_ask_question(
                    user_input=user_input,
                    file_name=file_name,
                    truncated_question=truncated_question,
                    truncation_step=truncation_step + 1,
                )
            elif truncation_step >= 4:
                logger.error(f"Error while processing request: {e}")
                raise HTTPException(
                    status_code=422,
                    detail=
                    "The input is too long. Please reduce the length of the messages.",
                )
        else:
            logger.error(f"Invalid request error: {e}")
            raise HTTPException(
                status_code=400,
                detail=
                f"Unable to process the request due to an invalid request error: {error_message}",
            )

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        raise HTTPException(status_code=500,
                            detail="Unable to process the request.")


def get_unique_filenames(matches):
    seen_filenames = set()
    filenames = []

    for doc in matches:
        file_name = doc["metadata"]["file_name"]

        # Remove 'uploads/' part from the filename if present
        file_name = file_name.replace("uploads/", "")

        if file_name not in seen_filenames:
            filenames.append(file_name)
            seen_filenames.add(file_name)

    return filenames
