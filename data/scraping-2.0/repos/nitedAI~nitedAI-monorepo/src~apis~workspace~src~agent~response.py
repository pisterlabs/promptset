# within this file we will follow the same structure as the workspace.py file
from flask import Blueprint, jsonify, request
from src.routes.routes import routes
from src.datasets.db import supabase
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

import json


# Use the same key as in the routes.py dictionary
base = "response"
response_bp = Blueprint(base, __name__)


@response_bp.route(routes[base]["base"], methods=routes[base]["methods"])
def get_response():
    # Get the message info from the request
    data = request.get_json()
    print("data: ", data)
    channel_id = data.get("channel_id")
    participant_uuid = data.get("participant_id")

    # Get conversation history from database
    message_data_response = supabase.rpc('get_last_n_messages_by_channel_id', {'channel_uuid': channel_id, 'last_n_messages': 15}).execute()
    print("DATABASE RESPONSE: ", message_data_response)
    message_data_list = message_data_response.data  # If this is already a list of dictionaries

    # Check the type of message_data_list to ensure it's what we expect
    print("message data list type: ", type(message_data_list))

    # Construct conversation history from the list of message dictionaries
    conversation_history = [
        f"{message['user_username'] if message['user_username'] is not None else 'participant'}: {message['content']}"
        for message in message_data_list
    ]

    # Message processing
    message = data.get('message', '').replace("@nitedAI", "")  # Default to an empty string if 'message' is not found

    # invoke the AI agent
    # create sparse priming representation of the conversation history
    # the conversation history is a list of dictionaries, each dictionary contains the message content and the user id
    # the user id is used to identify the user who sent the message
    # the message content is used to prime the AI agent
    # the AI agent will use the messages fields of user_username and message content to generate a spare priming response to use in another openAI API call
    # the sparse priming response will be used to generate the AI agent's response

    llm = OpenAI(model="text-davinci-003")

    # --------------------------------------------------------------
    # Create a PromptTemplate and LLMChain
    # --------------------------------------------------------------
    template_spr_chain = """# MISSION
    You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

    # THEORY
    LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

    # METHODOLOGY
    Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible about the conversation but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human. Use complete sentences.

    # USER
    What follows is the conversation history to be rendered as an SPR:
    {conversation_history}
    """

    prompt_spr_chain = PromptTemplate(template=template_spr_chain, input_variables=["conversation_history"])
    spr_chain = LLMChain(prompt=prompt_spr_chain, llm=llm)

    # --------------------------------------------------------------
    # Run the LLMChain
    # --------------------------------------------------------------
    input_string =""
    for message in conversation_history:
        input_string += message + "\n"
    # run the LLMChain
    chat_sparse_priming_rep = spr_chain.run(input_string)

    #append to that string the message that the user just sent
    chat_sparse_priming_rep += "\n" + data["message"]

    # --------------------------------------------------------------
    # Create a PromptTemplate and LLMChain
    # --------------------------------------------------------------
    template_response = """# MISSION
    You are a response writer. You will be given information by the AGENT which you are to render as a response.

    # THEORY
    LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

    # METHODOLOGY
    What follows is a Sparse Priming Representation (SPR) of the conversation history followed by the message the USER sent you are to respond to:

    {chat_spr}

    Render the input as a response to the USER. The idea is to capture as much, conceptually, as possible about the conversation in a detailed conversational manner. """

    prompt_response = PromptTemplate(template=template_response, input_variables=["chat_spr"])
    response_chain = LLMChain(prompt=prompt_response, llm=llm)

    # --------------------------------------------------------------
    # Run the LLMChain
    # --------------------------------------------------------------

    # run the LLMChain
    response = response_chain.run(chat_sparse_priming_rep)


    supabase.rpc('insert_message', {
        'channel_uuid': channel_id,
        'participant_uuid': participant_uuid,
        'message_content': response,
        'is_agent': True
    }).execute()


    return jsonify(response)


