import os

from flask import Flask, request
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from twilio.rest import Client

from helpers.chat_loaders import TwilioConversationChatLoader

app = Flask(__name__)

# Requires OPENAI_API_KEY to be set
chat = ChatOpenAI()

SYSTEM_PROMPT = """You are answering an incoming text message. 

You are very interested in learning as much as you can about the human on the other end. 

Gather as much information as you can about them especially around software development. 

Try to keep your messages succinct but keep things very friendly. 

Ask only one follow up question at a time."""


def get_twilio_client():
    return Client(
        os.environ["TWILIO_ACCOUNT_SID"],
        os.environ["TWILIO_AUTH_TOKEN"]
    )


def on_message_added(event):
    client = get_twilio_client()
    chat_service_sid = event["ChatServiceSid"]
    conversation_sid = event["ConversationSid"]
    # Get existing messages
    loader = TwilioConversationChatLoader(client, chat_service_sid, conversation_sid)
    # Create chat_model objects
    chat_sessions = loader.load()
    messages =  chat_sessions[0]["messages"]
    print(f"There are {len(messages)} and {messages[-1]} is the latest")
    # Pass to LLM with System Message Prepended
    response = chat([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    # Add Response back to Conversation
    client.conversations.v1.services(chat_service_sid).conversations(conversation_sid).messages.create(
        body=response.content
    )


HANDLERS = {
    "onMessageAdded": on_message_added
}

@app.route("/handle-conversation-event", methods=["POST"])
def handle_conversation_event():
    print(f"Request: {request.form}")
    event = request.form
    if event["EventType"] in HANDLERS:
        print(f"""Handling {event["EventType"]}""")
        HANDLERS[event["EventType"]](event)
    else:
        print(f"""Unhandled event type: {event["EventType"]}""")
    return {"status": "success"}