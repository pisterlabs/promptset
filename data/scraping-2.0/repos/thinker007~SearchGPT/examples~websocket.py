import json

import openai
from websock import *

from SearchGPT.ChatBot import ChatBot
from SearchGPT.Commands.SearchCommand import SearchCommand

chatbots = {}
nextID = 0

search = SearchCommand()

def on_data_receive(client, stData):
    global chatbots
    data = json.loads(stData)
    chatbot = chatbots[client]
    if data["type"] == "apikey":
        chatbot.apiKey = data["apikey"]
        key = data["apikey"]
        models = chatbot.listModels()
        server.send(client, json.dumps({"type": "models", "models": models, "apikey": key}))
    elif data["type"] == "model":
        chatbot.model = data["model"]
    if data["type"] == "question":
        try:
            reply = chatbot.ask(data["question"])
            server.send(client, json.dumps({"type": "message", "message": reply}))
        except openai.error.AuthenticationError:
            server.send(client, json.dumps({"type": "error", "error": "Invalid API Key"}))
        except openai.error.APIConnectionError:
            server.send(client,
                json.dumps({"type": "error", "error": "Could not connect to OpenAI API"}))
        except openai.error.APIError as e:
            server.send(client,
                json.dumps({"type": "error", "error": "OpenAI API Error: " + str(e)}))
        except Exception as e:
            server.send(client, json.dumps({"type": "error", "error": "Unknown Error: " + str(e)}))



def on_connection_open(client):
    global nextID
    nextID += 1
    chatbot = ChatBot("SearchGPT",
                      botType="Web UI based text-based search assist chatbot.",
                      progress=lambda p: server.send(client, json.dumps(p)), commands=[search])
    chatbots[client] = chatbot


def on_error(exception):
    """Called by the WebSocket server whenever an Exception is thrown."""
    # Your implementation here.


def on_connection_close(client):
    """Called by the WebSocket server when a connection is closed."""
    # Your implementation here.


def on_server_destruct():
    """Called immediately prior to the WebSocket server shutting down."""
    # Your implementation here.

server = WebSocketServer(
    "0.0.0.0",
    8080,
    on_data_receive=on_data_receive,
    on_connection_open=on_connection_open,
    on_error=on_error,
    on_connection_close=on_connection_close,
    on_server_destruct=on_server_destruct
)

server.serve_forever()
