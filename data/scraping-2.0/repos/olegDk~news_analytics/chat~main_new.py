from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from typing import List, Dict
from custom_tools import UnemploymentRateTool, PayrollsTool, CorporateTool
import os
import json
import logging
import traceback

# Turning validation into no op for multi-input tools support
ConversationalChatAgent._validate_tools = lambda *_, **__: ...

app = FastAPI()

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging configuration
logging.basicConfig(
    filename="chat_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
        self.conversation_memories: Dict[int, ConversationBufferWindowMemory] = {}
        self.agents: Dict[int, ConversationalChatAgent] = {}

    async def connect(self, client_id: int, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            self.conversation_memories[client_id] = ConversationBufferWindowMemory(
                memory_key="chat_history", k=5, return_messages=True
            )
            tools = [UnemploymentRateTool(), PayrollsTool(), CorporateTool()]
            self.agents[client_id] = initialize_agent(
                agent="chat-conversational-react-description",
                tools=tools,
                llm=llm,
                agent_instructions="Try any of Custom Tools first but if you are not sure what to choose, proceed with Final Answer.",
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                memory=self.conversation_memories[client_id],
            )
            logging.info(f"Client {client_id} connected.")
        except Exception as e:
            logging.error(
                f"Error occurred while connecting client {client_id}: {str(e)}"
            )
            logging.debug(traceback.format_exc())

    def disconnect(self, client_id: int):
        del self.active_connections[client_id]
        del self.conversation_memories[client_id]
        del self.agents[client_id]
        logging.info(f"Client {client_id} disconnected.")

    async def send_personal_message(self, client_id: int, message: str):
        try:
            await self.active_connections[client_id].send_text(message)
        except Exception as e:
            logging.error(
                f"Error occurred while sending personal message to client {client_id}: {str(e)}"
            )
            logging.debug(traceback.format_exc())


manager = ConnectionManager()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "OPENAI_API_KEY"
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo"
)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(client_id, websocket)
    while True:
        try:
            data = await websocket.receive_text()
            agent = manager.agents[client_id]
            logging.info(f"Agent executing on current data: {data}")
            agent_output = agent(data)
            logging.info(f"Agent output: {agent_output}")
            response_data = json.dumps(
                {
                    "reply": {
                        "reply": agent_output,
                        "sources": ["Tool Source"],
                        "type": "Tool Type",
                    }
                }
            )
            await manager.send_personal_message(client_id, response_data)
            logging.info(f"Message processed and sent to client {client_id}.")
        except WebSocketDisconnect:
            manager.disconnect(client_id)
            logging.info(f"WebSocket disconnected for client {client_id}.")
            break  # Exit the loop if client disconnects
        except Exception as e:
            logging.error(
                f"Error occurred in WebSocket connection with client {client_id}: {str(e)}\n"
                + f"Traceback: {traceback.format_exc()}"
            )
            # Continue to the next iteration of the loop if an error occurred
