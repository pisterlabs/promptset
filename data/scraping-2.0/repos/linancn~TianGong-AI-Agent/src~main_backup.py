import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ChatMessageHistory
from websockets.exceptions import ConnectionClosedOK

from .modules.ui import ui_config
from api.callback import StreamingLLMCallbackHandler
from api.query_data import (
    chat_history_chain,
    embedding_formatter,
    func_calling_chain,
    get_chain,
    seach_docs,
    search_internet,
    search_pinecone,
)
from api.schemas import ChatResponse
from api.check import checkApiLoginCode

# Load the environment variables from the .env file
load_dotenv()

ui = ui_config.create_ui_from_config()

# Access the variables using os.environ
openai_api_key = os.environ.get("openai_api_key")

app = FastAPI()

origins = [
    "http://localhost:8081",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"result": "Hello World"}


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)
    qa_chain = get_chain(stream_handler)

    while True:
        try:
            # Receive and send back the client message
            send_msg = await websocket.receive_text()

            send_msg_json = json.loads(send_msg)
            question = send_msg_json.get("question")

            resp = ChatResponse(sender="human", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # check = checkApiLoginCode(send_msg_json.get("userId"), send_msg_json.get("apiLoginCode"))
            check = "ok"

            if check != "ok":
                resp = ChatResponse(sender="bot", message="err", type="stream")
                await websocket.send_json(resp.dict())

            else:
                resp = ChatResponse(sender="bot", message="ok", type="stream")
                await websocket.send_json(resp.dict())
                chat_history = ChatMessageHistory()
                history = send_msg_json.get("history")
                if len(history) <= 1:
                    func_calling_response = func_calling_chain().run(question)

                    query = func_calling_response.get("query")

                    try:
                        created_at = json.loads(
                            func_calling_response.get("created_at", None)
                        )
                    except TypeError:
                        created_at = None

                    is_search_docs = False
                    search_docs_option = "Isolated"
                    # search_docs_option = 'Combined'

                    is_search_internet = True

                    if is_search_docs:
                        if search_docs_option == ui.search_docs_options_isolated:
                            docs_response = seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources."""
                        elif search_docs_option == ui.search_docs_options_combinedd:
                            if is_search_internet:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                            elif not is_search_internet:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                    elif not is_search_docs:
                        if is_search_internet:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                        elif not is_search_internet:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls."""
                else:
                    for i in range(0, len(history) - 1):
                        chat_history.add_user_message(history[i].get("question"))
                        chat_history.add_ai_message(history[i].get("answer"))

                    chat_history.add_user_message(question)

                    chat_history_response = chat_history_chain()(
                        {"input": chat_history.messages},
                    )
                    chat_history_summary = chat_history_response["text"]

                    func_calling_response = func_calling_chain().run(
                        chat_history_summary
                    )

                    query = func_calling_response.get("query")

                    try:
                        created_at = json.loads(
                            func_calling_response.get("created_at", None)
                        )
                    except TypeError:
                        created_at = None

                    if is_search_docs:
                        if search_docs_option == ui.search_docs_options_isolated:
                            docs_response = seach_docs(query, top_k=16)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources. Current conversation:"{chat_history_summary}"""
                        elif search_docs_option == ui.search_docs_options_combined:
                            if is_search_internet:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                internet_results = search_internet(query)
                                docs_response.extend(internet_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                            elif not is_search_internet:
                                embedding_results = search_pinecone(
                                    query, created_at, top_k=8
                                )
                                docs_response = seach_docs(query, top_k=8)
                                docs_response.extend(embedding_results)
                                input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{docs_response}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                    elif not is_search_docs:
                        if is_search_internet:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            internet_results = search_internet(query)
                            embedding_results.extend(internet_results)
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""
                        elif not is_search_internet:
                            embedding_results = search_pinecone(
                                query, created_at, top_k=16
                            )
                            input = f"""Provide a clear, well-organized, and critically analyzed respond to the following question of "{question}" in its original language, while leveraging the information of "{embedding_results}". Do not return any prefix like "AI:". Give corresponding detailed sources with urls. Current conversation:"{chat_history_summary}"""

                # Send the message to the chain and feed the response back to the client
                await qa_chain.acall(
                    {
                        "input": input,
                    }
                )

            # Send the end-response back to the client
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("WebSocketDisconnect")
            # TODO try to reconnect with back-off
            break
        except ConnectionClosedOK:
            logging.info("ConnectionClosedOK")
            # TODO handle this?
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())
