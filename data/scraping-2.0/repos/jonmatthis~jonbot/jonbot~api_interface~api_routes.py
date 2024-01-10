import asyncio
import base64
import json
from typing import Optional, TYPE_CHECKING, Any

from fastapi import FastAPI
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket

from jonbot.backend.controller.controller import Controller
from jonbot.backend.data_layer.models.conversation_models import ChatRequest, ImageChatRequest
from jonbot.backend.data_layer.models.database_request_response_models import UpsertDiscordMessagesRequest, \
    UpsertResponse, ContextMemoryDocumentRequest, UpsertDiscordChatsRequest
from jonbot.backend.data_layer.models.health_check_status import HealthCheckResponse
from jonbot.backend.data_layer.models.user_stuff.memory.context_memory_document import ContextMemoryDocument
from jonbot.backend.data_layer.models.voice_to_text_request import VoiceToTextRequest, VoiceToTextResponse

from jonbot.system.setup_logging.get_logger import get_jonbot_logger

if TYPE_CHECKING:
    from jonbot.backend.backend_database_operator.backend_database_operator import (
        BackendDatabaseOperations,
    )

logger = get_jonbot_logger()

HEALTH_ENDPOINT = "/health"

CHAT_ENDPOINT = "/chat"
IMAGE_CHAT_ENDPOINT = "/image_chat"
VOICE_TO_TEXT_ENDPOINT = "/voice_to_text"

UPSERT_MESSAGES_ENDPOINT = "/upsert_messages"
UPSERT_CHATS_ENDPOINT = "/upsert_chats"

GET_CONTEXT_MEMORY_ENDPOINT = "/get_context_memory"

VECTOR_SEARCH_ENDPOINT = "/vector_search"

CHAT_STATELESS_ENDPOINT = "/chat_stateless"


class StreamingPassthroughToWebsocketHandler(AsyncCallbackHandler):
    def __init__(self, websocket, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        token_output = {"type": "token", "content": token}
        try:
            await self.websocket.send_text(json.dumps(token_output))
        except Exception as e:
            logger.debug(f"Failed to send token: {token}, error: {str(e)}")
        pass


def register_api_routes(
        app: FastAPI,
        database_operations: "BackendDatabaseOperations",
        controller: Controller
):
    @app.get(HEALTH_ENDPOINT, response_model=HealthCheckResponse)
    async def health_check_endpoint():
        return HealthCheckResponse(status="alive")

    @app.get(GET_CONTEXT_MEMORY_ENDPOINT, response_model=Optional[ContextMemoryDocument])
    async def get_context_memory_endpoint(
            get_request: ContextMemoryDocumentRequest,
    ) -> ContextMemoryDocument:
        response = await database_operations.get_context_memory_document(
            request=get_request
        )

        if not response.success:
            logger.warning(
                f"Could not load context memory from database for context route: {get_request.query}"
            )
            return

        return response.data

    @app.post(VOICE_TO_TEXT_ENDPOINT, response_model=VoiceToTextResponse)
    async def voice_to_text_endpoint(
            voice_to_text_request: VoiceToTextRequest,
    ) -> VoiceToTextResponse:
        response = await controller.transcribe_audio(
            voice_to_text_request=voice_to_text_request
        )
        if response is None:
            return VoiceToTextResponse(success=False)
        return response

    @app.post(CHAT_ENDPOINT)
    async def chat_endpoint(chat_request: ChatRequest):
        logger.info(f"Received chat request: {chat_request}")
        return StreamingResponse(
            controller.get_response_from_chatbot(chat_request=chat_request),
            media_type="text/event-stream",
        )

    @app.post(IMAGE_CHAT_ENDPOINT)
    async def image_analysis_endpoint(image_analysis_request: ImageChatRequest) -> AIMessage:
        logger.info(f"Received image analysis request: {image_analysis_request}")
        return await controller.analyze_image(image_chat_request=image_analysis_request)

    @app.post(UPSERT_MESSAGES_ENDPOINT)
    async def upsert_messages_endpoint(
            request: UpsertDiscordMessagesRequest,
    ) -> UpsertResponse:
        return await database_operations.upsert_discord_messages(request=request)

    @app.post(UPSERT_CHATS_ENDPOINT)
    async def upsert_chats_endpoint(
            request: UpsertDiscordChatsRequest,
    ) -> UpsertResponse:
        response =  await database_operations.upsert_discord_chats(request=request)
        if not response.success:
            logger.error(f"Failed to upsert chats: {request}")
        return response


    @app.websocket(CHAT_STATELESS_ENDPOINT)
    async def chat_stateless_endpoint(websocket: WebSocket):

        user_id = websocket.query_params.get('id')
        crude_api_token = websocket.query_params.get('token')
        decoded_crude_api_token = base64.b64decode(crude_api_token.encode()).decode()

        if decoded_crude_api_token != 'ggbotapi-1199299301957388239120':
            await websocket.close(code=1000)
            return

        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()

                logger.info(f"Received on socket: {raw}")

                data = json.loads(raw)

                from langchain.llms import OpenAI
                from langchain.chat_models import ChatOpenAI
                from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

                model = OpenAI(
                    model_name=data['config']['model_name'],
                    temperature=data['config']['temperature'],
                    streaming=True,
                    callbacks=[StreamingPassthroughToWebsocketHandler(websocket=websocket)],
                    verbose=True
                )

                logger.info(f"System prompt: {data['system_prompts']}")

                system_prompt = "\n\n".join(data['system_prompts']) + "\n\n--------\n\n"

                from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])

                # build up the memory
                from langchain.memory import ConversationTokenBufferMemory, ConversationBufferWindowMemory
                # memory = ConversationTokenBufferMemory(
                memory = ConversationBufferWindowMemory(
                    # llm = model,
                    k=10,
                    return_messages=True,
                    # max_token_limit=4000
                )

                for msg in data['conversation_history']:
                    if msg["role"] == 'human':
                        memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == 'ai':
                        memory.chat_memory.add_ai_message(msg["content"])

                logger.info(f"Memory contents: {memory.load_memory_variables({})}")

                from langchain.chains import ConversationChain
                chain = ConversationChain(
                    llm=model,
                    memory=memory,
                    prompt=prompt,
                    verbose=True
                )

                # ai_response = await chain.apredict(input=data["new_user_input"] + "\nAI: ")

                ai_response = "wowwweeee!!"
                logger.info(f"Response: {ai_response}")

                # response = controller.get_response_from_chatbot(chat_request=chat_request)
                response = {'type': 'ai_response', 'content': ai_response}

                asyncio.create_task(websocket.send_text(json.dumps(response)))

        except Exception as e:
            logger.error(f"Error on websocket: {str(e)}")
            logger.trace(f"Error on websocket: {str(e)}", exc_info=True)
