import uuid
import openai
from openai import ChatCompletion
from starlette.responses import StreamingResponse
from src.app.chat.constants import ChatRolesEnum
from src.app.chat.models import BaseMessage, Message
from src.app.core.logs import logger
from src.app.settings import settings
from src.app.chat.streaming import stream_generator, format_to_event_stream
from src.app.chat.constants import ChatRolesEnum, NO_DOCUMENTS_FOUND
from src.app.chat.exceptions import RetrievalNoDocumentsFoundException
from src.app.chat.retrieval import process_retrieval
from src.app.db import messages_queries
from src.app.chat.models import BaseMessage, Message, ChatSummary
from uuid import UUID


class OpenAIService:
    @classmethod
    async def chat_completion_without_streaming(cls, input_message: BaseMessage) -> Message:
        completion: openai.ChatCompletion = await openai.ChatCompletion.acreate(
            model=input_message.model,
            api_key=settings.OPENAI_API_KEY,
            messages=[{"role": ChatRolesEnum.USER.value,
                       "content": input_message.augmented_message}]
        )

        completion = cls.extract_response_from_completion(
            chat_completion=completion)

        try:
            message_id = uuid.uuid4()
            messages_queries.insert_message(
                id=str(message_id),
                chat_id=str(input_message.chat_id),
                model=str(input_message.model),
                user_id=str(input_message.userId),
                agent_role=str(ChatRolesEnum.ASSISTANT.value),
                user_message=str(input_message.user_message),
                answer=str(completion)
            )
            logger.info("Message inserted to db successfully")

        except Exception as e:
            logger.error("Error while inserting message to db: " + str(e))
        return Message(
            model=input_message.model,
            message=completion,
            role=ChatRolesEnum.ASSISTANT.value
        )

    @staticmethod
    async def chat_completion_with_streaming(input_message: BaseMessage) -> StreamingResponse:
        subscription: openai.ChatCompletion = await openai.ChatCompletion.acreate(
            model=input_message.model,
            api_key=settings.OPENAI_API_KEY,
            messages=[{"role": ChatRolesEnum.USER.value,
                       "content": input_message.user_message}],
            stream=True,
        )
        return StreamingResponse(stream_generator(subscription), media_type="text/event-stream")

    @staticmethod
    def extract_response_from_completion(chat_completion: ChatCompletion) -> str:
        return chat_completion.choices[0]["message"]["content"]

    @classmethod
    async def qa_without_stream(cls, input_message: BaseMessage) -> Message:
        try:
            augmented_message = process_retrieval(
                message=input_message)
            logger.info("Context retrieved successfully.")
            return await cls.chat_completion_without_streaming(input_message=augmented_message)
        except RetrievalNoDocumentsFoundException:
            return Message(model=input_message.model, message=NO_DOCUMENTS_FOUND, role=ChatRolesEnum.ASSISTANT.value)

    @classmethod
    async def qa_with_stream(cls, input_message: BaseMessage) -> StreamingResponse:
        try:
            augmented_message: BaseMessage = process_retrieval(
                message=input_message)
            logger.info("Context retrieved successfully.")
            return await cls.chat_completion_with_streaming(input_message=augmented_message)
        except RetrievalNoDocumentsFoundException:
            return StreamingResponse(
                (format_to_event_stream(y) for y in "Not found"),
                media_type="text/event-stream",
            )


class ChatServices:
    @classmethod
    def get_messages(cls, user_id: str) -> list[Message]:
        return [Message(**message) for message in messages_queries.select_messages_by_user(user_id=user_id)]

    @classmethod
    def get_chats(cls, user_id: str) -> list[ChatSummary]:
        return [ChatSummary(**chat) for chat in messages_queries.select_chats_by_user(user_id=user_id)]

    @classmethod
    def get_chat_messages(cls, chat_id) -> list[Message]:
        return [Message(**message) for message in messages_queries.select_messages_by_chat(chat_id=chat_id)]

    @classmethod
    def get_chat(cls, chat_id) -> ChatSummary:
        return ChatSummary(**messages_queries.select_chat_by_id(chat_id=chat_id))

    @classmethod
    async def create_chat(cls, chat: ChatSummary):
        try:
            messages_queries.insert_chat(
                id=str(chat.id),
                user_id=str(chat.user_id),
                title=chat.title,
                model=chat.model,
                agent_role=str(ChatRolesEnum.ASSISTANT.value),
                created_at=chat.created_at,
                updated_at=chat.updated_at,
            )
            logger.info("chat before returning to frontend: " + str(chat))
            return chat
        except Exception as e:
            logger.error(f"Error creating chat: {e}")

    @classmethod
    def create_message(cls, message) -> Message:
        return Message(**messages_queries.insert_message(message=message))
