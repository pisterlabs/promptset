from django.db import DatabaseError
from openai.error import OpenAIError
import logging
from .models import ChatSession, ChatMessage

class ChatService:
    @staticmethod
    def process_user_message(user, user_message, openai_service=None):
        if openai_service is None:
            openai_service = OpenAIService()

        try:
            chat_session, created = ChatSession.objects.get_or_create(user=user)
            llm_response = chat_session.generate_response(user_message, openai_service)
            chat_message = ChatMessage.objects.create(
                user=user,
                user_message=user_message,
                model_response=llm_response
            )
            chat_session.add_message(role="user", content=user_message)
            chat_session.add_message(role="model", content=llm_response)
            chat_session.save()
            return chat_message
        except DatabaseError as e:
            logging.exception(f"Database error: {e}")
            raise DatabaseError(f"Database error: {e}") from e
        except OpenAIError as e:
            logging.exception(f"OpenAI error: {e}")
            raise OpenAIError(f"OpenAI error: {e}") from e
        except Exception as e:
            logging.exception(f"Unexpected error: {e}")
            raise
