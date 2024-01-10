import markdown
import openai
from celery import shared_task
from django.conf import settings
from markdown.extensions.fenced_code import FencedCodeExtension

from apps.chat.models import Chat, ChatMessage, MessageTypes
from apps.chat.serializers import ChatMessageSerializer


@shared_task(bind=True)
def get_chatgpt_response(self, chat_id: int, message: str) -> str:
    openai.api_key = settings.OPENAI_API_KEY
    chat = Chat.objects.get(id=chat_id)
    openai_response = openai.ChatCompletion.create(
        model=settings.OPENAI_MODEL,
        messages=chat.get_openai_messages(),
    )
    response_body = openai_response.choices[0].message.content.strip()
    message = ChatMessage.objects.create(
        chat_id=chat_id,
        message_type=MessageTypes.AI,
        content=response_body,
    )
    return ChatMessageSerializer(message).data


@shared_task
def set_chat_name(chat_id: int, message: str):
    chat = Chat.objects.get(id=chat_id)
    if not message:
        return
    elif len(message) < 20:
        # for short messages, just use them as the chat name. the summary won't help
        chat.name = message
        chat.save()
    else:
        # set the name with openAI
        openai.api_key = settings.OPENAI_API_KEY
        system_naming_prompt = """
    You are SummaryBot. When I give you an input, your job is to summarize the intent of that input.
    Provide only the summary of the input and nothing else.
    Summaries should be less than 100 characters long.
    """
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_naming_prompt,
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text: '{message}'",
                },
            ],
        )
        response_body = openai_response.choices[0].message.content.strip()
        chat.name = response_body[:100]
        chat.save()
