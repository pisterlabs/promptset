import os
import openai
import environ
from project.settings import BASE_DIR
from .models import Message

env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))
openai.api_key = env('OPENAI_API_KEY')


def take_answer(prompt, chat_id):
    """
    Generate an AI-generated answer based on the prompt and previous chat messages.

    This function retrieves the most recent 5 messages from the chat with the given ID,
    formats them as context for the AI model, appends the user's prompt, and sends it to
    the OpenAI Completion API to generate a response. The generated response is extracted,
    cleaned, and returned as the answer.

    Args:
        prompt (str): The user's prompt.
        chat_id (int): The ID of the chat.

    Returns:
        str: The AI-generated answer.
    """

    # Retrieve the most recent 5 messages from the chat
    previous_messages = Message.objects.filter(chat_id=chat_id).order_by('-id')[:5]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    for message in previous_messages:
        user_message = {"role": "user", "content": message.message_text}
        # ai_message = {"role": "assistant", "content": message.answer_text}
        # messages.extend([user_message, ai_message])

    user_prompt = {"role": "user", "content": prompt}
    messages.append(user_prompt)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = response.choices[0].message.content.strip()
    answer = answer.replace("AI:", "").strip()

    return answer
