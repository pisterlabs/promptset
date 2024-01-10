# 구 celeryapp/tasks.py
from celery import shared_task
import openai
from users.models import ChatHistory, UserRequestCount
from datetime import datetime


@shared_task
def process_openai_request(chat_history_id, user_message, user_id):
    # Check if the user has already reached their daily limit
    today = datetime.now().date()
    request_count, created = UserRequestCount.objects.get_or_create(
        user_id=user_id, date=today
    )

    # if exceeded, then user gets this message instead. No prompt deliver to openAI.
    if request_count.count >= 5:
        # User has reached the daily limit
        return "You have reached your daily limit of 5 requests."

    # system prompt
    predefined_prompt = {
        "role": "system",
        "content": "You are an anime expert. Your role is to listen to a user input and based on his/her expression, suggest any anime, light novel, visual novel or manga for this user.",
    }
    user_input = {"role": "user", "content": user_message}
    messages = [predefined_prompt, user_input]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        response_text = response["choices"][0]["message"]["content"]

        # Update the ChatHistory object with the OpenAI response
        chat_history = ChatHistory.objects.get(id=chat_history_id)
        chat_history.response = response_text
        chat_history.save()

        # Increment the user's request count
        request_count.count += 1
        request_count.save()

        return response_text
    except openai.error.OpenAIError as e:
        error_message = f"OpenAI error: {str(e)}, error details: {e.error}"
        # Consider how to handle errors, possibly setting the response to the error message
        chat_history = ChatHistory.objects.get(id=chat_history_id)
        chat_history.response = error_message
        chat_history.save()
        return error_message
    except Exception as e:
        error_message = f"Error processing OpenAI request: {str(e)}"
        # Handle error accordingly
        chat_history = ChatHistory.objects.get(id=chat_history_id)
        chat_history.response = error_message
        chat_history.save()
        return error_message
