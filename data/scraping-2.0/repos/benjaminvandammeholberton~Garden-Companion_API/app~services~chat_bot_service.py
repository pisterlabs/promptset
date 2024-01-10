"""
Service class for interacting with the ChatBot using OpenAI's GPT-3.5-turbo model.

Class: ChatBotService

Methods:
- send_answer(data: dict) -> dict:
    Sends a user input to the ChatBot model and retrieves the generated answer.

    Parameters:
    - data (dict): A dictionary containing user input under the key "user-input".

    Returns:
    dict: The generated answer from the ChatBot model.

    Note:
    The system message provides information about the ChatBot's expertise in vegetable gardens.
    The user's input is included in the conversation for context.

- increase_request(current_user: User):
    Increases the request counters and updates the last request datetime for a given user.

    Parameters:
    - current_user (User): The user for whom the request counters need to be updated.

    Note:
    The counters "chat_bot_day_requests" and "chat_bot_total_requests" are incremented by 1.
    The last request datetime is updated to the current timestamp.
"""

from datetime import datetime
from uuid import UUID
from openai import AsyncOpenAI

from app.models.user_model import User

client = AsyncOpenAI()

class ChatBotService:
    """
    Service class for interacting with the ChatBot using OpenAI's GPT-3.5-turbo model.
    """
    @staticmethod
    async def send_answer(data: dict) -> dict:
        """
        Sends a user input to the ChatBot model and retrieves the generated answer.

        Parameters:
        - data (dict): A dictionary containing user input under the key "user-input".

        Returns:
        dict: The generated answer from the ChatBot model.

        Note:
        The system message provides information about the ChatBot's expertise in vegetable gardens.
        The user's input is included in the conversation for context.
        """
        system_message = {"role": "system", "content": "You are a french assistant about vegetable\
                          gardens. If the user asks a question unrelated to gardening, letting them\
                          know that your expertise is in vegetable gardens."}
        completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            system_message,
            {"role": "user", "content": f'${data["user-input"]}'}
        ],
        temperature=0.1,
        )
        return(completion.choices[0].message.content)
    
    @staticmethod
    async def increase_request(current_user: User):
        """
        Increases the request counters and updates the last request datetime for a given user.

        Parameters:
        - current_user (User): The user for whom the request counters need to be updated.

        Note:
        The counters "chat_bot_day_requests" and "chat_bot_total_requests" are incremented by 1.
        The last request datetime is updated to the current timestamp.
        """
        await current_user.update({"$inc": {"chat_bot_day_requests": 1, "chat_bot_total_requests": 1},
            "$set": {"last_request_datetime": datetime.now()}})
        await current_user.save()
    
    @staticmethod
    async def get_number_of_requests_allowed(current_user: User):
        today = datetime.now().date()
        if current_user.last_request_datetime.date() != today:
            current_user.last_request_datetime = datetime.now()
            current_user.chat_bot_day_requests = 0
            await current_user.save()
            return {"chat_bot_day_requests": current_user.chat_bot_day_requests}
        else:
            return {"chat_bot_day_requests": current_user.chat_bot_day_requests}
