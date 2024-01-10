from revChatGPT.Official import Chatbot
from asgiref.sync import sync_to_async
from env import OPENAI_KEY


chatbot = Chatbot(api_key=OPENAI_KEY)

async def handle_response(message) -> str:
    response = await sync_to_async(chatbot.ask)(message)
    responseMessage = response["choices"][0]["text"]

    return responseMessage