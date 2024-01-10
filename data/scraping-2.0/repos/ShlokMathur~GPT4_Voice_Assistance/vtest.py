import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle

async def main():

    model = whisper.load_model("base")
    result = model.transcribe("test.mp3")
    user_input = result["text"]
    print(f"You said: {user_input}")

    bot = Chatbot(cookiePath='cookies.json')
    response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)

    for message in response["item"]["messages"]:
        if message["author"] == "bot":
            bot_response = message["text"]

    bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
    # Select only the bot response from the response dictionary
    for message in response["item"]["messages"]:
        if message["author"] == "bot":
            bot_response = message["text"]
    # Remove [^#^] citations in response
    bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
    print(bot_response)
    await bot.close()

if __name__ == "__main__":
    asyncio.run(main())
