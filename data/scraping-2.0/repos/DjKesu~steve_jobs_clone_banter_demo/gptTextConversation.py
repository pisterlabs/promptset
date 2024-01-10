import os
import openai
from tts import play_text_as_audio
from dotenv import load_dotenv
load_dotenv()

def createConversation(text):
    openai.api_key = os.getenv("openai-api-key")

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are Steve Jobs.'},
            {'role': 'user', 'content': text}
        ],
        temperature=0.7,
        stream=True
    )
    completed_text = ""
    for chunk in response:
        completed_text += chunk.choices[0].delta['content']
        play_text_as_audio(chunk.choices[0].delta['content'])

if __name__ == "__main__":
    user_input = input("Enter your message: ")
    createConversation(user_input)
