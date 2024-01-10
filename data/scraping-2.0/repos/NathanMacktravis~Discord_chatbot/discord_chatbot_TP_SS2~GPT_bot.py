import openai
from openai import ChatCompletion, OpenAI, AsyncOpenAI


OPENAI_API_KEY = "YOUR_OPENAI_TOKEN"

# Initialisation du client OpenAI
openai.api_key = OPENAI_API_KEY
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Function to get response from OpenAI
async def openai_chatbot(user_message):
    openai_response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
    )
    return openai_response['choices'][0]['message']['content']

