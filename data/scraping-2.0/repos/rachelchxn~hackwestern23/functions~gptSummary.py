from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())


client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

def getSummary(dialogue):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            # "content": f'''Extract the name and key points about the person rachel is talking to from this conversation dialogue:
            "content": f'''Extract the name and key points about the person I am talking to from this conversation dialogue:
                {dialogue}
                '''
            },
            {
            "role": "user",
            "content": ""
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content
