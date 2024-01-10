import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
 
openAIKey = os.getenv('OPENAI_KEY')

client = OpenAI(api_key=openAIKey)

chat_completion = client.chat.completions.create(
    model='gpt-4',
    messages=[{'role': 'user', 'content': 'Hello World'}]

)

print(chat_completion)
