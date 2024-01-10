from openai import OpenAI
from rich.console import Console
import os
from dotenv import load_dotenv
import datetime

# console
console = Console()

# load .env file
load_dotenv()

# openai api key
client = OpenAI(api_key=os.getenv('open_ai_api_key'))

# function is going to take the conversation 
def OAIapiCall(conversation):
    # set the model to use.
    model = 'gpt-3.5-turbo-1106' # 'gpt-3.5-turbo-1106'
    temperature = 0,

    response = client.chat.completions.create(
        model=model,
        messages=conversation,
    )

    res =  response.choices[0].message.content.strip()
    return res

