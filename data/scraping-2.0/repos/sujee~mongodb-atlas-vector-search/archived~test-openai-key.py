import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pprint

_ = load_dotenv(find_dotenv()) # read local .env file
# print ("OPENAI_API_KEY : ", os.environ.get('OPENAI_API_KEY') )

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key= os.environ.get("OPENAI_API_KEY"),
)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

pprint.pprint (chat_completion.model_dump_json(indent=4))