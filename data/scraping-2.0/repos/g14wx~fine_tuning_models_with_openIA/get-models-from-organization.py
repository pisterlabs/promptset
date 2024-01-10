import json

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)

response = client.models.list()
for datum in response.data:
    print("----------------------------------")
    print(f"{json.dumps(datum.model_dump(), indent=4)}")
