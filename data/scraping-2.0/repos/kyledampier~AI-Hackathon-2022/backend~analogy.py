import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

#
def getAnalogy(analogy_target, text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Generate an analogy about {analogy_target} from the following text: \n{text}\n",
        max_tokens=150,
        temperature=0.9
    )

    x = response.choices[0]['text'].replace("\n", " ")
    return x
    



