<<<<<<< HEAD
import os 
import openai

openai.api_key = os.getenv("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")
openai.Model.list()
=======
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("API_KEY")

## 

from openai import OpenAI

client = OpenAI(
   
    api_key=api_key,
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

# response['choices'][0]['message']['content']
#print(chat_completion.choices[0].message)

print(chat_completion.choices[0].message.content)
>>>>>>> d9906bf2862ac7a37930dd5816637dac8897f8b0
