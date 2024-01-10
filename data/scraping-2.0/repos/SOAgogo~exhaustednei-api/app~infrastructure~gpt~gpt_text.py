from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

if len(sys.argv) > 1:
    client = OpenAI()
    message_from_ruby = sys.argv[1]
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {"role": "system", "content": message_from_ruby},
      ]
    )
    print(completion.choices[0].message)
else:
    print("No message received from Ruby.")






