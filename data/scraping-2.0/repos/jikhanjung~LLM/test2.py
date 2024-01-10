from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
load_dotenv()
import os

api_key=os.environ.get("OPENAI_API_KEY")
print("api_key",api_key)

client = OpenAI()

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:personal::8KLC3aEp",
  messages=[
    {"role": "system", "content": "You are an expert on trilobite taphonomy. Answer with citation in the sentence if possible."},
    {"role": "user", "content": "Please summarize Hesselbo's (1987) study on trilobite taphonomy."},
  ]
)

print(completion.choices[0].message) 
