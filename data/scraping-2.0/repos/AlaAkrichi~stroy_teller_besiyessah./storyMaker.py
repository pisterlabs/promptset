from openai import OpenAI , OpenAIError
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise OpenAIError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)
def makeStory(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "system", "content": "Here is a story based on your prompt:"},
        ]
        
    )
    story = response.choices[0].message.content
    return story