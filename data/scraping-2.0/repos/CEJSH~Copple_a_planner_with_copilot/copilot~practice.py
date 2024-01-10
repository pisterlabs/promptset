from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import uvicorn  # You were missing this import

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Define prompt
prompt = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are a schedule summary. Displays a summary of your schedule in the latest order based on today's date.
"""

class InputText(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(input_text: InputText):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt}\nUser: {input_text.text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return {"summary": response.choices[0].text.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
