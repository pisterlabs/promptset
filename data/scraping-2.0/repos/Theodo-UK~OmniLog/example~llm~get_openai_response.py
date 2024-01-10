import os
from dotenv import load_dotenv
import openai
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        f"OPENAI_API_KEY not found: {OPENAI_API_KEY}\nMake sure that you have an OPENAI_API_KEY value in the .env file."
    )

openai.api_key = OPENAI_API_KEY


def get_openai_response(prompt: str) -> dict:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
    )

    return {
        "datetime_utc": datetime.utcfromtimestamp(response.created),
        "output": response.choices[0].text,
        "total_tokens": response.usage.total_tokens,
    }
