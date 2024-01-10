import openai
from config import settings
from fastapi import APIRouter
from pydantic import BaseModel


# Define models
class Question(BaseModel):
    question: str


class Reply(BaseModel):
    reply: str


# Setup API v1 router
v1 = APIRouter(prefix="/api/v1")

# Set OpenAI API key
openai.api_key = settings.openai_api_key


# Example assistant endpoint that uses GPT-4 to answer questions
@v1.post("/assistant")
def assistant(request: Question) -> Reply:
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": request.question},
        ],
        max_tokens=200,
    )

    answer = completion.choices[0].message.content
    return Reply(reply=answer)
