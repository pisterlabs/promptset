from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
from decouple import config

router = APIRouter()

SECRET_KEY = config("SECRET_KEY")

# Replace with your OpenAI API key
openai.api_key = SECRET_KEY


class EmailRequest(BaseModel):
    recipient_name: str
    recipient_email: str
    subject: str
    keywords: list[str]
    length: int


@router.post("/")
async def generate_email(email_request: EmailRequest):
    try:
        # Generate a personalized email using ChatGPT
        prompt = f"Compose a personalized email to {email_request.recipient_name}, about {email_request.subject}, add these keywords, {', '.join(email_request.keywords)}"

        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=prompt,
            max_tokens=email_request.length,
        )

        email_content = response.choices[0].text.strip()

        # In a real application, you might want to send this email_content to an email service for delivery.
        return {"email_content": email_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
