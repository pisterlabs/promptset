from openai import ChatCompletion
from fastapi import HTTPException, status
from app.models.schemas import EmailPrompt
from app.config.settings import Settings

settings = Settings()

def generate_email(email_prompt: str) -> str:
    openai_api_key = settings.openai_api_key
    
    req = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": email_prompt}],
        api_key=openai_api_key
    )
    
    if req:
        email_output = req.choices[0].message.content
        # email_split = email_output.split("\n")
        # for i in range(len(email_split)):
        #     email_split[i] = "<p>" + email_split[i] + "</p>"
        # email_output = "".join(email_split)
        return email_output
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid Request"
        )
