
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
import openai

app = FastAPI()

# Set OpenAI API Key
openai.api_key = "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"

class DietQuery(BaseModel):
    message: str

def get_diet_suggestion(message: str):
    prompt = "suggest an indian diet with the following conditions " + message
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


from fastapi import Query

@app.get('/suggest_diet')
def suggest_diet(message: str = Query(..., description="Diet conditions and preferences")):
    suggestion = get_diet_suggestion(message)
    return {"suggestion": suggestion}
