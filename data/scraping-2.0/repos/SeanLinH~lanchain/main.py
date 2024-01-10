import os
from typing import Optional
from fastapi import FastAPI, Form
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic(
   # defaults to 
    api_key=os.environ.get("API_CLAUD")
)

app = FastAPI()


@app.post("/chat/complete")
def read_item(text: str=Form(...)):
    comp = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {text}{AI_PROMPT}",
    )
    return {"messages":comp.completion}
