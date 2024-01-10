from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from openai import OpenAI,File
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

client = OpenAI()
AIGOLEMBASE = ''
class Medidetect(BaseModel):
    prompt: str
    base: Union[str, None] = None    
    filter: Union[str, None] = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static",html=True), name="static")
origins = [
    "https://egolem.online",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#import os
#openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/",response_class=HTMLResponse)
def read_root():
    return "<meta http-equiv='Refresh' content='0; url=/static/index.html' />"


@app.post("/items/")
async def create_medidetect(item: Medidetect):
    aiprompt = ''

    aiprompt = item.prompt
    completion = client.chat.completions.create(
      model="gpt-4-1106-preview",
        messages=[
        {"role": "system", "content": "You are factual assistant, you follow scientific facts and answers are brief."},
        {"role": "user", "content": aiprompt}
    ])

    response = completion.choices[0].message

    print(response)
    return response.content

