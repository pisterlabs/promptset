import requests
import os
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class Highlight(BaseModel):
    start: int
    end: int
    highlight: str
    highlight_summary: str


class HighlightVideoResponse(BaseModel):
    id: str
    highlights: List[Highlight]


class HighlightVideoBody(BaseModel):
    video_id: str
    prompt: str
    type: str = "highlight"

def make_chat_completion_request(prompt: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a reasoning expert."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

class Client:
    def __init__(self):
        self.api_url = os.getenv("TWELVE_LABS_BASE_URL")
        self.headers = {"x-api-key": os.getenv("TWELVE_LABS_API_KEY")}

    def index_youtube_video(self, index_id: str, youtube_url: str):
        task_url = f"{self.api_url}/tasks/external-provider"
        data = {
            "index_id": index_id,
            "url": youtube_url,
        }
        response = requests.post(task_url, headers=self.headers, json=data)
        # you can get the video ID from video_id
        return response.json()

    def highlight_video(self, body: HighlightVideoBody) -> HighlightVideoResponse:
        url = f"{self.api_url}/summarize"
        data = body.dict()
        response = requests.post(url, headers=self.headers, json=data)
        return HighlightVideoResponse(**response.json())

    def summarize(self, body: HighlightVideoBody) -> HighlightVideoResponse:
        url = f"{self.api_url}/summarize"
        data = body.dict()
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
    

    def explain(self, highlight: str, prompt: str):
        response = make_chat_completion_request(
            f"""Purpose: {prompt}
    Highlight: {highlight}
    Can you explain why this highlight is relevant for my purpose.
    Give a concise reason in 1 or 2 sentences.
    Let's think step by step.
    Reason:"""
        )
        return response
    
    def chat_completion(self, message: str):
        openai_url = "https://api.openai.com/v1/engines/davinci-codex/completions"
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        }
        response = requests.post(openai_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
