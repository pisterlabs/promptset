from imp import reload
import os

import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dataclasses
from typing import List, Dict, Tuple, Any
from main import *
from glob2 import glob
import json
from fastapi.staticfiles import StaticFiles



load_dotenv()
openai.api_key = os.getenv("OPENAI_ACCESS_KEY")


app = FastAPI()

app.mount("/assets/data", StaticFiles(directory="data"), name="data")



class Story(BaseModel):
    path: str
    language: str = "en"
    medias: List[Dict] = None

from typing import Optional, Union
class Video(BaseModel):
    path: str = ""
    text: str = ""
    media: str = ""
    actor: str = ""
    output: Optional[str] =None
    duration: Union[float, int] = 0.1
    generated: Optional[bool] = None
    raw: str = ""
    voice: str = ""
    src: Optional[str] = None
    texts: Any = None
    wait_until_finished: bool = False
    speed: float = 100
    
    def compute_duration(self):
        texts = self.text.split("|")
        self.texts = []
        self.duration = 0
        animation = self.media.lower()
        for slide in texts:
            try:
                slide, slide_duration = slide.split("t=")
            except:
                if slide == "" or not self.wait_until_finished:
                    slide_duration = 0.1
                else:
                    slide_duration = 7
            slide_duration = float(slide_duration)
            self.duration += slide_duration
            self.texts.append({
                "text": slide,
                "duration": slide_duration,
                "animation": animation
            })
            if animation == "input":
                animation = "typing"
        if self.wait_until_finished:
            self.texts.append({
                "text": "",
                "duration": 0.1,
                "animation": "video"
            })
            



display = []
default_text = ""
@app.get("/display")
async def displayBeamer():
    global display
    if len(display) > 0:
        response = display.pop(0)
    else:
        response = Video(text=default_text, media="video")
    response.compute_duration()
    print(response)
    return response



origins = [
    "http://localhost:*",
    "http://localhost:3000",
    "https://pollinations.ai",
    "https://*.pollinations.ai",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"healthy": "yes"}


@app.post("/open")
def load_story(story: Story):
    if story.medias is None and story.path.endswith(".txt"):
        parts = parse_template(story.path)
        story = Story(
            path=story.path,
            medias = [dataclasses.asdict(i) for i in parts],
        )
    elif story.path.endswith(".json"):
        with open(story.path) as f:
            story.medias = [Part(**i).__dict__ for i in json.load(f)]
    return story


@app.post("/generate")
async def generate(template: Story) -> Story:
    """
    Generate a story from a template.
    """
    target = ".".join(template.path.split(".")[:-1]) + "/generated"
    os.makedirs(target, exist_ok=True)
    #if template.language != "en":
    #    story = en_to_de(story)
    #story_en = fill_template_gpt3([Part(**i) for i in template.medias])
    #story_de = translate_parts(story_en)
    story_de = text_to_media([Part(**i) for i in template.medias], target=target)
    story_de = Story(path=template.path, medias=[i.__dict__ for i in story_de], language="de")
    return story_de


def answer_audience_questions(story_de):
    i = 0
    while i < len(story_de):
        part = story_de[i]
        if part.actor.lower() == "audience" and part.text != "" and part.text is not None:
            # if the next part is the answer skip
            if len(story_de) > i + 1 and story_de[i+1].actor == "GPT":
                i += 1
                continue
            answer = generate_answer(part.text)
            story_de.insert(i + 1, answer)
        i += 1
    return story_de


@app.post("/save")
async def save(template: Story) -> Story:
    """
    Save a story and generate missing medias.
    """
    parts = [Part(**i) for i in template.medias]
    if template.path.endswith("medias.json"):
        target = "/".join(template.path.split("/")[:-1])
    else:
        target = ".".join(template.path.split(".")[:-1]) + "/generated"
    
    # if template.language != "de":
    #     parts = translate_parts(parts)
    print("yo")
    parts = answer_audience_questions(parts)
    print("yo yo yo")
    story_de = text_to_media(parts, target=target)
    story_de = Story(path=target + "/medias.json", medias=[i.__dict__ for i in story_de])
    return story_de


def en_to_de(story):
    pass


def de_to_en(story):
    pass


@app.post("/play")
async def play(part: Video) -> Video:
    # os.system(f"ffplay -fs -autoexit '{video.path}'")
    global display
    global default_text
    part.compute_duration()
    default_text = part.texts[-1]["text"]
    display += [part]
    return part


@app.get("/available")
async def available():
    files = glob(f"data/*.txt") + glob(f"data/**/medias.json")
    return files


def main():
    """
    Run the server.
    """
    uvicorn.run("backend:app", host="0.0.0.0", port=8726, reload=True)


if __name__ == "__main__":
    main()
