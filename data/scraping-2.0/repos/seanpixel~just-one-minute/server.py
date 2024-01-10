from fastapi import FastAPI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

anthropic = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

app = FastAPI()

def generate_paragraph():
    completion = anthropic.completions.create(
        model="claude-2.1",
        max_tokens_to_sample=500,
        prompt=f"{HUMAN_PROMPT} Generate some text that will take a minute to read. It can be about anything. Output only the text and nothing else. Don't say anything afterwards. {AI_PROMPT}",
    )
    paragraph = completion.completion.replace('"', "")
    if paragraph.startswith("\n"):
        paragraph = paragraph[1:]

    if paragraph.startswith(" "):
        paragraph = paragraph[1:]
    
    print(paragraph)

    return paragraph

@app.get("/paragraph")
def read_root():
    paragraph = generate_paragraph()
    return {"text": paragraph}