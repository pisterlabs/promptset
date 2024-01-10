from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import argparse
import re
from dotenv import load_dotenv

load_dotenv()

MAX_INPUT_LENGTH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    if validate_length(user_input):
        print(f"User input: {user_input}")
        branding_result = generate_branding_snippet(user_input)
        keywords_result = generate_keywords(user_input)
        print(branding_result)
        print(keywords_result)
    else:
        raise ValueError(f"Input length is too long. Must be under {MAX_INPUT_LENGTH}. Submitted input is {user_input}")


def validate_length(prompt: str) -> bool:
    return len(prompt) <= 12


def generate_branding_snippet(prompt: str) -> str:
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate upbeat branding snippet for {prompt}:"
    print(enriched_prompt)

    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=enriched_prompt, temperature=0, max_tokens=32)

    # Extract output text.
    branding_text: str = response["choices"][0]["text"]
    branding_text = branding_text.strip()

    last_char = branding_text[-1]

    if last_char not in {".", "!", "?"}:
        branding_text += "..."
    print(f"Snippet: {branding_text}")
    return branding_text


def generate_keywords(prompt: str) -> list[str]:
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywords for {prompt}:"
    print(enriched_prompt)

    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=enriched_prompt, temperature=0, max_tokens=32)

    # Extract output text.
    keywords_text: str = response["choices"][0]["text"]

    # Strip whitespace
    keywords_text = keywords_text.strip()
    keywords_list = re.split(",|\n|;|-", keywords_text)
    keywords_list = [k.lower().strip() for k in keywords_list]
    keywords_list = [k for k in keywords_list if len(k) > 0]

    print(f"Keywords: {keywords_list}")
    return keywords_list


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to CopyKitt"}


@app.get("/generate_snippet")
async def generate_snippet_api(prompt: str):
    validate_input_length(prompt)
    snippet = generate_branding_snippet(prompt)
    return {"Snippet": f"{snippet}"}


@app.get("/generate_keywords")
async def generate_keywords_api(prompt: str):
    validate_input_length(prompt)
    keywords = generate_keywords(prompt)
    return {"Keywords": keywords}


@app.get("/generate_snippets_and_keywords")
async def generate_keywords_api(prompt: str):
    validate_input_length(prompt)
    snippet = generate_branding_snippet(prompt)
    keywords = generate_keywords(prompt)
    return {"snippet": snippet, "Keywords": keywords}


def validate_input_length(prompt: str):
    if len(prompt) >= MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input length is too long. Must be under {MAX_INPUT_LENGTH} characters.")


