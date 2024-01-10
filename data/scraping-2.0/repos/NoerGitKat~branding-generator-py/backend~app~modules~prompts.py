from os import getenv
import openai
from dotenv import load_dotenv
from typing import List
import re


load_dotenv()

openai.organization = getenv("OPENAI_ORG_ID")
openai.api_key = getenv("OPENAI_API_KEY")


def generate_branding_snippet(prompt: str) -> str:
    enriched_prompt = f"Generate branding snippet for {prompt}"
    print(f"Prompt: {enriched_prompt}")

    response = openai.Completion.create(
        model="text-ada-001",
        max_tokens=10,
        prompt=enriched_prompt,
        temperature=0.6
    )

    # Extract output text
    branding_text = response["choices"][0]["text"]

    # Strip whitespace
    branding_text = branding_text.strip()

    # Add ... to truncated statements
    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    print(f"Result: {branding_text}")

    return branding_text


def generate_keywords(prompt: str) -> List[str]:
    enriched_prompt = f"Generate branding keywords for {prompt}"
    print(f"Prompt: {enriched_prompt}")
    response = openai.Completion.create(
        model="text-ada-001",
        max_tokens=10,
        prompt=enriched_prompt,
        temperature=0.6
    )

    # Extract output text
    keywords_text = response["choices"][0]["text"]

    # Strip whitespace
    keywords_text.strip()

    # Split into list
    keywords = re.split(",|\n|;|-", keywords_text)
    keywords = [keyword.lower().strip() for keyword in keywords]
    keywords = [keyword for keyword in keywords if len(keyword) > 0]

    print(f"Result: {keywords}")

    return keywords
