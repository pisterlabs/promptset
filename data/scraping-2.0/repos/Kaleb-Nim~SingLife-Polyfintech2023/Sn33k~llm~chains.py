import openai
# from openai import AsyncOpenAI
from openai import AsyncAzureOpenAI, AzureOpenAI
import asyncio

from dotenv import load_dotenv
import os
from .prompts import PDF_CONTEXT_SUMMARIZER_PROMPT
print(load_dotenv('../.env'))

client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"), 
    api_key=os.getenv("OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
)

def FormatKeyInfo(pdf_text_info:str)->str:
    """Generates a video script from the relevant documents and query
    Output should be a dict with the following keys:
        list_of_scenes: list[dict]
            scene: str
            subtitles: list[str]
    """

    prompt = PDF_CONTEXT_SUMMARIZER_PROMPT.format(pdf_information = pdf_text_info)
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_API_ENGINE"),
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    print(f'Tokens used VIDEO_SCRIPT_PROMPT: {completion.usage}')
    print(f'completion: {completion}')
    return completion.choices[0].message.content