import openai
# from openai import AsyncOpenAI
from openai import AsyncAzureOpenAI
import asyncio
from utils import parse_json_output
from dotenv import load_dotenv
import os
from .prompts import VIDEO_SCRIPT_PROMPT,VIDEO_SCRIPT_JSON_OUTPUT,RELEVANT_DOCUMENT_FILTER_PROMPT,VIDEO_SCRIPT_EVALUATOR_PROMPT,VIDEO_SCRIPT_ENHANCER_PROMPT
print(load_dotenv('../.env'))

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"), 
    api_key=os.getenv("OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
)

def inquireChain(query:str) -> str:
    """So bascially askes"""

async def relevantDocumentFilter(relevant_documents:list[dict],query:str)->str:
    """Filters only the relevant documents by LLM"""

    # Concatenate all the documents into one string
    relevant_documents = [document["metadata"]["text"] for document in relevant_documents]
    # Async call to LLM for each document 

    prompt = RELEVANT_DOCUMENT_FILTER_PROMPT.format(documents = relevant_documents,userPrompt=query)
    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_API_ENGINE"),
        messages=[{"role":"system","content":f"Role:You are an assistant to complie all documents information are relevant to the user question: {query}"},
                  {"role": "user", "content": prompt}
                ],
        max_tokens = 1500
    )
    print(f'Tokens used for relevantDocumentFilter: {completion.usage}')
    return completion.choices[0].message.content

import asyncio

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"), 
    api_key=os.getenv("OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
)


async def generate_video(relevant_documents:str,query:str)->dict:
    """Generates a video script from the relevant documents and query
    Output should be a dict with the following keys:
        list_of_scenes: list[dict]
            scene: str
            subtitles: list[str]
    """

    prompt = VIDEO_SCRIPT_PROMPT.format(query=query,relevant_documents=relevant_documents,VIDEO_SCRIPT_JSON_OUTPUT=VIDEO_SCRIPT_JSON_OUTPUT)
    completion = await client.chat.completions.create(
        model="fintech-gpt4",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ],
        functions=[
            {
                "name": "format_video_script",
                "description": "Formats to a 30-45sec video script.",
                "parameters":{
                    "type": "object",
                    "properties": {
                    "list_of_scenes": {
                        "type": "array",
                        "description": "List of scenes for video script, there should be at least 6 scenes or more",
                        "items": {
                        "type": "object",
                        "properties": {
                            "scene": {
                            "type": "string",
                            "description": "Scene description for video should be visual and general. Max 5 words\nExample:family trip skiing | accident bike crash"
                            },
                            "subtitles": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "video subtitles script for video"
                            }
                            }
                        },
                        "required": ["scene", "subtitles"]
                        }
                    }
                    },
                    "required": ["list_of_scenes"]
                }
            }
        ],
        function_call={"name": "format_video_script"},
        max_tokens = 5000
    )
    print(f'Tokens used VIDEO_SCRIPT_PROMPT: {completion.usage}')
    print(f'VIDEO SCRIPT PROMPT:\n{prompt}\n------END-----',flush=True)

    return completion.choices[0].message


async def generate_video_agent(relevant_documents:str,query:str)->dict:
    """Generates a video script from the relevant documents and query
    Output should be a dict with the following keys:
        list_of_scenes: list[dict]
            scene: str
            subtitles: list[str]
    """

    prompt = VIDEO_SCRIPT_PROMPT.format(query=query,relevant_documents=relevant_documents,VIDEO_SCRIPT_JSON_OUTPUT=VIDEO_SCRIPT_JSON_OUTPUT)
    completion = await client.chat.completions.create(
        model="fintech-gpt4",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ],
        functions=[
            {
                "name": "format_video_script",
                "description": "Formats to a 30-45sec video script.",
                "parameters":{
                    "type": "object",
                    "properties": {
                    "list_of_scenes": {
                        "type": "array",
                        "description": "List of scenes for video script, there should be at least 6 scenes or more",
                        "items": {
                        "type": "object",
                        "properties": {
                            "scene": {
                            "type": "string",
                            "description": "Scene description for video should be visual and general. Max 5 words\nExample:family trip skiing | accident bike crash"
                            },
                            "subtitles": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "video subtitles script for video"
                            }
                            }
                        },
                        "required": ["scene", "subtitles"]
                        }
                    }
                    },
                    "required": ["list_of_scenes"]
                }
            }
        ],
        function_call={"name": "format_video_script"},
        max_tokens = 5000
    )
    print(f'Tokens used VIDEO_SCRIPT_PROMPT: {completion.usage}')
    print(f'VIDEO SCRIPT PROMPT:\n{prompt}\n------END-----',flush=True)

    video_script_json = parse_json_output(completion.choices[0].message)

    # Add FEEBACK PROMPT HERE
    prompt_feedback = VIDEO_SCRIPT_EVALUATOR_PROMPT.format(video_script=str(video_script_json))
    completion_feedback = await client.chat.completions.create(
        model=os.getenv("OPENAI_API_ENGINE"),
        temperature=0,
        # Add the previous video script as a message
        messages=[
            {"role": "user", "content": prompt},
            # {"role": "system","content":completion.choices[0].message.content},
            {"role": "user", "content": prompt_feedback}
        ],
        max_tokens = 1500
    )
    print(f"completion_feedback: {completion_feedback}")

    # Add Enhancer Prompt here
    prompt_enhancer = VIDEO_SCRIPT_ENHANCER_PROMPT.format(query=query,feedback=completion_feedback.choices[0].message.content,old_video_script=str(video_script_json))
    completion_video_script_enhanced = await client.chat.completions.create(
        model = "fintech-gpt4",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
            # {"role": "system","content":completion.choices[0].message.content},
            {"role": "user", "content": prompt_feedback},
            {"role":"system","content":completion_feedback.choices[0].message.content},
            {"role": "user", "content": prompt_enhancer}
        ],
        functions=[
            {
                "name": "format_video_script",
                "description": "Formats to a 30-45sec video script.",
                "parameters":{
                    "type": "object",
                    "properties": {
                    "list_of_scenes": {
                        "type": "array",
                        "description": "List of scenes for video script, there should be at least 6 scenes or more",
                        "items": {
                        "type": "object",
                        "properties": {
                            "scene": {
                            "type": "string",
                            "description": "Scene description for video should be visual and general. Max 5 words\nExample:family trip skiing | accident bike crash"
                            },
                            "subtitles": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "video subtitles script for video"
                            }
                            }
                        },
                        "required": ["scene", "subtitles"]
                        }
                    }
                    },
                    "required": ["list_of_scenes"]
                }
            }
        ],
        function_call={"name": "format_video_script"}

    )
    return completion_video_script_enhanced.choices[0].message