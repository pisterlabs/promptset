import os
from content_generator.openai_handler import openai_prompt_runner
from fastapi import APIRouter

router = APIRouter()

@router.get("/content/hash_tag_generator/{title}")
async def hash_tag_generator(title: str):
    result = openai_prompt_runner(f"Generate 10 relevant hash tags for the following video title {title}, please avoid any kind of explanations other that the hash tag of the video")
    return {"hash_tags": result}

@router.get("/content/description_generator/{title}")
async def description_generator(title: str):
    result = openai_prompt_runner(f"Generate a relevant description for the following video title {title}, please avoid any kind of explanations other that the description of the video")
    return {"description": result}

@router.get("/content/script_generator/{title}/{time}")
async def script_generator(title:str,time:str):
    result = openai_prompt_runner(f"Create a well defined script for a YouTube video of {time} minutes on the title {title} Please avoid any type of explanation other than the script itself",)
    return {"script": result}

