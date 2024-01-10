import os
import openai
from dotenv import load_dotenv
from flask import current_app

load_dotenv()

def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')

def create_prompt_for_ideas(channel_description, channel_name):
    prompt = f"Generate three creative and engaging video ideas for YouTube Shorts based on the channel description: '{channel_description}' and channel name: '{channel_name}'."
    return prompt

def generate_video_ideas(channel_description, channel_name):
    openai.api_key = get_openai_api_key()
    prompt = create_prompt_for_ideas(channel_description, channel_name)
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=3,
            stop=None,
            temperature=0.7
        )
        ideas = [idea['text'].strip() for idea in response.choices]
        return ideas
    except openai.error.OpenAIError as e:
        current_app.logger.error(f"OpenAI API error: {e}")
        return []

def create_prompt_for_script(title):
    prompt = f"Create a detailed script for a YouTube Short video titled: '{title}'. Include a compelling opening, engaging content, and a call-to-action."
    return prompt

def generate_video_script(title):
    openai.api_key = get_openai_api_key()
    prompt = create_prompt_for_script(title)
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            stop=None,
            temperature=0.7
        )
        script = response.choices[0].text.strip()
        return script
    except openai.error.OpenAIError as e:
        current_app.logger.error(f"OpenAI API error: {e}")
        return ""