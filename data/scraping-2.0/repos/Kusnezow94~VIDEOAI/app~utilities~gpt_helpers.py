import openai
from flask import current_app

def generate_prompt(channel_description, channel_name):
    """
    Create a prompt for GPT-3.5 to generate video ideas based on the channel's description and name.
    """
    prompt_text = f"Generate creative and engaging video ideas for YouTube Shorts based on the following channel:\n\nChannel Name: {channel_name}\nChannel Description: {channel_description}\n\nIdeas:"
    return prompt_text

def get_gpt3_response(prompt_text):
    """
    Send the prompt to GPT-3.5 and get the response.
    """
    try:
        openai.api_key = current_app.config['OPENAI_API_KEY']
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            max_tokens=150,
            n=3,
            stop=None,
            temperature=0.7
        )
        return response.choices
    except Exception as e:
        current_app.logger.error(f"Error in getting response from GPT-3: {e}")
        return None

def generate_video_script(idea):
    """
    Generate a script for a YouTube Short video based on the selected idea.
    """
    prompt_text = f"Create a detailed script for a YouTube Short video based on the following idea:\n\nIdea: {idea}\n\nScript:"
    return get_gpt3_response(prompt_text)

def generate_video_metadata(idea):
    """
    Generate title, description, and hashtags for a YouTube Short video based on the selected idea.
    """
    prompt_text = f"Create a compelling title, a brief description, and relevant hashtags for a YouTube Short video based on the following idea:\n\nIdea: {idea}\n\nTitle:\n\nDescription:\n\nHashtags:"
    return get_gpt3_response(prompt_text)