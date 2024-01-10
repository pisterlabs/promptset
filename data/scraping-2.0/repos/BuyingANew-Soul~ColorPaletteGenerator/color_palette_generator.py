import openai
import os
from dotenv import load_dotenv, find_dotenv
import json

# Loading environment variables
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")


def get_colors_davinci(text):
    prompt = f"""
    Generate color palette from user Text input. The colors should fit the theme, mood or instructions in the Text input. The palettes should be between 2 and 8 colors.

    Q: Convert the following verbal description of a color palette into a list of colors: The Mediterranean Sea
    A: ["#006699", "#66CCCC", "#F0E68C", "#008000", "#F08080"]

    Q: Convert the following verbal description of a color palette into a list of colors: sage, nature, earth
    A: ["#EDF1D6", "#9DC08B", "#609966", "#40513B"]

    Desired Output Format: JSON array of hexadecimal color codes

    Q: {text}
    A:
    """

    openai.api_key = api_key
    response = openai.Completion.create(
        prompt = prompt,
        model="text-davinci-003",
        max_tokens=200,
        temperature=0.5
    )

    return json.loads(response["choices"][0]["text"])


def get_colors_gpt_turbo(text):
    
    messages = [
        {"role": "system", "content": "Generate color palette from user Text input. The colors should fit the theme, mood or instructions in the Text input. The palettes should be between 2 and 8 colors."},
        {"role": "user", "content": "Convert the following verbal description of a color palette into a list of colors: The Mediterranean Sea"},
        {"role": "assistant", "content": '["#006699", "#66CCCC", "#F0E68C", "#008000", "#F08080"]'},
        {"role": "user", "content": "Convert the following verbal description of a color palette into a list of colors: sage, nature, earth"},
        {"role": "assistant", "content": '["#EDF1D6", "#9DC08B", "#609966", "#40513B"]'},
        {"role": "user", "content": f"Convert the following verbal description of a color palette into a list of colors: {text}"}
    ]

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        messages = messages,
        model="gpt-3.5-turbo",
        max_tokens=200,
        temperature=0.3
    )
    
    return json.loads(response["choices"][0]["message"]["content"])