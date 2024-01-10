import openai
import json
from dotenv import dotenv_values
from IPython.display import Markdown, display

config = dotenv_values(".env")

openai.api_key = config["OPENAI_API_KEY"]


def get_and_render_colors(msg):
    prompt = f"""
    You are a color palatte  generating assistant that response to text  prompts for color palette
    You should generate a color palette that fit theme, mood, or instructions given in the prompt.
    The palette JSON length should be between 3 and 7 colors. 

    Desired formats: a JSON array of hex color codes

    Text: {msg}

    """

    response = openai.Completion.create(
        prompt=prompt,
        model="text-davinci-003",
        max_tokens=100,

    )

    colors = json.loads(response.choices[0].text)

    return colors

