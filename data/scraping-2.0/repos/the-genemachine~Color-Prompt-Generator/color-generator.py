import openai
import json
import tkinter

from dotenv import dotenv_values

config = dotenv_values(".env")

openai.api_key = config["OPENAI_API_KEY"]

# This is a multiline string that describes the task to be completed by the OpenAI model.
prompt = """
You are a color palette generating assitant that responds to text prompts for color palettes

You should generate color palettes that are aesthetically pleasing and match the text prompt

You should generate 5 colors per palette

Desired Format: a JSON array of hex color codes

Text : A beautiful sunset

Result : ["#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]
"""

# This line sends the prompt to the OpenAI API and stores the response
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=200
)

# This line prints out the actual generated text which is stored under the choices key in the response
print(response["choices"][0] ["text"])