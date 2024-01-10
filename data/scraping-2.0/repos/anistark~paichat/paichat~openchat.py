import openai
import webbrowser
from colorama import init as colorama_init
from colorama import Fore
from .config import get_config

def list_engines():
    # list engines
    engines = openai.Engine.list()

    # print the first engine's id
    print(engines.data[0].id)

def gpt_completion(prompt):
    openai.api_key = get_config()['OPENAI_API_KEY']
    # create a completion
    completion = openai.Completion.create(engine="text-davinci-002", prompt="Hello world")
    # print the completion
    print("GPT:", completion.choices[0].text)

def gpt_image(prompt):
    # create a completion
    image = openai.Image.create(prompt="Hello world", n=4, size="256x256")
    # print the completion
    print(f"{Fore.LIGHTGREEN_EX}GPT: Your image is opening in default web browser...")
    webbrowser.open(image.data[0].url)
