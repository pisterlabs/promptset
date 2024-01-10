import os
import openai
from load_dotenv import load_dotenv


def load_env_variables():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


def get_prompt_image(prompt, n, size):
    return openai.Image.create(prompt=prompt, n=n, size=size)


def generate_img(prompt: str):
    load_env_variables()
    response = get_prompt_image(prompt, n=1, size="256x256")
    print(response["data"][0]["url"])
    return response["data"][0]["url"]


if __name__ == "__main__":
    prompt = "An eco-friendly computer from the 90s in the style of vaporwave"
    generate_img(prompt)
