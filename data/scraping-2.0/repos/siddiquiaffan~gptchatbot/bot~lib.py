import openai

from .constant import Env

openai.api_key = Env.OPENAI_API_KEY


def get_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        stop=["You:", "AI:"],
    )
    return response["choices"][0]["text"]


def get_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        # model=MODEL_ENGINE,
        n=1,
        size="1024x1024",
    )
    img_url = response["data"][0]["url"]
    return img_url
