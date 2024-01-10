import openai
from bot.config import OPENAI_KEY

openai.api_key = OPENAI_KEY


def openia_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user",
                   "content": prompt}],
    )
    return response.choices[0].message.content


def openia_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
    )
    return response["data"][0]["url"]
