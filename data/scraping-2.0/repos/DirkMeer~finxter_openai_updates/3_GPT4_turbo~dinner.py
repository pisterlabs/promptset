from decouple import config
from openai import OpenAI
from pathlib import Path
import base64

client = OpenAI(api_key=config("OPENAI_API_KEY"))
get_path = lambda filename: Path(__file__).parent / filename


def image_encoder(image_path: str):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")


def dinnerGPT(image_name: str):
    image_path = get_path(image_name)
    base64_image = image_encoder(image_path)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a dinner AI. You will receive a picture of a fridge and recommend what the user can cook using the ingredients you see.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What can I cook from the stuff in my fridge?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer


# print(dinnerGPT("fridge.jpg"))


def dinnerGPTmulti(image_name_list: list[str]):
    messages = [
        {
            "role": "system",
            "content": "You are a dinner AI. You will receive one or more pictures of fridges/refrigerators/pantries/etc and recommend what the user can cook using the ingredients you see.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What can I cook from the stuff in my fridge?",
                },
                # This is where the images will be appended
            ],
        },
    ]

    image_path_list = [get_path(image_name) for image_name in image_name_list]
    base64_image_list = [image_encoder(image_path) for image_path in image_path_list]
    for image in base64_image_list:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                },
            },
        )

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    return answer


print(dinnerGPTmulti(["fridge.jpg", "fridge2.jpg"]))
