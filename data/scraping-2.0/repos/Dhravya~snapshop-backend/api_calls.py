import httpx
from openai import OpenAI
from funcs import functions, other_func
import asyncify
from helpers.redis_helpers import get_user, create_generation
import asyncio
import json
from helpers.upload_image import upload_image
import requests

client = OpenAI()

async def ask_shopwise(item_name: str):
    print("Asking shopwise for " + item_name)

    response = requests.get(
        f"https://dropit2-production.up.railway.app/googleSearch?itemName={item_name}",
    )

    try:
        if response.status_code == 200:
            return response.json()
        else:
            print("ShopWise API failed")
            return None
    except Exception as e:
        print(e)
        return None


@asyncify
def get_fashion_image(base64_image: str, user_gender: str):
    image = upload_image(base64_image)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image? Describe it in as detail as possible. You are a fashion stylist. Be as descriptive about the fashion items as possible.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=250,
    )

    function_response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What’s in this image? Describe it in as detail as possible. You are a fashion stylist. Be as descriptive about the fashion items as possible. Here's the description of the image: {response.choices[0]}. The gender of the user is {user_gender}",
                    },
                ],
            }
        ],
        max_tokens=500,
        functions=functions,
        function_call="auto",
    )

    function_response = json.loads(
        function_response.choices[0].message.function_call.arguments
    )

    return function_response


@asyncify
def get_fashion_recommendation(user_context: str, user_gender: str):
    function_response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Create a fashion recommendation for the user. Here's the user's context: {user_context}. Keep in mind the gender of the user which is {user_gender}",
                    },
                ],
            }
        ],
        max_tokens=500,
        functions=other_func,
        function_call="auto",
    )

    function_response = json.loads(
        function_response.choices[0].message.function_call.arguments
    )

    return function_response


async def get_fashion_and_user_image(original_image: str, user_email: str):
    user = get_user(user_email)

    if user is None:
        raise Exception("User not found")

    output_json = await get_fashion_image(original_image, user.email)
    print(output_json)

    shopping_links = output_json["fashion_items_as_keywords"]
    print(shopping_links)

    shopping_links = await asyncio.gather(
        *[ask_shopwise(keyword) for keyword in shopping_links]
    )
    output_json["fashion_items_as_keywords"] = [link for link in shopping_links if link]
    output_json["original_image"] = upload_image(original_image)

    if not isinstance(output_json, dict):
        output_json = json.loads(output_json)

    print(output_json)

    try:
        create_generation(generated_json_output_as_dict=output_json)
    except Exception as e:
        print(e)

    return output_json


async def get_fashion_recommendation_with_shopping_links(
    user_context: str, user_email: str
):
    user = get_user(user_email)

    if user is None:
        raise Exception("User not found")

    output_json = await get_fashion_recommendation(user_context, user.email)
    print(output_json)

    shopping_links = output_json["fashion_items_as_keywords"]
    print(shopping_links)

    shopping_links = await asyncio.gather(
        *[ask_shopwise(keyword) for keyword in shopping_links]
    )
    output_json["fashion_items_as_keywords"] = [link for link in shopping_links if link]
    output_json["original_image"] = user_context

    if not isinstance(output_json, dict):
        output_json = json.loads(output_json)

    try:
        create_generation(generated_json_output_as_dict=output_json)
    except Exception as e:
        print(e)

    return output_json
