from PIL import Image
import io
import os
import requests
import os
import openai
import requests
import json
from .azure_setup import upload_to_blob_storage
from .models import UserHistory, UserContext, User
from decouple import config

API_URL = "https://api-inference.huggingface.co/models/Adrenex/chamana"
headers = {"Authorization": config('HUGGING_FACE_TOKEN')}
openai.api_key = config('OPENAI_API_KEY')


context = [{'role': 'system', 'content': """
You are a NLP expert and do the following:
1) User will enter a prompt (to a text to image model), then the prompt is to be returned.
2) Then again, the user will be asked to enter another prompt, which will be in continuation of the first prompt. Then this needs to be merged appropriately with the first prompt, and new prompt will be returned.
3) This will go on and on as long as user inputs.
4) Make sure to Strictly return the new prompt only, nothing else, even without The first prompt is.

Example 1: 
First prompt: Red kurta
Return: Red Kurta

Second Prompt: Keep it long
Return: Red long kurta

Third Prompt: suitable for diwali occasion
Return: Red long kurta for Diwali

Forth Prompt: For kids
Display: Red long kurta for Kids and for Diwali

Return: Keep it short
Display: Red short kurta for Kids and for Diwali

Notice that, Short will replace Long. 

Example 2:
First prompt: Green Tshirt
Return: Green Tshirt

Second Prompt: make it red
Return: Red Tshirt

Notice that red replaces Green since both are colors
"""}]


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def collect_messages(prompt, id):
    global context  # userid from User table
    user = User.objects.get(id=id)
    context_entries = UserContext.objects.filter(user=user).first()
    print(context_entries, " ", user)  # <-------------------
    if context_entries is not None:
        print("hello", context_entries)
        context = context_entries.context
        context = json.loads(context.replace("'", "\""))
        context.append({'role': 'user', 'content': f"{prompt}"})
        response = get_completion_from_messages(context)
        context.append({'role': 'assistant', 'content': f"{response}"})
        print("\n", context)
        print(type(context))
        context_entries.context = context
        context_entries.save(update_fields=['context'])
    else:
        print("hell")
        print(context)
        context.append({'role': 'user', 'content': f"{prompt}"})
        response = get_completion_from_messages(context)
        context.append({'role': 'assistant', 'content': f"{response}"})
        user = User.objects.get(id=id)
        context_entries = UserContext.objects.create(
            user=user, context=context)
        print(context_entries)  # <-------------------
        context_entries.save()
    return response  # new prompt


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def text2image(prompt, id):
    print(prompt, id)      # <-------------------
    new_prompt = collect_messages(prompt, id)
    print(new_prompt, id)  # <-------------------
    image_bytes = query({
        "inputs": new_prompt,
    })

    image = Image.open(io.BytesIO(image_bytes))

    image_path = "test1.png"
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image file '{image_path}' deleted successfully.")

    image.save("test1.png", format="png")
    return upload_to_blob_storage(), new_prompt


def resetContext():
    context = [{'role': 'system', 'content': """
You are a NLP expert and do the following:
1) User will enter a prompt (to a text to image model), then the prompt is to be returned.
2) Then again, the user will be asked to enter another prompt, which will be in continuation of the first prompt. Then this needs to be merged appropriately with the first prompt, and new prompt will be returned.
3) This will go on and on as long as user inputs.
4) Make sure to Strictly return the new prompt only, nothing else, even without The first prompt is.

Example 1: 
First prompt: Red kurta
Return: Red Kurta

Second Prompt: Keep it long
Return: Red long kurta

Third Prompt: suitable for diwali occasion
Return: Red long kurta for Diwali

Forth Prompt: For kids
Display: Red long kurta for Kids and for Diwali

Return: Keep it short
Display: Red short kurta for Kids and for Diwali

Notice that, Short will replace Long. 

Example 2:
First prompt: Green Tshirt
Return: Green Tshirt

Second Prompt: make it red
Return: Red Tshirt

Notice that red replaces Green since both are colors
"""}]
