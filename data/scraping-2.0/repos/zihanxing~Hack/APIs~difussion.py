import requests
# You can access the image with PIL.Image for example
import io
from PIL import Image

import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')



def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=.5, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_imagePrompt(text):
    prompt = f"""
    Based on the following sentence, provide me a prompt for image generation. It is generated for kids. \
    example return: pixel art, a cute corgi, simple, flat colors \
    ```{text}```
    """
    response = get_completion(prompt)
    
    return response

def generate(prompt):

    API_URL = "https://api-inference.huggingface.co/models/nerijs/pixel-art-xl"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": prompt,
    })


    image = Image.open(io.BytesIO(image_bytes))
    return image

def main():

    line = \
    "\
    To make your papa happy, you can do some things that he likes. You can give him a big hug and tell him that you love him. You can also draw him a picture or make him a card to show him how much you care. Another thing you can do is help him with chores around the house, like picking up toys or setting the table. Spending time with him and doing things together, like playing games or going for a walk, can also make him happy. Remember, your papa loves you very much, so anything you do to show your love and kindness will make him happy!\
    "
    image_prompt = get_imagePrompt(line)

    print('====================image_prompt====================')
    print(image_prompt)
    print('====================================================')

    image = generate(image_prompt)
    # image.show()

    return image

if __name__ == '__main__':
    main()