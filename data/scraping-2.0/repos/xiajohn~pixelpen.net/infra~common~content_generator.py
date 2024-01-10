import openai
from unidecode import unidecode
import os
# from PIL import Image, ImageDraw, ImageFont
import textwrap
import base64
import io
import requests, json
from utils import Utils
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('../.env'))
import time 
class ContentGenerator:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_text(self, prompt, max_tokens=2500, temperature=0.7, retries=3):
        for _ in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "When it comes to writing content, two factors are crucial, perplexity and burstiness. Perplexity measures the complexity of text. Separately, burstiness compares the variations of sentences. Humans tend to write with greater burstiness, for example, with some longer or complex sentences alongside shorter ones. AI sentences tend to be more uniform. Therefore, when writing the following content I am going to ask you to create, I need it to have a good amount of perplexity and burstiness.",
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }

                    ]
                )
                generated_text = response.choices[0].message.content
                clean_text = unidecode(generated_text)
                return clean_text
            except openai.error.APIError as e:
                if 'safety' in str(e):
                    prompt = f"Write a response on a safer version of the topic {prompt} without giving any options or lists."
                else:
                    raise e
                
            except openai.error.RateLimitError as e:
                wait_time = 3
                print(f"Rate limit exceeded, waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)

        raise Exception("Failed to generate text after multiple retries")

    def generate_image(self, prompt, retries=3):
        base64_image_data_list = []
        for _ in range(retries):
            try:
                response = openai.Image.create(
                    prompt=prompt, n=1, size="256x256", response_format="b64_json")
                for data_object in response['data']:
                    base64_image_data = data_object['b64_json']
                    base64_image_data_list.append(base64_image_data)

                if base64_image_data_list:
                    # If the list is not empty, the image was generated successfully
                    break
            except openai.error.APIError as e:
                if 'safety' in str(e):
                    # Modify the prompt to make it safer
                    prompt = f"Generate an image for a safer version of the topic {prompt}"
                else:
                    # If it's not a safety-related error, raise the exception
                    raise e
            except openai.error.RateLimitError as e:
                wait_time = 3
                print(f"Rate limit exceeded, waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
        else:
            raise Exception("Failed to generate image after multiple retries")

        # If you only need the first image data, you can return the first item in the list
        return base64_image_data_list[0]

    
    def get_embedding(self, text, model='text-embedding-ada-002'):
        url = 'https://api.openai.com/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
        }
        data = {
            'input': text,
            'model': model
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        embedding = response.json()['data'][0]['embedding']
        return embedding
    
    import json

    def get_topics_from_blog(self):
        """
        Get all the topics from the blog JSON data.

        Args:
        blog_json (str): The blog data in JSON format.

        Returns:
        list: A list of topics from the blog.
        """
        # Load the JSON data
        blog_data = Utils.load_json("user_input.json")

        # Get the topics
        topics = [item['topic'] for item in blog_data['website_building']['topics']]

        return topics

