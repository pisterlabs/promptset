import os
import openai
import urllib.request

image_prompt_list = ["a labrador chilling on the beach with a cocktail in his paw", "a white cat chilling in Tokyo with a hamburger in his paw"]

def generate_image(image_prompt_list):
  for image_prompt in image_prompt_list:
    
    openai.api_key_path = "OPENAI_API_KEY.env"
    response = openai.Image.create(
      prompt=f"{image_prompt}",
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']
    save_image(image_url, image_prompt)

# saving image in specified path

def save_image(image_url, image_prompt):
  urllib.request.urlretrieve(f'{image_url}', f'images/{image_prompt}.png')
