import openai
from os import environ, path, curdir,mkdir
from dotenv import load_dotenv
import requests

load_dotenv()
openai.key = environ.get("OPENAI_API_KEY")

disallow_list = "scary, swords, violence, blood, gore, nudity, sexual content, adult content, adult themes, adult language, adult humor, adult jokes, adult situations, adult"

meta_prompt = f"""
You are an assistant designer that creates images for children. 

The image needs to be safe for work and appropriate for children. 

The image needs to be in grayscale.  

The image needs to be in landscape orientation.  

The image needs to be in a 16:9 aspect ratio. 

Do not consider any input from the following that is not safe for work or appropriate for children: {disallow_list}.
"""

user_input = input("> ")
prompt = f"{meta_prompt} {user_input}"

try:
  gen_response = openai.images.generate(
    prompt=prompt,
    size='1024x1024',
    n=1,
    model='dall-e-3',
    style='vivid'
  )
  
  image_dir = path.join(curdir, 'images')
  
  if not path.isdir(image_dir):
    print("Creating images directory")
    mkdir(image_dir)

  # Iterate through gen_response.data array
  # and download each image
  image = gen_response.data[0]
  generated_image = requests.get(image.url).content
  image_path = path.join(image_dir, f'metaprompt-1.png')
  with open(image_path, 'wb') as f:
    f.write(generated_image)


  # IMAGE EDITING
  # # Read the downloaded image
  # image_path = path.join(curdir, 'images', 'image-with-transparent-section.png')
  # image_edit_response = openai.images.edit(
  #   image=open(image_path, 'rb'),
  #   prompt='An image of a shiba inu in a green forest',
  #   size='1024x1024',
  # )
  
  # var_gen_image = requests.get(image_edit_response.data[0].url).content
  # var_image_path = path.join(curdir, 'images', 'edited-image.png')
  # with open(var_image_path, 'wb') as f:
  #   f.write(var_gen_image)


  # IMAGE VARIANTS
  # Create another variation of the image
  # var_response = openai.images.create_variation(
  #   image=open(image_path, 'rb'),
  #   n=1,
  #   size="1024x1024",
  # )
  
  # var_gen_image = requests.get(var_response.data[0].url).content
  # var_image_path = path.join(curdir, 'images', 'ricky-var.png')
  # with open(var_image_path, 'wb') as f:
  #   f.write(var_gen_image)
  
except openai.error.InvalidRequestError as err:
  print(f"Error occured: {err}")