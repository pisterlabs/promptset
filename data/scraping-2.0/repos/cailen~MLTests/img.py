import os
import openai
from PIL import Image


openai.api_key = os.environ["OPENAI_API_KEY"]

# convert png rgb to png rgba
img = Image.open('IMG_4567.png')
img = img.convert('RGBA')
img = img.crop((0, 0, 1024, 1024))
img.save('IMG_4567.rgba.png')

# edit image with openai to convert the color of the house
response = openai.Image.create_variation(
    image=open("IMG_4567.rgba.png", "rb"),
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']

# print the response
print(image_url)
