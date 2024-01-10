import os
import openai

key = os.getenv("OPENAI_API_KEY")

#request an image creation
response = openai.Image.create(
  prompt="a giant purple highland cow with hawaiian tatoos on its horns walking in a lush green field and a lake in the back",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

#request an edit of an image
response = openai.Image.create_edit(
  image=open("sunlit_lounge.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A sunlit indoor lounge area with a pool containing a flamingo",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

#request image variations
response = openai.Image.create_variation(
  image=open("corgi_and_cat_paw.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']