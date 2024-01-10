# %%
import openai
import os
import dotenv
from PIL import Image

dotenv.load_dotenv()

# API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# %%
#generation
response = openai.Image.create(
  prompt="a photo of a happy corgi puppy sitting and facing forward, studio light, longshot",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

# %%
#edit -- it is the same image but it need to remove some part and this part will be replaced, describe the image again with the all new image content
# Load and convert the input image to the supported format
input_image = Image.open("images/happy_corgi.png").convert("RGBA")
mask_image = Image.open("images/mask.png")
mask_image = mask_image.resize((1024, 1024))
mask_image = mask_image.convert("RGBA")

# Save the converted images to temporary files
input_image_path = "images/converted_happy_corgi.png"
mask_image_path = "images/converted_mask.png"
input_image.save(input_image_path)
mask_image.save(mask_image_path)


response = openai.Image.create_edit(
  image=open(input_image_path, "rb"),
  mask=open(mask_image_path, "rb"),
  prompt="a photo of a happy corgi puppy with fancy sunglasses on sitting and facing forward, studio light, longshot",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

# %%
#variations
response = openai.Image.create_variation(
  image=open("images/corgi_with_sunglasses.png", "rb"),
  n=3,
  size="1024x1024"
)
image_url = response['data']