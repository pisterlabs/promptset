# import openai
from openai import OpenAI
from PIL import Image
import urllib.request
from io import BytesIO
from IPython.display import display
# import openai


# Get OpenAI key
open_ai_key_file = "api-key.txt"
with open(open_ai_key_file, "r") as f:
  for line in f:
    OPENAI_KEY = line
    break

# openai.api_key = OPENAI_KEY
client = OpenAI(api_key=OPENAI_KEY)

# Get list of flower species
list_of_flowers = []
with open("species.txt", "r") as f:
  for line in f:
    list_of_flowers.append(line.strip().lower())

# client = OpenAI()

# Store the URLs generated for each photo
url_list = []
# How many photos at a time, I believe we can only do 5 per minute so we should split up this task
batch_size = 1

# for x in list_of_flowers:
response = client.images.generate(
    model="dall-e-2",
    prompt="a black parrot",
    size="256x256",
    quality="standard",
    n=batch_size,
    style="natural"
)
url_list.extend([obj.url for obj in response.data])

print(url_list)

image_url = response.data[0].url
with urllib.request.urlopen(image_url) as url:
  image = Image.open(BytesIO(url.read()))

display(image)