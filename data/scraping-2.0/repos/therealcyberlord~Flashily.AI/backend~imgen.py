from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="a tree sloth in a blue hoodie, wearing black sony wireless headphones. He sips on a clear plastic cup of ice latte through a light blue straw. The tree sloth sits in high school styled chair with the table attached. There is a laptop on the high school styled chair, showing computer science lectures",
    size="1024x1024",
    quality="hd",
    n=1,
)

image_url = response.data[0].url
print(image_url)
