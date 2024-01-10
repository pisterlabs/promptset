import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key)

def edit_image(image_path, mask_path, prompt, size="1024x1024", n=1):
    with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
        response = client.images.edit(
            model="dall-e-2",
            image=image_file,
            mask=mask_file,
            prompt=prompt,
            n=n,
            size=size
        )
    return response.data[0].url

if __name__ == "__main__":
    image_path = "sunlit_lounge.png"
    mask_path = "mask.png"
    prompt = "A sunlit indoor lounge area with a pool containing a flamingo"
    print(edit_image(image_path, mask_path, prompt))
