# image_generator/text_to_image.py
import openai
from config.settings import OPENAI_API_KEY

class TextToImage:
    def __init__(self):
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate_image(self, prompt, size="1024x1024", quality="standard"):
        try:
            # Call the DALL-E 3 API to generate an image
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            # Extract the URL of the generated image
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            print(f"An error occurred while generating the image: {e}")
            return None

if __name__ == "__main__":
    # For testing purposes
    image_generator = TextToImage()
    prompt = "A scenic view of mountains during sunset"
    image_url = image_generator.generate_image(prompt)
    print(f"Generated Image URL: {image_url}")
