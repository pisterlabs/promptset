from openai import OpenAI
import os

# Set up the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",  # Use DALL-E 3 model
            prompt=prompt,
            n=1,  # Generate one image
            size="1024x1024"  # Specify image size
        )
        return response.data[0].url  # Return the URL of the generated image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
